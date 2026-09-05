"""
Streaming (LocalAgreement-2) prototype over faster-whisper — STANDALONE.

Push-to-talk: hold F8 to talk, release to finalize, ESC to quit.
Does NOT touch app.py. Purpose: measure streaming latency/quality on THIS
hardware before deciding whether to fold streaming into the real PTT flow.

The streaming algorithm itself now lives in streaming_core.py (shared with
benchmark/bench_dictation.py, which runs the same code over a WAV file with
no mic and no human). This file is just the live push-to-talk driver.

How it works (the "context + streaming" reconciliation you asked about):
  - While you hold F8, audio accumulates in a rolling buffer.
  - Every ~1s of new audio, Whisper re-transcribes the WHOLE buffer (so it
    always has context) with word timestamps.
  - LocalAgreement-2 commits only the words that stayed identical across the
    last two passes. The unstable tail is re-decoded next pass.
  - Once the buffer grows past TRIM_BUFFER_S, the already-committed audio is
    trimmed off (keeping ~1s of left context), so cost stays bounded even on
    a long dictation / a Raspberry Pi.

On release it prints two finalizations so you can compare:
  - STREAM: committed prefix (fixed during the hold) + one final tail flush.
            Uses beam_size=1 (greedy) for speed.
  - FULL:   one-shot transcribe of the entire recording, beam_size=5 + VAD —
            i.e. exactly what app.py does today. This is the quality/latency
            baseline you're trying to beat on the "wait after release".

For a reproducible measurement over a fixed audio file (no mic, no human,
WER included), use `python benchmark/bench_dictation.py` instead.

Run it yourself (it needs the mic + your keyboard):
    ! python streaming_prototype.py            # uses model from config.json
    ! python streaming_prototype.py small      # force a model size
"""
import sys
import time
import queue
import threading

import numpy as np
import sounddevice as sd
from pynput import keyboard

from faster_whisper import WhisperModel
from config_manager import load_config
from system_info import physical_core_count
from streaming_core import (
    SAMPLE_RATE,
    MIN_CHUNK_S,
    FULL_BEAM,
    OnlineASR,
    transcribe_one_shot,
)

PTT_KEY = keyboard.Key.f8


class Session:
    def __init__(self, model, language, device):
        self.model = model
        self.language = language
        self.device = device
        self.audio_q = queue.Queue()
        self.recording = False
        self.stream = None

    # ---- audio ----
    def _callback(self, indata, frames, t, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        if self.recording:
            self.audio_q.put(indata.copy().flatten())

    def start(self):
        self.recording = True
        self.online = OnlineASR(
            self.model, self.language,
            on_error=lambda e: print(f"\n[pass falló] {e}", file=sys.stderr),
        )
        self.full_chunks = []
        self.pass_times = []
        self.commit_lat = []
        with self.audio_q.mutex:
            self.audio_q.queue.clear()
        self.t_start = time.time()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=self.device, callback=self._callback,
        )
        self.stream.start()
        print("\n🔴 grabando… (soltá F8 para finalizar)")
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        self.t_release = time.time()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

    # ---- streaming loop ----
    def _loop(self):
        since_last = 0
        while True:
            try:
                chunk = self.audio_q.get(timeout=0.1)
                self.online.insert_audio(chunk)
                self.full_chunks.append(chunk)
                since_last += len(chunk)
            except queue.Empty:
                pass
            if since_last >= MIN_CHUNK_S * SAMPLE_RATE:
                since_last = 0
                self._run_pass()
            if not self.recording and self.audio_q.empty():
                break
        self._finalize()

    def _run_pass(self):
        t0 = time.time()
        committed = self.online.process_iter()
        dt = time.time() - t0
        self.pass_times.append(dt)
        now = time.time()
        for (_s, e_abs, _w) in committed:
            self.commit_lat.append((now - self.t_start) - e_abs)
        self._render(dt)

    def _render(self, dt):
        committed = self.online.committed_text()
        tail = self.online.tentative_tail()
        buf_s = self.online.buffer_seconds
        line = f"\r🟢 {committed} ⟨{tail}⟩  [pass {dt:.2f}s · buf {buf_s:4.1f}s]"
        sys.stdout.write(line[:200].ljust(200))
        sys.stdout.flush()

    # ---- finalize + report ----
    def _finalize(self):
        stream_text = self.online.finish()
        stream_wait = time.time() - self.t_release

        full_audio = (np.concatenate(self.full_chunks)
                      if self.full_chunks else np.zeros(1, dtype=np.float32))
        rec_dur = self.t_release - self.t_start

        tf = time.time()
        full_text = transcribe_one_shot(self.model, full_audio, self.language,
                                        beam_size=FULL_BEAM)
        full_wait = time.time() - tf

        passes = self.pass_times
        avg_pass = sum(passes) / len(passes) if passes else 0.0
        max_pass = max(passes) if passes else 0.0
        avg_lat = sum(self.commit_lat) / len(self.commit_lat) if self.commit_lat else 0.0
        max_lat = max(self.commit_lat) if self.commit_lat else 0.0
        kept_up = "SÍ" if max_pass <= MIN_CHUNK_S else "NO (se atrasa)"

        print("\n" + "=" * 68)
        print(f"  duración grabada : {rec_dur:5.1f}s")
        print(f"  pasadas streaming: {len(passes)}  (avg {avg_pass:.2f}s · max {max_pass:.2f}s)")
        print(f"  ¿le gana al habla?: {kept_up}   (pasada debe ser < {MIN_CHUNK_S:.0f}s)")
        print(f"  latencia commit  : avg {avg_lat:.2f}s · max {max_lat:.2f}s  (qué tan atrás del vivo)")
        if self.online.failed_passes:
            print(f"  pasadas fallidas : {self.online.failed_passes}")
        print("-" * 68)
        print(f"  ESPERA tras soltar → STREAM: {stream_wait:5.2f}s   |   FULL (hoy): {full_wait:5.2f}s")
        print("-" * 68)
        print(f"  STREAM (beam=1): {stream_text}")
        print(f"  FULL   (beam=5): {full_text}")
        print("=" * 68)
        print("\nMantené F8 para otra prueba, ESC para salir.")


def main():
    cfg = load_config()
    # Default to 'small' (streaming-appropriate), NOT the config's model (medium
    # is too heavy to keep up here). Override: `python streaming_prototype.py base`.
    model_size = sys.argv[1] if len(sys.argv) > 1 else "small"
    language = cfg.get("language", "es")
    if language == "auto":
        language = None
    mic = cfg.get("microphone", "default")
    device = None if mic == "default" else int(mic)
    # Ignore config's cpu_threads (2) for streaming — use all physical cores.
    # Override with a 2nd CLI arg: `python streaming_prototype.py small 4`.
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else physical_core_count()

    print(f"Cargando modelo '{model_size}' (lang={language}, cpu_threads={threads})…")
    print("  [streaming usa núcleos FÍSICOS, ignora el cpu_threads=2 del config]")
    model = WhisperModel(model_size, device="auto", compute_type="int8",
                         cpu_threads=threads, num_workers=1)
    print("Modelo listo.")

    session = Session(model, language, device)

    print(f"\n▶  Mantené {PTT_KEY} para hablar, soltá para transcribir. ESC para salir.")

    def on_press(key):
        if key == PTT_KEY and not session.recording:
            session.start()

    def on_release(key):
        if key == PTT_KEY:
            session.stop()
        elif key == keyboard.Key.esc:
            return False   # stops the listener

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
