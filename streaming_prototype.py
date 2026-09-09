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
    ! python streaming_prototype.py                 # small, physical cores
    ! python streaming_prototype.py base            # force a model size
    ! python streaming_prototype.py small 3         # ...and cpu_threads
    ! python streaming_prototype.py small --trim 8  # shorter rolling buffer

--trim is the knob that matters on the Pi: every pass re-transcribes the
WHOLE buffer, so a 20s buffer costs 20s of audio per pass. Trimming to 8s
bounds that. It is a flag and not a new default because Windows optimizes
for dictation quality and Nito for latency — different profiles, same code.
"""
import argparse
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
    TRIM_BUFFER_S,
    FULL_BEAM,
    OnlineASR,
    transcribe_one_shot,
)

PTT_KEY = keyboard.Key.f8


class Session:
    def __init__(self, model, language, device, trim_buffer_s=TRIM_BUFFER_S):
        self.model = model
        self.language = language
        self.device = device
        self.trim_buffer_s = trim_buffer_s
        self.audio_q = queue.Queue()
        self.recording = False
        self.stream = None
        # True from start() until _finalize() returns. start() refuses to
        # re-enter while set: the report prints after the release, and an
        # impatient second F8 would otherwise swap self.online out from
        # under the thread still finalizing the previous take.
        self.busy = False

    # ---- audio ----
    def _callback(self, indata, frames, t, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        if self.recording:
            self.audio_q.put(indata.copy().flatten())

    def start(self):
        if self.busy:
            return
        self.busy = True
        self.recording = True
        self.online = OnlineASR(
            self.model, self.language,
            trim_buffer_s=self.trim_buffer_s,
            on_error=lambda e: print(f"\n[pass falló] {e}", file=sys.stderr),
        )
        self.full_chunks = []
        self.pass_times = []
        self.commit_lat = []

        # Audio drenado de la cola pero todavía no entregado a OnlineASR.
        # El hilo productor escribe acá; el de pasadas lo vacía de un saque
        # justo antes de transcribir. OnlineASR sigue viéndose a sí mismo
        # como single-threaded, que es por qué streaming_core no se toca.
        self._pending = []
        self._pending_samples = 0
        self._pending_lock = threading.Lock()
        self._drained = threading.Event()

        with self.audio_q.mutex:
            self.audio_q.queue.clear()
        self.t_start = time.time()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=self.device, callback=self._callback,
        )
        self.stream.start()
        print("\n🔴 grabando… (soltá F8 para finalizar)")
        threading.Thread(target=self._drain_loop, daemon=True).start()
        threading.Thread(target=self._pass_loop, daemon=True).start()

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

    # ---- streaming loops ----
    # Antes esto era un solo hilo: sacaba un chunk de la cola y después se
    # bloqueaba adentro de _run_pass() por toda la pasada. Durante ese rato
    # nadie drenaba audio_q. La cola no tiene tope así que no se perdía
    # audio, pero la cadencia de pasadas quedaba irregular — y en la Pi,
    # donde una pasada puede pasarse de MIN_CHUNK_S, el atraso se acumula en
    # vez de corregirse. Ahora un hilo sólo drena y otro sólo transcribe.

    def _drain_loop(self):
        """Productor: saca audio de la cola. No transcribe nunca."""
        while True:
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                if not self.recording:
                    break
                continue
            with self._pending_lock:
                self._pending.append(chunk)
                self._pending_samples += len(chunk)
            self.full_chunks.append(chunk)
        self._drained.set()

    def _take_pending(self):
        """Vacía el buffer de entrega y devuelve (chunks, cantidad_de_samples)."""
        with self._pending_lock:
            chunks, self._pending = self._pending, []
            n, self._pending_samples = self._pending_samples, 0
        return chunks, n

    def _pass_loop(self):
        """Consumidor: corre las pasadas de Whisper. No toca la cola."""
        since_last = 0
        while True:
            chunks, n = self._take_pending()
            for c in chunks:
                self.online.insert_audio(c)
            since_last += n

            if since_last >= MIN_CHUNK_S * SAMPLE_RATE:
                since_last = 0
                self._run_pass()
                continue

            if self._drained.is_set():
                # El productor ya salió, así que no puede llegar nada nuevo.
                # Absorbé lo que haya entrado entre el take de arriba y esto.
                chunks, _ = self._take_pending()
                for c in chunks:
                    self.online.insert_audio(c)
                break

            time.sleep(0.02)
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
        print(f"  config           : trim {self.trim_buffer_s:.0f}s")
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
        self.busy = False


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Prototipo de dictado streaming (LocalAgreement-2).",
    )
    # Posicionales, para no romper las invocaciones que ya estaban documentadas.
    p.add_argument("model", nargs="?", default="small",
                   help="tamaño del modelo Whisper (default: small)")
    p.add_argument("threads", nargs="?", type=int, default=None,
                   help="cpu_threads (default: núcleos físicos)")
    p.add_argument("--trim", type=float, default=TRIM_BUFFER_S, metavar="SEG",
                   help=f"segundos de buffer antes de recortar "
                        f"(default: {TRIM_BUFFER_S:.0f}; en la Pi probá 8)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    cfg = load_config()
    # Default to 'small' (streaming-appropriate), NOT the config's model (medium
    # is too heavy to keep up here).
    model_size = args.model
    language = cfg.get("language", "es")
    if language == "auto":
        language = None
    mic = cfg.get("microphone", "default")
    device = None if mic == "default" else int(mic)
    # Ignore config's cpu_threads (2) for streaming — use all physical cores.
    threads = args.threads if args.threads is not None else physical_core_count()

    print(f"Cargando modelo '{model_size}' (lang={language}, cpu_threads={threads}, "
          f"trim={args.trim:.0f}s)…")
    print("  [streaming usa núcleos FÍSICOS, ignora el cpu_threads=2 del config]")
    model = WhisperModel(model_size, device="auto", compute_type="int8",
                         cpu_threads=threads, num_workers=1)
    print("Modelo listo.")

    session = Session(model, language, device, trim_buffer_s=args.trim)

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
