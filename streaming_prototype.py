"""
Streaming (LocalAgreement-2) prototype over faster-whisper — STANDALONE.

Push-to-talk: hold F8 to talk, release to finalize, ESC to quit.
Does NOT touch app.py. Purpose: measure streaming latency/quality on THIS
hardware before deciding whether to fold streaming into the real PTT flow.

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

Run it yourself (it needs the mic + your keyboard):
    ! python streaming_prototype.py            # uses model from config.json
    ! python streaming_prototype.py small      # force a model size
"""
import re
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

SAMPLE_RATE = 16000
MIN_CHUNK_S = 1.0        # minimum new audio before running a streaming pass
TRIM_BUFFER_S = 20.0     # trim committed audio once the buffer passes this
STREAM_BEAM = 1          # greedy during streaming (speed)
FULL_BEAM = 5            # matches app.py's one-shot decode (quality baseline)

PTT_KEY = keyboard.Key.f8

PROMPTS = {
    "es": "Hola, ¿cómo estás? Esto es un texto de ejemplo con comas, puntos y mayúsculas.",
    "en": "Hello, how are you? This is an example text with commas, periods, and capitalization.",
}


def _norm(word: str) -> str:
    """Normalize a token for agreement comparison (ignore case/punctuation)."""
    return re.sub(r"[^\w]", "", word.strip().lower(), flags=re.UNICODE)


def _join(words) -> str:
    """faster-whisper word tokens already carry a leading space; join raw."""
    return re.sub(r"\s+", " ", "".join(w[2] for w in words)).strip()


class HypothesisBuffer:
    """
    LocalAgreement-2 committer (after ufal/whisper_streaming).

    Holds the previous pass's uncommitted hypothesis; on each new pass it
    commits the longest common prefix of words that agree with the previous
    pass. Words are (start_abs, end_abs, text) in absolute stream time.
    """

    def __init__(self):
        self.buffer = []              # previous pass, still tentative
        self.last_committed_time = 0.0

    def insert(self, words, offset):
        # Shift buffer-local times to absolute; drop anything we already passed.
        shifted = [(s + offset, e + offset, w) for (s, e, w) in words]
        return [t for t in shifted if t[0] > self.last_committed_time - 0.1]

    def flush(self, new_words):
        """Commit the agreeing prefix between `new_words` and the prev pass."""
        committed = []
        new = list(new_words)
        while new and self.buffer:
            if _norm(new[0][2]) == _norm(self.buffer[0][2]):
                committed.append(new[0])
                self.last_committed_time = new[0][1]
                new.pop(0)
                self.buffer.pop(0)
            else:
                break
        self.buffer = new           # remainder becomes next pass's reference
        return committed


class OnlineASR:
    """Rolling-buffer streaming wrapper around a faster-whisper model."""

    def __init__(self, model, language):
        self.model = model
        self.language = language
        self.base_prompt = PROMPTS.get(language or "es", PROMPTS["es"])
        self.audio = np.array([], dtype=np.float32)
        self.time_offset = 0.0        # seconds trimmed off the front
        self.hyp = HypothesisBuffer()
        self.committed = []           # list of (start_abs, end_abs, text)

    def insert_audio(self, chunk):
        self.audio = np.concatenate([self.audio, chunk])

    def _prompt(self):
        tail = _join(self.committed)[-200:]
        return f"{self.base_prompt} {tail}".strip()

    def _transcribe(self):
        if len(self.audio) < int(0.2 * SAMPLE_RATE):
            return []
        segments, _ = self.model.transcribe(
            self.audio,
            language=self.language,
            beam_size=STREAM_BEAM,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=self._prompt(),
            # VAD ON even on partial buffers: strips the trailing silence so
            # Whisper doesn't hallucinate long token runs over it (that was
            # blowing up decode time). Stable words still come from LocalAgreement.
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        words = []
        for seg in segments:
            for w in (seg.words or []):
                words.append((w.start, w.end, w.word))
        return words

    def _maybe_trim(self):
        if len(self.audio) / SAMPLE_RATE <= TRIM_BUFFER_S or not self.committed:
            return
        last_end_abs = self.committed[-1][1]
        cut_local = max(0.0, (last_end_abs - self.time_offset) - 1.0)  # keep 1s ctx
        cut = int(cut_local * SAMPLE_RATE)
        if 0 < cut < len(self.audio):
            self.audio = self.audio[cut:]
            self.time_offset += cut / SAMPLE_RATE

    def process_iter(self):
        """Run one streaming pass; return the words newly committed this pass."""
        words = self.hyp.insert(self._transcribe(), self.time_offset)
        committed = self.hyp.flush(words)
        self.committed.extend(committed)
        self._maybe_trim()
        return committed

    def finish(self):
        """Final flush on release: commit the remaining tentative tail too."""
        words = self.hyp.insert(self._transcribe(), self.time_offset)
        self.committed.extend(self.hyp.flush(words))
        self.committed.extend(self.hyp.buffer)   # accept the last tail as final
        self.hyp.buffer = []
        return _join(self.committed)

    def tentative_tail(self):
        return _join(self.hyp.buffer)


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
        self.online = OnlineASR(self.model, self.language)
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
        committed = _join(self.online.committed)
        tail = self.online.tentative_tail()
        buf_s = len(self.online.audio) / SAMPLE_RATE
        line = f"\r🟢 {committed} ⟨{tail}⟩  [pass {dt:.2f}s · buf {buf_s:4.1f}s]"
        sys.stdout.write(line[:200].ljust(200))
        sys.stdout.flush()

    # ---- finalize + report ----
    def _finalize(self):
        t0 = time.time()
        stream_text = self.online.finish()
        stream_wait = time.time() - self.t_release

        full_audio = (np.concatenate(self.full_chunks)
                      if self.full_chunks else np.zeros(1, dtype=np.float32))
        rec_dur = self.t_release - self.t_start

        tf = time.time()
        segments, _ = self.model.transcribe(
            full_audio,
            language=self.language,
            beam_size=FULL_BEAM,
            initial_prompt=PROMPTS.get(self.language or "es", PROMPTS["es"]),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )
        full_text = "".join(s.text for s in segments).strip()
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
