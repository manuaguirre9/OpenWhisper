"""
Streaming core (LocalAgreement-2) — pure logic, no audio/keyboard I/O.

Extracted verbatim from streaming_prototype.py so that three consumers can
share ONE implementation instead of drifting apart:
  - streaming_prototype.py  (live mic, push-to-talk)
  - benchmark/bench_dictation.py (offline, reproducible measurement)
  - app.py, if/when streaming gets folded into the real PTT flow

Nothing here imports sounddevice or pynput, so it is importable headless
(CI, a benchmark run over a WAV file, unit tests).

How it works
------------
While audio accumulates, Whisper re-transcribes the WHOLE rolling buffer
(so it always has context) with word timestamps. LocalAgreement-2 commits
only the words that stayed identical across the last two passes; the
unstable tail is re-decoded next pass. Once the buffer grows past
TRIM_BUFFER_S, already-committed audio is trimmed off (keeping
KEEP_CONTEXT_S of left context), so cost stays bounded on long dictations.
"""
import re
from typing import Callable, List, Optional, Tuple

import numpy as np

# --- audio ---
SAMPLE_RATE = 16000

# --- streaming schedule ---
MIN_CHUNK_S = 1.0        # minimum new audio before running a streaming pass
TRIM_BUFFER_S = 20.0     # trim committed audio once the buffer passes this
KEEP_CONTEXT_S = 1.0     # left context kept after a trim
MIN_DECODE_S = 0.2       # skip a pass on buffers shorter than this

# --- decoding ---
STREAM_BEAM = 1          # greedy during streaming (speed)
FULL_BEAM = 5            # matches app.py's one-shot decode (quality baseline)
STREAM_VAD_SILENCE_MS = 300
FULL_VAD_SILENCE_MS = 500
PROMPT_TAIL_CHARS = 200  # how much committed text is fed back as prompt

# Initial prompts per language. They bias the model towards proper punctuation
# and capitalization. Using the prompt in the wrong language hurts accuracy,
# so we pick one per detected/forced language.
# NOTE: transcription_engine.py still keeps its own copy of this dict. If you
# touch one, touch both — or better, make that module import from here.
PROMPTS = {
    "es": "Hola, ¿cómo estás? Esto es un texto de ejemplo con comas, puntos y mayúsculas.",
    "en": "Hello, how are you? This is an example text with commas, periods, and capitalization.",
}

Word = Tuple[float, float, str]   # (start_abs, end_abs, text)


def base_prompt_for(language: Optional[str]) -> str:
    """Pick the punctuation-biasing prompt for a language ('es' when unknown)."""
    return PROMPTS.get(language or "es", PROMPTS["es"])


def norm_word(word: str) -> str:
    """Normalize a token for agreement comparison (ignore case/punctuation)."""
    return re.sub(r"[^\w]", "", word.strip().lower(), flags=re.UNICODE)


def join_words(words) -> str:
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
        self.buffer: List[Word] = []      # previous pass, still tentative
        self.last_committed_time = 0.0

    def insert(self, words, offset: float) -> List[Word]:
        # Shift buffer-local times to absolute; drop anything we already passed.
        shifted = [(s + offset, e + offset, w) for (s, e, w) in words]
        return [t for t in shifted if t[0] > self.last_committed_time - 0.1]

    def flush(self, new_words) -> List[Word]:
        """Commit the agreeing prefix between `new_words` and the prev pass."""
        committed: List[Word] = []
        new = list(new_words)
        while new and self.buffer:
            if norm_word(new[0][2]) == norm_word(self.buffer[0][2]):
                committed.append(new[0])
                self.last_committed_time = new[0][1]
                new.pop(0)
                self.buffer.pop(0)
            else:
                break
        self.buffer = new           # remainder becomes next pass's reference
        return committed


class OnlineASR:
    """
    Rolling-buffer streaming wrapper around a faster-whisper model.

    on_error: optional callable(exception) invoked when a decoding pass
    raises. The pass is then treated as producing no words and the loop
    continues — a single bad pass must never kill the dictation thread,
    which would silently freeze the transcript on its last partial.
    (A pass that HANGS instead of raising cannot be cancelled: CTranslate2
    offers no interrupt. `slow_passes` counts passes over the budget so a
    caller can at least notice and report it.)
    """

    def __init__(
        self,
        model,
        language: Optional[str],
        beam_size: int = STREAM_BEAM,
        trim_buffer_s: float = TRIM_BUFFER_S,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ):
        self.model = model
        self.language = language
        self.beam_size = beam_size
        self.trim_buffer_s = trim_buffer_s
        self.on_error = on_error
        self.base_prompt = base_prompt_for(language)
        self.audio = np.array([], dtype=np.float32)
        self.time_offset = 0.0        # seconds trimmed off the front
        self.hyp = HypothesisBuffer()
        self.committed: List[Word] = []
        self.failed_passes = 0
        self.slow_passes = 0

    # --- input ---
    def insert_audio(self, chunk: np.ndarray):
        self.audio = np.concatenate([self.audio, chunk])

    @property
    def buffer_seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    # --- decoding ---
    def _prompt(self) -> str:
        tail = join_words(self.committed)[-PROMPT_TAIL_CHARS:]
        return f"{self.base_prompt} {tail}".strip()

    def _transcribe(self) -> List[Word]:
        if len(self.audio) < int(MIN_DECODE_S * SAMPLE_RATE):
            return []
        try:
            segments, _ = self.model.transcribe(
                self.audio,
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=self._prompt(),
                # VAD ON even on partial buffers: strips the trailing silence so
                # Whisper doesn't hallucinate long token runs over it (that was
                # blowing up decode time). Stable words still come from
                # LocalAgreement.
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=STREAM_VAD_SILENCE_MS),
            )
            words: List[Word] = []
            for seg in segments:
                for w in (seg.words or []):
                    words.append((w.start, w.end, w.word))
            return words
        except Exception as exc:  # noqa: BLE001 - one bad pass must not kill the loop
            self.failed_passes += 1
            if self.on_error is not None:
                self.on_error(exc)
            return []

    # --- buffer management ---
    def _maybe_trim(self):
        if self.buffer_seconds <= self.trim_buffer_s or not self.committed:
            return
        last_end_abs = self.committed[-1][1]
        cut_local = max(0.0, (last_end_abs - self.time_offset) - KEEP_CONTEXT_S)
        cut = int(cut_local * SAMPLE_RATE)
        if 0 < cut < len(self.audio):
            self.audio = self.audio[cut:]
            self.time_offset += cut / SAMPLE_RATE

    # --- passes ---
    def process_iter(self) -> List[Word]:
        """Run one streaming pass; return the words newly committed this pass."""
        words = self.hyp.insert(self._transcribe(), self.time_offset)
        committed = self.hyp.flush(words)
        self.committed.extend(committed)
        self._maybe_trim()
        return committed

    def finish(self) -> str:
        """Final flush on release: commit the remaining tentative tail too."""
        words = self.hyp.insert(self._transcribe(), self.time_offset)
        self.committed.extend(self.hyp.flush(words))
        self.committed.extend(self.hyp.buffer)   # accept the last tail as final
        self.hyp.buffer = []
        return join_words(self.committed)

    # --- readouts ---
    def committed_text(self) -> str:
        return join_words(self.committed)

    def tentative_tail(self) -> str:
        return join_words(self.hyp.buffer)


def transcribe_one_shot(model, audio: np.ndarray, language: Optional[str],
                        beam_size: int = FULL_BEAM) -> str:
    """
    The decode app.py does today: one pass over the whole recording,
    beam=5 + VAD. This is the quality/latency baseline streaming has to beat
    on the wait-after-release.
    """
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        initial_prompt=base_prompt_for(language),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=FULL_VAD_SILENCE_MS),
        condition_on_previous_text=False,
    )
    return "".join(s.text for s in segments).strip()
