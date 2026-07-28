import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from faster_whisper import WhisperModel
import numpy as np

from system_info import resolve_cpu_threads

# Initial prompts per language. They bias the model towards proper punctuation
# and capitalization. Using the prompt in the wrong language hurts accuracy,
# so we pick one per detected/forced language.
PROMPTS = {
    "es": "Hola, ¿cómo estás? Esto es un texto de ejemplo con comas, puntos y mayúsculas.",
    "en": "Hello, how are you? This is an example text with commas, periods, and capitalization.",
}

# HuggingFace repos used by faster-whisper for each model size.
_HF_REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# Approximate total download size per model (MB). Used to fake a percentage
# from the cache directory's growing byte count.
_EXPECTED_DOWNLOAD_MB = {
    "tiny": 75,
    "base": 145,
    "small": 470,
    "medium": 1500,
    "large-v3": 3050,
}


def _model_cache_dir(model_size: str) -> Path:
    repo_id = _HF_REPO_MAP.get(model_size, f"Systran/faster-whisper-{model_size}")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_root / ("models--" + repo_id.replace("/", "--"))


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def _prefetch_with_progress(model_size: str,
                             progress_cb: Callable[[float, float], None]):
    """
    Download the Whisper model files via huggingface_hub.snapshot_download
    while a polling thread reports `progress_cb(downloaded_mb, total_mb)`.

    If the model is already fully cached, returns instantly without calling
    progress_cb (no download = no progress to show).
    """
    repo_id = _HF_REPO_MAP.get(model_size, f"Systran/faster-whisper-{model_size}")
    cache_dir = _model_cache_dir(model_size)
    expected_mb = float(_EXPECTED_DOWNLOAD_MB.get(model_size, 200))

    initial_mb = _dir_size_mb(cache_dir)
    if initial_mb >= expected_mb * 0.95:
        # Already cached.
        return

    stop_event = threading.Event()

    def _poll():
        while not stop_event.is_set():
            try:
                current = _dir_size_mb(cache_dir)
                progress_cb(current, expected_mb)
            except Exception:
                pass
            time.sleep(0.3)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id)
        # Flush a final 100% so the UI doesn't sit at 99%.
        try:
            progress_cb(expected_mb, expected_mb)
        except Exception:
            pass
    finally:
        stop_event.set()
        poller.join(timeout=1.0)


class Transcriber:
    def __init__(
        self,
        model_size="base",
        device="auto",
        compute_type="int8",
        cpu_threads=0,
        vocabulary="",
        beam_size=1,
        progress_cb: Optional[Callable[[float, float], None]] = None,
    ):
        """
        Initializes the faster-whisper model.
        - model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
        - device: 'auto' (chooses CUDA if available, else CPU)
        - compute_type: 'int8' is the safest cross-platform CPU default.
                        'float16' is better for CUDA GPUs.
        - cpu_threads: 0 = auto = physical core count (best for CTranslate2,
                       avoids SMT oversubscription). Positive values are
                       clamped to the logical core count. See system_info.
        - vocabulary: free-form string of names/jargon/technical terms to bias
                      the model towards (appended to the initial prompt).
        - beam_size: decoder beam width for dictation. 1 (greedy) is much
                     faster with minimal quality loss on short/medium clips;
                     raise to 5 for max accuracy at the cost of latency.
        - progress_cb: optional callable(downloaded_mb, total_mb) used to
                       report download progress on first-time model fetch.
                       Skipped when the model is already cached locally.
        """
        cpu_threads = resolve_cpu_threads(cpu_threads)

        self.vocabulary = vocabulary or ""
        self.model_size = model_size
        self.beam_size = beam_size

        # If the model needs downloading and the caller wants progress,
        # pre-fetch with a polling thread so the UI can show a percentage.
        # When already cached, this is a no-op.
        if progress_cb is not None:
            try:
                _prefetch_with_progress(model_size, progress_cb)
            except Exception as e:
                print(f"[Transcriber] Prefetch failed (will let WhisperModel handle it): {e}")

        print(
            f"[Transcriber] Loading Whisper model '{model_size}' "
            f"(device={device}, compute={compute_type}, cpu_threads={cpu_threads})..."
        )
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=1,
            )
            print("[Transcriber] Model loaded successfully.")
        except Exception as e:
            print(f"[Transcriber] Error loading model with compute='{compute_type}': {e}")
            print("[Transcriber] Falling back to compute_type='float32'...")
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type="float32",
                cpu_threads=cpu_threads,
                num_workers=1,
            )

    def set_vocabulary(self, vocabulary: str):
        """Update custom vocabulary without reloading the model."""
        self.vocabulary = vocabulary or ""

    def _build_prompt(self, language):
        base = PROMPTS.get(language or "es", PROMPTS["es"])
        vocab = self.vocabulary.strip()
        if vocab:
            # Append user vocabulary so the model sees these tokens in context
            return f"{base} {vocab}"
        return base

    def transcribe(self, audio_array: np.ndarray, language=None) -> str:
        """
        Transcribes a 1D numpy array of audio data (float32, 16kHz).
        """
        if len(audio_array) == 0:
            return ""

        prompt = self._build_prompt(language)

        # VAD filter removes silence chunks, which is faster AND eliminates
        # the common hallucinations whisper produces on silence
        # (e.g. "gracias por ver el video").
        # condition_on_previous_text=False stops the model from dragging
        # context from prior sentences, which matters for short dictation.
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=self.beam_size,
            language=language,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )

        print(
            f"[Transcriber] Detected language '{info.language}' "
            f"with probability {info.language_probability:.2f}"
        )

        text = "".join(segment.text for segment in segments)
        return text.strip()

    def transcribe_file(self, audio_path, language=None, segment_cb=None, cancel_check=None):
        """
        Transcribe a long audio file (mp3/wav/m4a/mp4/ogg/flac/...).
        faster-whisper decodes via its bundled av/ffmpeg, so any container
        the model already supports works without extra steps.

        Returns (segments, info) where:
          - segments: list of {"start": float, "end": float, "text": str}
          - info: {"language": str, "language_probability": float, "duration": float}

        segment_cb: optional callable(segment_dict, fraction in [0,1]) called
                    for each segment as soon as it's produced. Lets the UI
                    stream the transcript incrementally instead of waiting
                    for completion.
        cancel_check: optional callable() returning True to abort early.
        """
        prompt = self._build_prompt(language)
        segments_gen, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )

        duration = info.duration or 1.0
        collected = []
        for segment in segments_gen:
            if cancel_check is not None and cancel_check():
                break
            seg = {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text,
            }
            collected.append(seg)
            if segment_cb is not None:
                segment_cb(seg, min(1.0, segment.end / duration))

        return collected, {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
        }

    def warmup(self):
        """
        Runs a tiny inference through the model to keep it hot in RAM.
        VAD is disabled here so that silent audio still triggers a real
        forward pass (with VAD on, silence is filtered out and the model
        is never actually invoked).
        """
        try:
            silent_audio = np.zeros(16000, dtype=np.float32)
            segments, _ = self.model.transcribe(
                silent_audio,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            # segments is a generator; consume it to force the work
            for _ in segments:
                pass
        except Exception:
            pass
