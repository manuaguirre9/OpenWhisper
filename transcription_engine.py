import os
from faster_whisper import WhisperModel
import numpy as np

# Initial prompts per language. They bias the model towards proper punctuation
# and capitalization. Using the prompt in the wrong language hurts accuracy,
# so we pick one per detected/forced language.
PROMPTS = {
    "es": "Hola, ¿cómo estás? Esto es un texto de ejemplo con comas, puntos y mayúsculas.",
    "en": "Hello, how are you? This is an example text with commas, periods, and capitalization.",
}


class Transcriber:
    def __init__(
        self,
        model_size="base",
        device="auto",
        compute_type="int8",
        cpu_threads=0,
        vocabulary="",
    ):
        """
        Initializes the faster-whisper model.
        - model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
        - device: 'auto' (chooses CUDA if available, else CPU)
        - compute_type: 'int8' is the safest cross-platform CPU default.
                        'float16' is better for CUDA GPUs.
        - cpu_threads: 0 = use all logical CPUs. Only applied on CPU device.
        - vocabulary: free-form string of names/jargon/technical terms to bias
                      the model towards (appended to the initial prompt).
        """
        if cpu_threads == 0:
            cpu_threads = os.cpu_count() or 4

        self.vocabulary = vocabulary or ""
        self.model_size = model_size

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
                num_workers=2,
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
                num_workers=2,
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
            beam_size=5,
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
