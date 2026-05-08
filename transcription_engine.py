from faster_whisper import WhisperModel
import numpy as np

class Transcriber:
    def __init__(self, model_size="base", device="auto", compute_type="int8"):
        """
        Initializes the faster-whisper model.
        - model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
        - device: 'auto' (chooses CUDA if available, else CPU)
        - compute_type: 'int8' is usually the safest cross-platform fallback for CPU, 
                        'float16' is better for GPU.
        """
        print(f"[Transcriber] Loading Whisper model '{model_size}' on device: {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("[Transcriber] Model loaded successfully.")
        except Exception as e:
            print(f"[Transcriber] Error loading model: {e}")
            # Fallback to float32 if int8 is not supported on this CPU
            print("[Transcriber] Falling back to compute_type='float32'...")
            self.model = WhisperModel(model_size, device=device, compute_type="float32")

    def transcribe(self, audio_array: np.ndarray, language=None) -> str:
        """
        Transcribes a 1D numpy array of audio data (float32, 16kHz).
        """
        if len(audio_array) == 0:
            return ""
            
        print("[Transcriber] Transcribing audio...")
        
        # Whisper trick: providing a well-punctuated initial prompt 
        # forces the model to use proper capitalization and punctuation.
        prompt = "Hola, ¿cómo estás? Esto es un texto de ejemplo con comas, puntos y mayúsculas."
        
        # beam_size=5 is default. language=None means auto-detect.
        segments, info = self.model.transcribe(
            audio_array, 
            beam_size=5, 
            language=language,
            initial_prompt=prompt
        )
        
        print(f"[Transcriber] Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        text = ""
        for segment in segments:
            text += segment.text + " "
            
        return text.strip()

    def warmup(self):
        """
        Runs a quick silent audio through the model to keep it hot in memory.
        This prevents Windows from paging the model out to disk when idle.
        """
        try:
            # 1 second of silence
            silent_audio = np.zeros(16000, dtype=np.float32)
            self.model.transcribe(silent_audio, beam_size=1)
        except Exception:
            pass
