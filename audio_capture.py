import sounddevice as sd
import numpy as np
import queue

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.q = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        self.stream = None

    def _callback(self, indata, frames, time, status):
        """This is called for each audio block by sounddevice."""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            # We must copy indata because it will be overwritten by the audio hardware
            self.q.put(indata.copy())

    def start_recording(self, device_id=None):
        print(f"[AudioRecorder] Starting recording on device: {device_id}...")
        self.is_recording = True
        self.audio_data = []
        
        # Clear any stale data in the queue
        while not self.q.empty():
            self.q.get()
            
        # If device_id is "default", sounddevice expects None
        sd_device = None if device_id == "default" else device_id
            
        # Start a new InputStream
        # Whisper expects 16kHz, mono, float32
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            callback=self._callback, 
            dtype='float32',
            device=sd_device
        )
        self.stream.start()

    def stop_recording(self) -> np.ndarray:
        print("[AudioRecorder] Stopping recording...")
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        # Drain the queue to collect all recorded chunks
        while not self.q.empty():
            self.audio_data.append(self.q.get())
            
        if len(self.audio_data) > 0:
            # Concatenate all chunks into a single 1D numpy array
            return np.concatenate(self.audio_data, axis=0).flatten()
            
        return np.array([], dtype=np.float32)
