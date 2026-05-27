import threading
from collections import deque

import sounddevice as sd
import numpy as np
import queue


class AudioRecorder:
    # Roughly normalizes RMS into [0, 1]. Lower divisor = more sensitive
    # bars: normal speech (RMS ~0.04-0.08) now lands in the 60-100% range
    # instead of 25-50%, so peaks really show.
    _RMS_NORM = 0.06
    _LEVEL_HISTORY = 64

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.q = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        self.stream = None

        # Recent RMS values for the UI waveform. Updated from the audio
        # callback (PortAudio thread), read from the Qt UI thread, so it
        # needs locking.
        self._levels_lock = threading.Lock()
        self._levels = deque(maxlen=self._LEVEL_HISTORY)

    def _callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block (PortAudio thread)."""
        if status:
            print(f"Audio status: {status}")
        if not self.is_recording:
            return

        chunk = indata.copy()
        self.q.put(chunk)

        # Compute RMS for the waveform overlay. Cheap (one mean+sqrt per block).
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        norm = min(1.0, rms / self._RMS_NORM)
        with self._levels_lock:
            self._levels.append(norm)

    def start_recording(self, device_id=None):
        print(f"[AudioRecorder] Starting recording on device: {device_id}...")
        self.is_recording = True
        self.audio_data = []

        while not self.q.empty():
            self.q.get()
        with self._levels_lock:
            self._levels.clear()

        sd_device = None if device_id == "default" else device_id

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

        # Guard against double-stop: a second close() on a closed PortAudio
        # stream raises and would crash the orchestrator thread.
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[AudioRecorder] Error closing stream: {e}")
            finally:
                self.stream = None

        while not self.q.empty():
            self.audio_data.append(self.q.get())

        if len(self.audio_data) > 0:
            return np.concatenate(self.audio_data, axis=0).flatten()

        return np.array([], dtype=np.float32)

    def get_latest_level(self) -> float:
        """
        Lightly smoothed estimate of the most recent input level (0..1).
        Used by the UI to drive a VU-meter-style equalizer where all bars
        react to the *current* loudness instead of audio history.
        """
        with self._levels_lock:
            if not self._levels:
                return 0.0
            tail = list(self._levels)[-3:]
        return sum(tail) / len(tail)

    def get_levels(self, n: int) -> np.ndarray:
        """
        Snapshot of the most recent RMS values, padded/truncated to size n.
        Oldest first, newest last (so bars scroll from right to left visually).
        Returns an array of zeros if no audio has been captured yet.
        """
        with self._levels_lock:
            levels = list(self._levels)

        out = np.zeros(n, dtype=np.float32)
        if not levels:
            return out

        if len(levels) <= n:
            out[-len(levels):] = levels
        else:
            out[:] = levels[-n:]
        return out
