import threading
from collections import deque

import sounddevice as sd
import numpy as np
import queue


class HighPass:
    """Pasa-altos Butterworth de 4º orden (dos biquads), con estado entre bloques.

    Por qué: MEDIDO en las entradas del usuario (Audient EVO4, "Mic | Line 2"
    y también la 1/2), hay un zumbido de 50 Hz con armónico en 100 Hz a
    -23 dBFS, más fuerte que la voz; y Silero lo marca como habla continua.
    Un 2º orden a 80 Hz lo bajaba solo ~8 dB; el 4º orden a 100 Hz lo baja
    ~24 dB. Por debajo de 100 Hz de la voz solo queda parte del fundamental
    grave, que a los modelos de ASR no les hace falta (la telefonía corta
    en 300 Hz).
    """

    def __init__(self, cutoff_hz=100.0, sample_rate=16000):
        import math
        w0 = 2.0 * math.pi * cutoff_hz / sample_rate
        cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
        # Butterworth de 4º orden = dos secciones de 2º orden con Q 0.5412 y 1.3066.
        self._sections = []
        for q in (0.54119610, 1.30656296):
            alpha = sin_w0 / (2.0 * q)
            a0 = 1.0 + alpha
            b = np.array([(1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0]) / a0
            a = np.array([-2.0 * cos_w0, 1.0 - alpha]) / a0
            self._sections.append([b, a, np.zeros(2)])

    def reset(self):
        for sec in self._sections:
            sec[2][:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Direct Form II transpuesta, muestra a muestra. Bloques de 100ms
        (1600 muestras) tardan ~2ms en Python; no vale la pena scipy."""
        y = np.asarray(x, dtype=np.float64)
        for sec in self._sections:
            (b0, b1, b2), (a1, a2), z = sec[0], sec[1], sec[2]
            z1, z2 = z
            out = np.empty_like(y)
            for i in range(len(y)):
                xi = y[i]
                yi = b0 * xi + z1
                z1 = b1 * xi - a1 * yi + z2
                z2 = b2 * xi - a2 * yi
                out[i] = yi
            z[0], z[1] = z1, z2
            y = out
        return y.astype(np.float32)


class AudioRecorder:
    # Roughly normalizes RMS into [0, 1]. Lower divisor = more sensitive
    # bars: normal speech (RMS ~0.04-0.08) now lands in the 60-100% range
    # instead of 25-50%, so peaks really show.
    _RMS_NORM = 0.06
    _LEVEL_HISTORY = 64

    def __init__(self, sample_rate=16000, highpass_hz=100.0):
        self.sample_rate = sample_rate
        # Ver HighPass. 0/None lo desactiva.
        self._hp = HighPass(highpass_hz, sample_rate) if highpass_hz else None
        self.q = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        self.stream = None

        # Recent RMS values for the UI waveform. Updated from the audio
        # callback (PortAudio thread), read from the Qt UI thread, so it
        # needs locking.
        self._levels_lock = threading.Lock()
        self._levels = deque(maxlen=self._LEVEL_HISTORY)

        # El dictado por segmentos consume audio MIENTRAS se graba, así que la
        # cola la vacían dos lados (ese consumidor y stop_recording). El lock
        # serializa el drenaje: sin él los bloques podrían quedar en
        # audio_data en otro orden que el que se habló.
        self._drain_lock = threading.Lock()

    def _callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block (PortAudio thread)."""
        if status:
            print(f"Audio status: {status}")
        if not self.is_recording:
            return

        chunk = indata.copy()
        if self._hp is not None:
            flat = self._hp.process(chunk[:, 0].astype(np.float32))
            chunk = flat.reshape(-1, 1).astype(np.float32)
        self.q.put(chunk)

        # Compute RMS for the waveform overlay. Cheap (one mean+sqrt per block).
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        norm = min(1.0, rms / self._RMS_NORM)
        with self._levels_lock:
            self._levels.append(norm)

    def drain_new(self) -> np.ndarray:
        """Pasa a `audio_data` lo que el callback haya encolado y lo devuelve.

        Para consumir audio mientras la grabación sigue viva. `audio_data` sigue
        acumulando TODO, así que `stop_recording()` devuelve la grabación
        completa igual: quien consuma en vivo tiene que llevar su propia cuenta
        de cuánto ya procesó.
        """
        new = []
        with self._drain_lock:
            while not self.q.empty():
                chunk = self.q.get()
                self.audio_data.append(chunk)
                new.append(chunk)
        if not new:
            return np.array([], dtype=np.float32)
        return np.concatenate(new, axis=0).flatten()

    def start_recording(self, device_id=None):
        print(f"[AudioRecorder] Starting recording on device: {device_id}...")
        self.is_recording = True
        self.audio_data = []
        if self._hp is not None:
            self._hp.reset()

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

        with self._drain_lock:
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
