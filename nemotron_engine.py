"""
Motor Nemotron 3.5 ASR streaming (NVIDIA) sobre sherpa-onnx. Sin UI, sin mic.

POR QUÉ EXISTE
--------------
Es el único de los tres que hace las dos cosas a la vez: muestra texto mientras
hablás Y escribe con puntuación. MEDIDO en este Ryzen, clip de 47s alimentado a
1,00x tiempo real:

                        1er texto   espera al soltar   puntúa   modelos   licencia
    Moonshine small       1,63s          0,05s           no        1      no comercial
    Híbrido (MS+Whisper)  1,63s          0,38s           sí        2      no comercial
    Nemotron 3.5 560ms    1,80s          0,00s           sí        1      comercial

La espera 0,00s no es redondeo: es streaming de verdad, al soltar no queda nada
por decodificar. Y el costo de CPU es bajísimo: RTF 0,187 con 4 hilos y 0,270
con 2, o sea que corre a 3,7x tiempo real usando dos hilos.

IDIOMA
------
El modelo cubre 40 locales con AUTODETECCIÓN, y por eso acá no se le fija
ninguno: el usuario habla castellano con términos en inglés sueltos, y a veces
frases enteras en inglés. VERIFICADO sin configurar idioma: el clip castellano
sale perfecto con signos de apertura (`¿Hay algún trabajo…?`) y el clip inglés
también (`When we took our seats at the breakfast table, …`). Fijar "es" haría
peor lo mixto, que es el caso real de uso.

LICENCIA
--------
OpenMDW-1.1, uso comercial explícito. Es la razón por la que este camino sirve
para nitoOS y el de Moonshine (Community License, no comercial fuera de inglés)
no.

POR QUÉ NO CORTA POR FRASES
---------------------------
sherpa-onnx sabe detectar fin de frase por silencio, pero MEDIDO sobre el mismo
clip, cortar DESTRUYE la puntuación: el endpoint dispara y resetea antes de que
el modelo emita el token de cierre.

    sin endpoint     7 signos (`…del día presente.`, `¿Hay algún trabajo…?`)
    endpoint 0,8s    1 signo
    endpoint 1,5s    2 signos

Y esperar más silencio casi no ayuda. Como la puntuación es justamente la razón
para preferir este motor, acá se corre UN SOLO stream por toma: los parciales
van al globo y el texto entero se cierra en finish().

Consecuencia: `on_line` no dispara nunca, y todo sale por `pending_text()`. En
modo hold eso no cambia nada (ahí nada se escribe hasta que soltás). En modo
toggle significa que el texto entra al soltar en vez de frase por frase.

Uso:
    engine = NemotronEngine(num_threads=4)
    session = engine.start_session(on_partial=mostrar, on_line=escribir)
    session.insert_audio(chunk)        # float32 16kHz
    texto = session.finish()
"""
import os
import tarfile
import threading
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from config_manager import CONFIG_DIR

SAMPLE_RATE = 16000

MODEL_DIR = Path(CONFIG_DIR) / "models" / "nemotron-3.5-streaming-560ms"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2"
)
_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def models_ready() -> bool:
    return all((MODEL_DIR / f).exists() for f in _FILES)


def ensure_model(progress_cb: Optional[Callable[[float, float], None]] = None):
    """Baja y desempaqueta el modelo si falta (453 MB comprimido)."""
    if models_ready():
        return
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    tarball = MODEL_DIR.parent / "nemotron-560ms.tar.bz2"

    def _hook(blocknum, blocksize, totalsize):
        if progress_cb and totalsize > 0:
            done = min(totalsize, blocknum * blocksize)
            progress_cb(done / (1024 * 1024), totalsize / (1024 * 1024))

    print(f"[Nemotron] Descargando modelo… ({MODEL_URL})")
    urllib.request.urlretrieve(MODEL_URL, tarball, reporthook=_hook)
    print("[Nemotron] Desempaquetando…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:bz2") as tf:
        for member in tf.getmembers():
            name = os.path.basename(member.name)
            if name in _FILES:
                member.name = name
                tf.extract(member, MODEL_DIR)
    try:
        tarball.unlink()
    except OSError:
        pass
    print("[Nemotron] Modelo listo.")


class NemotronSession:
    """Una toma de dictado. No es thread-safe: alimentarla desde un solo hilo,
    igual que MoonshineSession y SegmentASR."""

    def __init__(self, recognizer, on_partial=None, on_line=None, on_error=None):
        self._rec = recognizer
        self._stream = recognizer.create_stream()
        self.on_partial = on_partial
        self.on_line = on_line
        self.on_error = on_error
        self._lines: List[str] = []
        self._partial = ""
        self._finished = False

    # --- entrada ---
    def insert_audio(self, chunk: np.ndarray):
        if self._finished or len(chunk) == 0:
            return
        try:
            self._stream.accept_waveform(
                SAMPLE_RATE, np.asarray(chunk, dtype=np.float32))
            self._drain()
        except Exception as exc:  # noqa: BLE001 - una toma mala no mata la app
            self._error(exc)

    def _drain(self, final=False):
        """Decodifica lo que haya listo y muestra el texto acumulado.

        No cierra frases: ver el encabezado del módulo. El texto crece durante
        toda la toma y se cierra recién en finish().
        """
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)

        text = (self._rec.get_result(self._stream) or "").strip()
        if text != self._partial:
            self._partial = text
            if not final and self.on_partial is not None:
                self.on_partial(text)

    def _error(self, exc):
        print(f"[Nemotron] {exc}")
        if self.on_error is not None:
            self.on_error(exc)

    # --- salida ---
    def text(self) -> str:
        return " ".join(self._partial.split()).strip()

    def pending_text(self) -> str:
        """TODO el texto: como `on_line` nunca disparó, nada se entregó antes.

        Así el orquestador escribe lo mismo en hold (usa text()) y en toggle
        (usa pending_text()), sin duplicar nada."""
        return self.text()

    def finish(self) -> str:
        if not self._finished:
            self._finished = True
            try:
                self._stream.input_finished()
                self._drain(final=True)
            except Exception as exc:  # noqa: BLE001
                self._error(exc)
        return self.text()


class NemotronEngine:
    """Carga el modelo una vez y abre sesiones sobre él."""

    def __init__(self, num_threads: int = 4,
                 progress_cb: Optional[Callable[[float, float], None]] = None,
                 **_ignored):
        import sherpa_onnx

        ensure_model(progress_cb)
        self._lock = threading.Lock()
        print(f"[Nemotron] Cargando modelo ({num_threads} hilos)…")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(MODEL_DIR / "tokens.txt"),
            encoder=str(MODEL_DIR / "encoder.int8.onnx"),
            decoder=str(MODEL_DIR / "decoder.int8.onnx"),
            joiner=str(MODEL_DIR / "joiner.int8.onnx"),
            num_threads=num_threads,
            provider="cpu",
            model_type="nemo_transducer",
            decoding_method="greedy_search",
            enable_endpoint_detection=False,   # ver el encabezado: cortar mata la puntuación
        )
        self.vocabulary = ""
        # La ventana de archivos identifica los modelos por este campo (compara
        # contra el del Transcriber de Whisper para reusar el que ya esté cargado).
        self.model_size = "nemotron"
        print("[Nemotron] Listo.")

    def set_vocabulary(self, vocabulary: str):
        """Nemotron en sherpa-onnx sesga por 'hotwords', que se pasan al crear
        el stream y no al modelo. No está cableado todavía: por ahora el
        vocabulario propio del usuario lo aplica corrections.apply() sobre el
        texto, que en la práctica es más confiable que el sesgo."""
        self.vocabulary = vocabulary or ""

    def start_session(self, on_partial=None, on_line=None, on_error=None) -> NemotronSession:
        with self._lock:
            return NemotronSession(self.recognizer, on_partial=on_partial,
                                   on_line=on_line, on_error=on_error)

    # ------------------------------------------------- transcribir archivos --
    #
    # La ventana de "Transcribir archivo…" habla con un objeto que expone
    # `model_size` y `transcribe_file()`. Implementarlos acá deja que el motor
    # entre en esa lista sin que la ventana tenga que saber que no es Whisper,
    # y permite REUSAR el modelo que el dictado ya tiene cargado en vez de
    # levantar otros 650 MB.

    def transcribe_file(self, audio_path, language=None, segment_cb=None,
                        cancel_check=None, task="transcribe"):
        """Transcribe un archivo entero. Devuelve (segmentos, info) con la
        misma forma que Transcriber.transcribe_file.

        No usa el detector de fin de frase de sherpa-onnx —arruina la
        puntuación, ver el encabezado— sino que decodifica de un tirón y
        DESPUÉS corta en frases usando los timestamps por token. Así el SRT y
        la diarización tienen tiempos, y el texto conserva los signos.
        """
        if task == "translate":
            raise ValueError(
                "Nemotron no traduce: solo transcribe en el idioma hablado. "
                "Para traducir al inglés elegí un modelo Whisper."
            )
        from faster_whisper.audio import decode_audio   # trae av/ffmpeg: mp3, m4a, mp4…

        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        total = len(audio)
        duration = total / SAMPLE_RATE if total else 1.0

        with self._lock:
            stream = self.recognizer.create_stream()

        segmentos, emitidos = [], 0
        paso = SAMPLE_RATE * 10          # 10s de audio por vuelta: progreso fluido
        for inicio in range(0, total, paso):
            if cancel_check is not None and cancel_check():
                return segmentos, {"language": language or "auto",
                                   "language_probability": 0.0, "duration": duration}
            stream.accept_waveform(SAMPLE_RATE, audio[inicio:inicio + paso])
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            # Entregar las frases YA cerradas, para que la ventana muestre el
            # texto mientras avanza en vez de esperar al final.
            frac = min(1.0, (inicio + paso) / total) if total else 1.0
            nuevos = self._segments_from(stream, cerrar=False)
            for seg in nuevos[emitidos:]:
                segmentos.append(seg)
                if segment_cb is not None:
                    segment_cb(seg, frac)
            emitidos = len(nuevos)

        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        for seg in self._segments_from(stream, cerrar=True)[emitidos:]:
            segmentos.append(seg)
            if segment_cb is not None:
                segment_cb(seg, 1.0)

        return segmentos, {"language": language or "auto",
                           "language_probability": 0.0, "duration": duration}

    # Una frase cierra con alguno de estos, o tras un silencio largo.
    _FIN_DE_FRASE = (".", "?", "!", "…")
    _PAUSA_LARGA = 0.8      # segundos entre tokens que cuentan como corte
    _MAX_SEG = 15.0         # ningún subtítulo más largo que esto

    def _segments_from(self, stream, cerrar):
        """Arma {start, end, text} a partir de los tokens y sus tiempos.

        `cerrar=False` deja afuera la última frase si todavía no terminó: el
        decoder puede seguir agregándole palabras. Con `cerrar=True` se entrega
        lo que quede. Los transductores no reescriben lo ya emitido, así que
        una frase cerrada no va a cambiar después.
        """
        import json

        datos = json.loads(self.recognizer.get_result_as_json_string(stream))
        tokens = datos.get("tokens") or []
        tiempos = datos.get("timestamps") or []
        if not tokens or len(tiempos) < len(tokens):
            return []

        salida, actual, t0 = [], [], None
        fin_palabra = None      # último token que NO es puntuación: marca el end
        for i, tok in enumerate(tokens):
            if t0 is None:
                t0 = tiempos[i]
            actual.append(tok)
            if not self._solo_puntuacion(tok):
                fin_palabra = tiempos[i]
            cierra_frase = tok.strip().endswith(self._FIN_DE_FRASE)
            hueco = (tiempos[i + 1] - tiempos[i]) if i + 1 < len(tokens) else 0.0
            corta = (cierra_frase
                     or hueco >= self._PAUSA_LARGA
                     or (tiempos[i] - t0) >= self._MAX_SEG)

            # El punto de una frase suele llegar DESPUÉS del silencio que la
            # separa de la siguiente. Si cortáramos por ese silencio, el signo
            # quedaría como un segmento suelto que dice solo ".", y el SRT se
            # llenaría de subtítulos de un carácter. Así que si lo único que
            # falta es ese signo, se espera una vuelta más y entra en la frase
            # que le corresponde.
            if corta and not cierra_frase and i + 1 < len(tokens) \
                    and self._solo_puntuacion(tokens[i + 1]):
                continue

            if corta:
                texto = "".join(actual).strip()
                if texto:
                    salida.append({"start": float(t0),
                                   "end": float(fin_palabra if fin_palabra is not None else tiempos[i]),
                                   "text": " " + texto})
                actual, t0, fin_palabra = [], None, None
        if cerrar and actual:
            texto = "".join(actual).strip()
            if texto:
                salida.append({"start": float(t0),
                               "end": float(fin_palabra if fin_palabra is not None else tiempos[-1]),
                               "text": " " + texto})
        return salida

    @staticmethod
    def _solo_puntuacion(token: str) -> bool:
        t = token.strip()
        return bool(t) and all(c in ".,;:¿?¡!…-–—\"'" for c in t)

    def close(self):
        pass
