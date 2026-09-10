"""
Motor Moonshine v2 para dictado con texto en vivo. Sin UI, sin teclado, sin mic.

Por qué existe: Whisper no puede mostrar texto mientras hablás en una CPU
normal. Cada pasada de streaming re-decodifica el buffer entero (medido en
este Ryzen: 1 a 5s por pasada con `small`), así que el texto llega 6 a 60s
tarde. Moonshine es un modelo pensado para streaming: procesa el audio una
sola vez, de forma incremental, y emite parciales cada ~0,5s.

MEDIDO (spike_moonshine.py, clips del banco, alimentados a tiempo real):
SMALL_STREAMING es: espera al soltar 0,01-0,17s, primer parcial ~1,2s,
WER 0-0,9% (una tilde), ~280MB de RSS.

Cómo se comporta el stream, y por qué la API de acá es como es:
  - Los PARCIALES reescriben la frase entera. No son append-only: en el clip
    corto 7 de 8 parciales cambiaron palabras ya mostradas. Sirven para
    MOSTRAR, no para escribir en la app destino.
  - Las LÍNEAS COMPLETAS (on_line_completed) sí son definitivas. Esas son las
    que se pueden escribir en el destino a medida que cierran.

Uso:
    engine = MoonshineEngine("es", model_size="small", vocabulary="Nobi, inox")
    session = engine.start_session(on_partial=mostrar, on_line=escribir)
    session.insert_audio(chunk)        # float32 16kHz, desde el mic
    texto = session.finish()           # al terminar: texto completo
    resto = session.pending_text()     # lo que on_line NO llegó a entregar

Licencia: el código de moonshine-voice es MIT; los modelos que NO son inglés
están bajo la Moonshine Community License (no comercial por encima de 1M USD
de facturación anual). Ver https://www.moonshine.ai/license
"""
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "es"
# Idiomas con modelo publicado (moonshine-voice 0.1.5). "auto" no existe en
# Moonshine: el modelo es por idioma.
SUPPORTED_LANGUAGES = {"ar", "es", "de", "en", "ja", "ko", "vi", "uk", "zh", "tl"}

try:  # la biblioteca es opcional: sin ella el motor no carga pero el módulo importa
    from moonshine_voice import TranscriptEventListener as _ListenerBase
except Exception:  # pragma: no cover - depende de la máquina
    _ListenerBase = object


def resolve_language(language: Optional[str]) -> str:
    """Moonshine necesita un idioma fijo. 'auto' o desconocido → español."""
    if not language or language == "auto":
        return DEFAULT_LANGUAGE
    lang = language.lower().split("-")[0]
    return lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def split_vocabulary(vocabulary: str) -> List[str]:
    """El vocabulario de la app es texto libre; Moonshine quiere una lista de
    términos sin comas (usa la coma como separador interno)."""
    if not vocabulary:
        return []
    terms: List[str] = []
    for raw in vocabulary.replace(";", ",").replace("\n", ",").split(","):
        term = raw.strip()
        if term and term not in terms:
            terms.append(term)
    return terms


def arch_for(model_size: str, language: str):
    """Mapea el `model_size` de la app (tiny/base/small/medium) a una
    arquitectura streaming de Moonshine. Solo inglés tiene medium."""
    from moonshine_voice import ModelArch

    size = (model_size or "small").lower()
    if size == "tiny":
        return ModelArch.TINY_STREAMING
    if size == "medium" and language == "en":
        return ModelArch.MEDIUM_STREAMING
    return ModelArch.SMALL_STREAMING


class _Listener(_ListenerBase):
    """Traduce los eventos de Moonshine al vocabulario de la sesión."""

    def __init__(self, session: "MoonshineSession"):
        self._session = session

    def on_line_text_changed(self, event):
        self._session._line_changed(event.line.line_id, event.line.text)

    def on_line_completed(self, event):
        self._session._line_completed(event.line.line_id, event.line.text)

    def on_error(self, event):
        self._session._error(getattr(event, "error", event))


class MoonshineSession:
    """Una toma de dictado: del primer bloque de audio al texto final.

    Lleva la cuenta de qué líneas ya entregó por `on_line`, para que quien
    escribe en vivo pueda pedir al final SOLO lo que falta (`pending_text`).
    No es thread-safe por diseño: alimentarla siempre desde el mismo hilo,
    igual que SegmentASR.
    """

    def __init__(
        self,
        stream,
        on_partial: Optional[Callable[[str], None]] = None,
        on_line: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ):
        self._stream = stream
        self.on_partial = on_partial
        self.on_line = on_line
        self.on_error = on_error
        self._lines: Dict[int, str] = {}     # line_id -> último texto, en orden
        self._emitted: set = set()           # line_ids ya entregados por on_line
        self._finished = False
        self.errors: List[str] = []
        self._listener = _Listener(self)
        self._stream.add_listener(self._listener)
        self._stream.start()

    # --- eventos (los llama Moonshine desde el hilo que alimenta audio) ---
    def _line_changed(self, line_id, text):
        self._lines[line_id] = text or ""
        if self.on_partial is not None and line_id not in self._emitted:
            self.on_partial(self._lines[line_id].strip())

    def _line_completed(self, line_id, text):
        self._lines[line_id] = text or ""
        if line_id in self._emitted:
            return
        self._emitted.add(line_id)
        clean = self._lines[line_id].strip()
        if clean and self.on_line is not None:
            self.on_line(clean)

    def _error(self, exc):
        self.errors.append(repr(exc))
        if self.on_error is not None:
            self.on_error(exc)

    # --- entrada ---
    def insert_audio(self, chunk: np.ndarray):
        if self._finished or len(chunk) == 0:
            return
        self._stream.add_audio(np.asarray(chunk, dtype=np.float32).tolist(), SAMPLE_RATE)

    # --- salida ---
    def text(self) -> str:
        return " ".join(t.strip() for t in self._lines.values() if t.strip()).strip()

    def pending_text(self) -> str:
        """Líneas que `on_line` no entregó (la última, si no llegó a cerrar)."""
        pending = [t.strip() for lid, t in self._lines.items()
                   if lid not in self._emitted and t.strip()]
        return " ".join(pending).strip()

    def finish(self) -> str:
        """Cierra el stream (decodifica la cola, dispara los on_line que falten)
        y devuelve el texto completo de la toma."""
        if not self._finished:
            self._finished = True
            try:
                self._stream.stop()
            except Exception as exc:  # noqa: BLE001 - la cola perdida no mata la toma
                self._error(exc)
            try:
                self._stream.close()
            except Exception:
                pass
        return self.text()


class MoonshineEngine:
    """Carga un modelo de Moonshine para un idioma y abre sesiones sobre él."""

    def __init__(
        self,
        language: Optional[str],
        model_size: str = "small",
        vocabulary: str = "",
        update_interval: float = 0.5,
        progress_cb: Optional[Callable[[float, float], None]] = None,
    ):
        from moonshine_voice import Transcriber, get_model_for_language

        self.language = resolve_language(language)
        self.update_interval = update_interval
        self._lock = threading.Lock()

        def on_progress(fraction, _file):
            if progress_cb is not None:
                progress_cb(fraction * 100.0, 100.0)

        arch = arch_for(model_size, self.language)
        try:
            path, resolved = get_model_for_language(self.language, arch, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001 - esa arquitectura no existe para el idioma
            print(f"[Moonshine] {arch.name} no disponible para '{self.language}' ({exc}); uso el default.")
            path, resolved = get_model_for_language(self.language, None, on_progress=on_progress)

        print(f"[Moonshine] Cargando {resolved.name} para '{self.language}'…")
        self.transcriber = Transcriber(model_path=path, model_arch=resolved,
                                       update_interval=update_interval)
        self.model_arch = resolved.name
        self.vocabulary = ""
        self.set_vocabulary(vocabulary)
        print("[Moonshine] Listo.")

    def set_vocabulary(self, vocabulary: str):
        """Sesga el decoder hacia los términos del usuario. Vale en caliente."""
        self.vocabulary = vocabulary or ""
        terms = split_vocabulary(self.vocabulary)
        try:
            self.transcriber.set_keyterms(terms or None)
        except Exception as exc:  # noqa: BLE001 - sin sesgo se dicta igual
            print(f"[Moonshine] No pude aplicar el vocabulario: {exc}")

    def start_session(self, on_partial=None, on_line=None, on_error=None) -> MoonshineSession:
        with self._lock:
            stream = self.transcriber.create_stream(update_interval=self.update_interval)
        return MoonshineSession(stream, on_partial=on_partial, on_line=on_line, on_error=on_error)

    def close(self):
        try:
            self.transcriber.close()
        except Exception:
            pass
