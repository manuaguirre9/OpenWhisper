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
import contextlib
import math
import re
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np

# --- audio ---
SAMPLE_RATE = 16000

# --- streaming schedule ---
MIN_CHUNK_S = 1.0        # minimum new audio before running a streaming pass
TRIM_BUFFER_S = 20.0     # trim committed audio once the buffer passes this
KEEP_CONTEXT_S = 1.0     # left context kept after a trim
MIN_DECODE_S = 0.2       # skip a pass on buffers shorter than this
COMMIT_NGRAM_MAX = 16    # palabras del final ya comiteado que se buscan para dedup
                         # (el whisper_streaming original usa 5; MEDIDO acá, es poco:
                         #  en un clip corto el buffer nunca se recorta, así que la
                         #  pasada final re-propone desde bastante atrás. Con 5, un
                         #  solapamiento de 6 palabras dejaba "para" afuera de la
                         #  ventana, ningún n-grama coincidía y la frase entera salía
                         #  duplicada: "...cubra el seguro. Para que lo cubra el seguro.")
MAX_TS_DRIFT_S = 1.5     # cuánto se corren los timestamps entre pasadas (medido ~1s)

# --- decoding ---
STREAM_BEAM = 1          # greedy during streaming (speed)
FULL_BEAM = 1            # el decode del dictado real (transcription_engine.py usa
                         # config["beam_size"], default 1). Estuvo en 5 y NO era el
                         # baseline de la app: inflaba la columna "hoy" del banco.
                         # MEDIDO (Pi 5, small int8, 3 hilos): beam 5 cuesta +21% en
                         # el clip de 47s (33,4s contra 27,6s) y +2% en el de 4,8s,
                         # con el MISMO texto — 0,0% de WER en los dos, en los dos beams.

# --- ventana del encoder ---
# Whisper padea SIEMPRE a 30s y el encoder corre sobre esa ventana entera, hable
# quien hable 2 segundos o 25. MEDIDO en la Pi (small int8, clip de 4,8s): de los
# 7,2s de espera, 4,4s son encoder y 2,8s decoder — o sea que el 62% del costo es
# encodear silencio. Padear solo lo necesario es el único lugar donde estaba la
# grasa, y sale casi gratis en calidad (ver short_window()).
FULL_WINDOW_FRAMES = 3000    # 30s: lo que asume el pad_or_trim de faster-whisper
WINDOW_PAD_S = 2.0           # silencio que se le deja DESPUÉS del audio
MIN_WINDOW_S = 5.0           # piso: nunca una ventana más corta que esto
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


_window_tls = threading.local()
_pad_or_trim_patched = False


def _install_dynamic_pad() -> bool:
    """Reemplaza `faster_whisper.transcribe.pad_or_trim` por una versión que
    padea lo que hace falta en vez de 3000 frames fijos.

    Se hace por parche porque la biblioteca no expone ningún otro punto: el
    padeo está hardcodeado adentro del loop de `generate_segments`, y es UN
    símbolo el que cubre los tres usos (la ventana de cada seek, el batched y la
    detección de idioma). El parche se instala una vez y es INERTE por default:
    solo cambia algo para el hilo que esté adentro de `short_window()`, así ni
    otro hilo ni el pipeline batched (que apila features y necesita que todas
    midan lo mismo) ven nada distinto.
    """
    global _pad_or_trim_patched
    if _pad_or_trim_patched:
        return True
    try:
        import faster_whisper.transcribe as fw
    except Exception:
        return False

    original = fw.pad_or_trim

    def dynamic_pad_or_trim(array, length: int = FULL_WINDOW_FRAMES, *, axis: int = -1):
        pad = getattr(_window_tls, "pad_frames", None)
        if pad is not None and length == FULL_WINDOW_FRAMES:
            floor = getattr(_window_tls, "min_frames", 0)
            # Nunca menos de lo que ya mide el array: recortar audio sería
            # perder palabras, que es justo lo contrario de lo que se busca.
            length = min(length, max(floor, array.shape[axis] + pad))
        return original(array, length, axis=axis)

    fw.pad_or_trim = dynamic_pad_or_trim
    _pad_or_trim_patched = True
    return True


@contextlib.contextmanager
def short_window(pad_s: float = WINDOW_PAD_S, min_s: float = MIN_WINDOW_S,
                 enabled: bool = True):
    """Adentro de este `with`, Whisper encodea `audio + pad_s` en vez de 30s.

    MEDIDO (Pi 5, small int8, beam 1, 3 hilos; espera de punta a punta por el
    pipeline REAL de faster-whisper, VAD y fallbacks incluidos):

        audio    30s (hoy)   ventana corta
         2,0s       5,91s        1,27s   (4,7x)
         4,8s       7,27s        2,55s   (2,9x)
         6,0s       6,89s        2,50s   (2,8x)
        10,0s       8,96s        4,52s   (2,0x)
        16,0s      11,73s        8,09s   (1,4x)
        >29s          igual: la ventana ya era de 30s

    El texto sale IDÉNTICO en los 5 casos y en 3 clips distintos (es-AR corto,
    es-AR largo recortado, en). En un recorte de 8s la ventana corta salió
    MEJOR: agarró "Lo del parque", que la de 30s se comía.

    El pad importa: sin suficiente silencio atrás el modelo no emite
    <|endoftext|>, sigue generando y repite la frase hasta max_length (medido,
    llamando al decoder pelado: 19s de decode para 4,8s de audio). Por el
    pipeline real —que trae VAD y el fallback por compression_ratio— aguanta
    hasta +0,5s en todo lo medido; los 2s del default son margen barato: cada
    segundo extra de ventana cuesta ~0,08s de encoder. (whisper.cpp expone lo
    mismo como --audio-ctx y la fórmula empírica de esa comunidad,
    audio_frames + 128 posiciones, son +2,56s: el mismo número por otro lado.)

    ⚠ ESTO ES SEGURO POR LA COMPAÑÍA QUE TIENE, NO SOLO POR EL PAD. Medido sobre
    12 notas de voz reales: por este camino (vad_filter=True + initial_prompt +
    condition_on_previous_text=False) no degeneró ninguna, y el total bajó 1,30x
    —2,5 a 4x en las notas cortas—. Las MISMAS 12 notas, con el mismo recorte
    pero sin VAD, sin prompt y con condition_on_previous_text en su default
    (que es la config del transcriptor de ULTRON), dieron una nota de 10,3s
    tardando 90,58s: el bucle degenerado, en producción y en silencio. Si vas a
    llevar este recorte a otro lugar, llevate las tres opciones con él y medí
    con audio real antes.
    """
    if not enabled or not _install_dynamic_pad():
        yield False
        return
    prev_pad = getattr(_window_tls, "pad_frames", None)
    prev_min = getattr(_window_tls, "min_frames", None)
    _window_tls.pad_frames = int(math.ceil(pad_s * 100))       # 100 frames de mel = 1s
    _window_tls.min_frames = int(math.ceil(min_s * 100))
    try:
        yield True
    finally:
        _window_tls.pad_frames = prev_pad
        _window_tls.min_frames = prev_min


class HypothesisBuffer:
    """
    LocalAgreement-2 committer (after ufal/whisper_streaming).

    Holds the previous pass's uncommitted hypothesis; on each new pass it
    commits the longest common prefix of words that agree with the previous
    pass. Words are (start_abs, end_abs, text) in absolute stream time.
    """

    def __init__(self):
        self.buffer: List[Word] = []      # previous pass, still tentative
        self.committed_tail: List[Word] = []   # últimas comiteadas, para dedup por texto
        self.last_committed_time = 0.0

    def insert(self, words, offset: float) -> List[Word]:
        """Pasa a tiempo absoluto y saca el prefijo que ya se emitió.

        MEDIDO (2026-09-09, clip largo es-AR): los timestamps de una palabra se
        corren HASTA ~1s entre pasadas, porque cada pasada decodifica una
        ventana distinta. Con "ya emitido" decidido por tiempo —comparando el
        comienzo (versión original) o el final (primer intento de arreglo)— la
        palabra siguiente a la última comiteada cae del lado equivocado de la
        frontera, queda descartada en TODAS las pasadas siguientes y se pierde
        en silencio. Trazado real:

            SALTO de frontera 3.94 -> 9.88 comiteando [... 'ahí', 'me']
            DESCARTA 'puedo' [8.88-9.84]  frontera=9.88

        'puedo' va DESPUÉS de 'me' en el audio, pero viene fechada antes.

        Así que el tiempo decide GRUESO (tirar lo que quedó muy atrás) y el
        TEXTO decide FINO: se saca el prefijo que repite literalmente la cola
        ya comiteada, hasta COMMIT_NGRAM_MAX palabras. Es el dedup por n-gramas
        del whisper_streaming original, que acá faltaba. Falla perdiendo
        precisión en el dedup (a lo sumo una repetición visible), nunca
        borrando una palabra que nadie emitió.
        """
        shifted = [(s + offset, e + offset, w) for (s, e, w) in words]

        # Grueso: solo el corrimiento medido de margen. Con más tolerancia (se
        # probó con KEEP_CONTEXT_S sumado) el prefijo ya emitido sobrevive al
        # filtro, y en un clip corto —donde el buffer nunca se recorta— la
        # pasada final re-propone la oración ENTERA desde el principio: el
        # dedup por n-gramas no la agarra porque compara contra la COLA
        # comiteada, no contra todo lo emitido, y la frase sale duplicada
        # (medido: "...cubra el seguro. Para que lo cubra el seguro.").
        floor = self.last_committed_time - MAX_TS_DRIFT_S
        new = [t for t in shifted if t[1] > floor]

        # Fino: si el arranque repite el final ya comiteado, se saca por texto.
        tail = self.committed_tail
        for n in range(min(len(tail), len(new), COMMIT_NGRAM_MAX), 0, -1):
            if all(norm_word(tail[-n + i][2]) == norm_word(new[i][2]) for i in range(n)):
                return new[n:]
        return new

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
        # La cola comiteada es lo que insert() usa para deduplicar por texto.
        if committed:
            self.committed_tail = (self.committed_tail + committed)[-COMMIT_NGRAM_MAX:]
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
        short_window_enabled: bool = False,
    ):
        self.model = model
        self.language = language
        self.beam_size = beam_size
        self.trim_buffer_s = trim_buffer_s
        self.on_error = on_error
        # APAGADO A PROPÓSITO. El recorte de ventana (short_window) es 3x en el
        # one-shot y acá ROMPE: MEDIDO en la Pi con tiny, trim 3s, min_chunk 2s,
        # los dos clips es-AR —
        #
        #     clip     espera            pasada más lenta      WER
        #     4,8s      3,75s -> 42,32s   1,50s -> 41,66s   25% -> 50%
        #    47,1s      8,61s -> 198,08s  2,48s -> 41,40s   13% ->  9,7%
        #
        # 41s de decode para un buffer de 3s con `tiny` es el modelo generando
        # hasta max_length: sin bastante silencio atrás no emite <|endoftext|>.
        # Una pasada de streaming es justo el caso peor — buffer corto, prompt
        # largo (la cola ya comiteada) y word_timestamps=True. El one-shot no
        # tiene nada de eso y por eso sí aguanta.
        #
        # Queda como flag y no borrado para que el hallazgo se pueda reproducir
        # (bench: --stream-short-window) el día que se toque el prompt o el trim.
        self.short_window_enabled = short_window_enabled
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
            with short_window(enabled=self.short_window_enabled):
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
                        beam_size: int = FULL_BEAM,
                        short_window_enabled: bool = True,
                        initial_prompt: Optional[str] = None) -> str:
    """
    El decode que hace el dictado: una pasada sobre toda la grabación,
    greedy + VAD. Es la referencia de calidad y de ESPERA que el streaming
    tiene que ganarle.

    `short_window_enabled` recorta la ventana del encoder a lo que dura el
    audio (ver short_window(): 2,9x en un dictado de 5s, mismo texto). Se puede
    apagar para medir el antes/después sin tocar el resto.
    """
    with short_window(enabled=short_window_enabled):
        segments, _ = model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt or base_prompt_for(language),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=FULL_VAD_SILENCE_MS),
            condition_on_previous_text=False,
        )
        return "".join(s.text for s in segments).strip()
