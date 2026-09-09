"""
Dictado por SEGMENTOS: decodificar cada frase apenas se cierra, mientras el
usuario sigue hablando. Es el tercer camino, al lado de one-shot y streaming.

POR QUÉ EXISTE (medido en la Pi 5, `small` int8, 3 hilos)
---------------------------------------------------------
Con la ventana del encoder ya recortada (`streaming_core.short_window`), el
costo del dictado dejó de estar en el encoder y pasó al DECODER, que es
autorregresivo y por lo tanto escala con la cantidad de palabras:

    clip 4,8s   -> 2,61s totales: encoder 0,53s · resto 2,08s
    clip 47,1s  -> 25,1s totales: encoder 4,65s · resto 20,5s

De ahí sale que no queda nada más para exprimir de una sola pasada al final: la
única forma de que la ESPERA no crezca con lo que hablás es decodificar MIENTRAS
hablás. La CPU está ociosa igual — hoy el one-shot no hace nada hasta que soltás
la tecla.

POR QUÉ NO ES EL STREAMING QUE YA TENÍAMOS
------------------------------------------
`streaming_core.OnlineASR` (LocalAgreement-2) re-transcribe el buffer ENTERO en
cada pasada y decide qué palabras están "estables" comparando pasadas
consecutivas. Eso existe para mostrar texto ANTES de que termines la frase, y se
paga carísimo: cada palabra se decodifica muchas veces, hace falta aritmética de
timestamps (que se corren entre pasadas — ver el fix del 2026-09-09) y dedup por
n-gramas. Medido en esta Pi, `small` no le sigue el ritmo ni de cerca.

Acá cada pedazo de audio se decodifica UNA sola vez, entero, cuando ya está
completo. No hay timestamps, no hay dedup, no hay palabras a medio confirmar. Lo
que se resigna es el texto durante la frase; lo que se gana es que la espera al
soltar sea SOLO la última frase.

Y hay un detalle que lo hace encajar justo con el recorte de ventana: un
segmento de VAD **termina en silencio por construcción**, que es exactamente la
condición que le falta a una pasada de streaming (buffer cortado a mitad de
frase) y que la hacía degenerar — el modelo no emite <|endoftext|> y genera
hasta max_length. Acá el recorte es seguro por la misma razón que lo es en el
one-shot.
"""
from typing import Callable, List, Optional

import numpy as np

from streaming_core import (
    FULL_BEAM,
    SAMPLE_RATE,
    base_prompt_for,
    transcribe_one_shot,
)

# Silencio que tiene que haber DESPUÉS de una frase para darla por cerrada.
# 600ms es el valor que usan los dictados por VAD; por debajo se corta a mitad
# de frase cuando alguien duda, por encima se pierde la ventana para decodificar
# mientras el usuario sigue hablando.
MIN_SILENCE_MS = 600
# Una frase más corta que esto no se cierra sola: es un "eh" o un golpe de mesa.
MIN_SPEECH_MS = 300
# Sin ninguna pausa, un monólogo nunca cerraría un segmento y esto se degradaría
# a un one-shot gigante al final. A los 20s se corta igual: el peor caso pasa a
# ser "una pasada de 20s", no "una pasada de lo que dure el discurso".
MAX_SEGMENT_S = 20.0
# Margen de audio que se conserva a cada lado del tramo que marcó el VAD, para
# no comerse la primera consonante ni la última sílaba.
EDGE_PAD_S = 0.2
# Cuánto texto ya comiteado se le pasa como initial_prompt al segmento
# siguiente, para que la puntuación y las mayúsculas sigan la frase anterior.
PROMPT_TAIL_CHARS = 200


class SegmentASR:
    """Buffer de audio que se va vaciando a medida que las frases se cierran.

    Uso:
        seg = SegmentASR(model, "es")
        seg.insert_audio(chunk)   # desde el callback del micrófono
        seg.poll()                # cada tanto: decodifica lo que ya cerró
        texto = seg.finish()      # al soltar la tecla: decodifica la cola

    `on_error` recibe cualquier excepción de una decodificación: una frase que
    falla no puede matar el dictado entero (mismo criterio que OnlineASR).
    """

    def __init__(
        self,
        model,
        language: Optional[str],
        beam_size: int = FULL_BEAM,
        min_silence_ms: int = MIN_SILENCE_MS,
        min_speech_ms: int = MIN_SPEECH_MS,
        max_segment_s: float = MAX_SEGMENT_S,
        on_error: Optional[Callable[[BaseException], None]] = None,
        on_commit: Optional[Callable[[str], None]] = None,
        base_prompt: Optional[str] = None,
    ):
        self.model = model
        self.language = language
        self.beam_size = beam_size
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_segment_s = max_segment_s
        self.on_error = on_error
        self.on_commit = on_commit
        # La app arma su propio prompt (incluye el vocabulario que cargó el
        # usuario). Si no viene, el de la biblioteca.
        self.base_prompt = base_prompt or base_prompt_for(language)

        self.audio = np.array([], dtype=np.float32)
        self.committed: List[str] = []
        self.failed_segments = 0
        self.decoded_segments = 0

    # --- entrada ---
    def insert_audio(self, chunk: np.ndarray):
        self.audio = np.concatenate([self.audio, chunk])

    @property
    def buffer_seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    # --- VAD ---
    def _speech_spans(self, min_speech_ms: Optional[int] = None):
        """Tramos de habla del buffer actual, en muestras."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        return get_speech_timestamps(
            self.audio,
            VadOptions(
                min_silence_duration_ms=self.min_silence_ms,
                min_speech_duration_ms=(self.min_speech_ms if min_speech_ms is None
                                        else min_speech_ms),
                max_speech_duration_s=self.max_segment_s,
            ),
        )

    def _tail_has_speech(self) -> bool:
        """¿Queda algo que decir en la cola, o es puro silencio?

        Vale la pena preguntarlo: el caso bueno del dictado es terminar de
        hablar, esperar un beat y recién ahí soltar la tecla. Ese beat cierra la
        última frase, poll() la decodifica mientras la tecla sigue apretada, y lo
        único que queda en el buffer es silencio. Sin este chequeo, finish()
        pagaba una pasada ENTERA para transcribir ese silencio y la espera se iba
        de ~0,4s a 2,4s (medido con --release-delay 0.8). El VAD cuesta ~30ms.

        El umbral de habla mínima baja a 100ms a propósito: acá el riesgo no es
        decodificar de más sino perder un "sí" o un "no" al final.
        """
        try:
            return bool(self._speech_spans(min_speech_ms=100))
        except Exception:
            return True   # ante la duda, decodificar: perder audio es peor

    def _cut_point(self) -> Optional[int]:
        """Hasta dónde se puede decodificar YA, o None si no cerró nada.

        Una frase está cerrada si después de su final hay al menos
        `min_silence_ms` de audio: eso es el silencio que la cierra. La última
        frase del buffer casi nunca lo está — el usuario sigue hablando — y por
        eso se queda para la pasada siguiente o para finish().
        """
        spans = self._speech_spans()
        if not spans:
            # Todo silencio: no hay nada que decodificar, y el buffer se puede
            # tirar salvo la cola (podría ser el arranque de una palabra).
            return None
        silence = int(self.min_silence_ms * SAMPLE_RATE / 1000)
        closed = [s for s in spans if len(self.audio) - s["end"] >= silence]
        if not closed:
            # Sin pausas: si ya se hizo muy largo, se corta igual en el final
            # del último tramo para no volver a un one-shot gigante.
            if self.buffer_seconds >= self.max_segment_s:
                return spans[-1]["end"]
            return None
        return closed[-1]["end"]

    # --- decodificación ---
    def _prompt(self) -> str:
        tail = " ".join(self.committed)[-PROMPT_TAIL_CHARS:]
        return f"{self.base_prompt} {tail}".strip()

    def _decode(self, audio: np.ndarray) -> str:
        try:
            return transcribe_one_shot(
                self.model, audio, self.language, beam_size=self.beam_size,
                initial_prompt=self._prompt(),
            )
        except Exception as exc:  # noqa: BLE001 - una frase mala no mata el dictado
            self.failed_segments += 1
            if self.on_error is not None:
                self.on_error(exc)
            return ""

    def poll(self) -> str:
        """Decodifica lo que ya haya cerrado. Devuelve el texto nuevo (o "")."""
        cut = self._cut_point()
        if cut is None:
            return ""
        pad = int(EDGE_PAD_S * SAMPLE_RATE)
        chunk = self.audio[: min(len(self.audio), cut + pad)]
        # El buffer arranca donde termina lo consumido: cada muestra se
        # decodifica UNA vez.
        self.audio = self.audio[cut:]
        text = self._decode(chunk)
        self.decoded_segments += 1
        if text:
            self.committed.append(text)
            if self.on_commit is not None:
                self.on_commit(text)
        return text

    def finish(self) -> str:
        """Al soltar la tecla: decodifica la cola y devuelve el texto completo."""
        if len(self.audio) >= int(0.2 * SAMPLE_RATE) and self._tail_has_speech():
            text = self._decode(self.audio)
            self.decoded_segments += 1
            if text:
                self.committed.append(text)
                if self.on_commit is not None:
                    self.on_commit(text)
        self.audio = np.array([], dtype=np.float32)
        return self.text()

    # --- lectura ---
    def text(self) -> str:
        return " ".join(t for t in self.committed if t).strip()
