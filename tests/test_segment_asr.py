"""
Tests del dictado por segmentos.

Lo que se fija acá es la regla de corte —cuándo una frase se da por cerrada— y
el invariante que hace que esto sea barato: **cada muestra de audio se
decodifica UNA sola vez**. Si un refactor rompe eso, el modo segmento se
convierte en el streaming por buffer que justamente vino a reemplazar, y el
síntoma (más CPU, mismo texto) no se ve en ninguna salida.

El VAD real (silero) se stubea: acá se prueba la lógica de cierre, no el
detector de voz, y un test unitario no puede depender de que un modelo ONNX
clasifique ruido sintético como habla.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from segment_asr import SegmentASR  # noqa: E402
from streaming_core import SAMPLE_RATE  # noqa: E402


class FakeModel:
    """Devuelve un texto por llamada y anota cuánto audio recibió."""

    def __init__(self, textos=None):
        self.textos = list(textos or [])
        self.recibido = []
        self.prompts = []

    def transcribe(self, audio, **kw):
        self.recibido.append(len(audio) / SAMPLE_RATE)
        self.prompts.append(kw.get("initial_prompt", ""))
        texto = self.textos.pop(0) if self.textos else "texto"

        class Seg:
            pass

        seg = Seg()
        seg.text = texto
        return [seg], None


def _spans(*pares):
    """[(inicio_s, fin_s), ...] -> lo que devolvería el VAD, en muestras."""
    return [{"start": int(a * SAMPLE_RATE), "end": int(b * SAMPLE_RATE)} for a, b in pares]


def _armar(model, audio_s, spans, **kw):
    seg = SegmentASR(model, "es", **kw)
    seg.insert_audio(np.zeros(int(audio_s * SAMPLE_RATE), dtype=np.float32))
    seg._speech_spans = lambda: spans
    return seg


def test_no_cierra_mientras_sigue_hablando():
    """El habla llega hasta el final del buffer: todavía no hay pausa."""
    seg = _armar(FakeModel(), 5.0, _spans((0.5, 5.0)))
    assert seg._cut_point() is None
    assert seg.poll() == ""


def test_no_cierra_con_una_pausa_demasiado_corta():
    """300ms de silencio es dudar, no terminar la frase."""
    seg = _armar(FakeModel(), 5.0, _spans((0.5, 4.7)))
    assert seg._cut_point() is None


def test_cierra_con_silencio_suficiente():
    m = FakeModel(["Hola que tal."])
    seg = _armar(m, 5.0, _spans((0.5, 4.0)))
    assert seg.poll() == "Hola que tal."
    assert seg.text() == "Hola que tal."


def test_cada_muestra_se_decodifica_una_sola_vez():
    """El buffer arranca donde terminó lo consumido. Sin esto, el costo vuelve
    a ser cuadrático como en el streaming por buffer."""
    m = FakeModel(["uno", "dos"])
    seg = _armar(m, 10.0, _spans((0.5, 4.0), (5.0, 8.5)))
    seg.poll()                       # cierra hasta 8.5s
    assert seg.buffer_seconds == pytest.approx(10.0 - 8.5, abs=0.05)
    seg.insert_audio(np.zeros(int(3 * SAMPLE_RATE), dtype=np.float32))
    seg._speech_spans = lambda: _spans((0.2, 1.5))
    seg.poll()
    # lo decodificado no puede sumar más que el audio que entró
    assert sum(m.recibido) <= 13.0 + 0.5


def test_solo_se_cierra_el_ultimo_tramo_con_pausa():
    """Dos frases cerradas se decodifican juntas: son audio contiguo y whisper
    rinde mejor con la frase entera que con pedazos."""
    m = FakeModel(["uno dos"])
    seg = _armar(m, 10.0, _spans((0.5, 4.0), (5.0, 8.5)))
    seg.poll()
    assert len(m.recibido) == 1
    assert m.recibido[0] == pytest.approx(8.7, abs=0.05)   # 8.5 + el margen


def test_monologo_sin_pausas_se_corta_igual():
    """Sin este corte, hablar 3 minutos sin respirar deja TODO para el final."""
    m = FakeModel(["algo"])
    seg = _armar(m, 25.0, _spans((0.0, 25.0)), max_segment_s=20.0)
    assert seg._cut_point() is not None
    assert seg.poll() == "algo"


def test_silencio_puro_no_decodifica_nada():
    m = FakeModel()
    seg = _armar(m, 5.0, [])
    assert seg.poll() == ""
    assert m.recibido == []


def test_un_segmento_que_falla_no_mata_el_dictado():
    class Rota(FakeModel):
        def transcribe(self, audio, **kw):
            raise RuntimeError("boom")

    errores = []
    seg = _armar(Rota(), 5.0, _spans((0.5, 4.0)), on_error=errores.append)
    assert seg.poll() == ""
    assert seg.failed_segments == 1 and len(errores) == 1
    # y sigue aceptando audio
    seg.insert_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))


def test_el_prompt_arrastra_lo_ya_comiteado():
    """Sin esto, cada frase arranca de cero y la puntuación no engancha."""
    m = FakeModel(["Primera frase.", "segunda."])
    seg = _armar(m, 10.0, _spans((0.5, 4.0), (5.0, 8.5)))
    seg.poll()
    seg.insert_audio(np.zeros(int(3 * SAMPLE_RATE), dtype=np.float32))
    seg._speech_spans = lambda: _spans((0.2, 1.5))
    seg.poll()
    assert "Primera frase." in m.prompts[1]


def test_finish_decodifica_la_cola_y_devuelve_todo():
    m = FakeModel(["uno", "y dos"])
    seg = _armar(m, 10.0, _spans((0.5, 4.0), (5.0, 8.5)))
    seg.poll()
    assert seg.finish() == "uno y dos"
    assert seg.buffer_seconds == 0


def test_finish_sin_audio_util_no_llama_al_modelo():
    m = FakeModel()
    seg = SegmentASR(m, "es")
    seg.insert_audio(np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32))
    assert seg.finish() == ""
    assert m.recibido == []


def test_finish_no_decodifica_una_cola_de_puro_silencio():
    """El caso bueno del dictado: terminás, esperás un beat, soltás. El beat ya
    cerró la última frase; la cola es silencio y pagarle una pasada entera
    llevaba la espera de ~0,4s a 2,4s (medido)."""
    m = FakeModel(["Hola que tal."])
    seg = _armar(m, 6.0, _spans((0.5, 4.0)))
    seg.poll()
    seg._speech_spans = lambda min_speech_ms=None: []     # la cola es silencio
    assert seg.finish() == "Hola que tal."
    assert len(m.recibido) == 1


def test_finish_si_decodifica_una_cola_con_habla():
    m = FakeModel(["Hola.", "sí"])
    seg = _armar(m, 6.0, _spans((0.5, 4.0)))
    seg.poll()
    seg._speech_spans = lambda min_speech_ms=None: _spans((0.3, 0.6))
    assert seg.finish() == "Hola. sí"
    assert len(m.recibido) == 2


def test_ante_un_vad_roto_finish_decodifica_igual():
    """Falla perdiendo tiempo, nunca perdiendo audio."""
    m = FakeModel(["algo"])
    seg = SegmentASR(m, "es")
    seg.insert_audio(np.zeros(int(2 * SAMPLE_RATE), dtype=np.float32))

    def rota(min_speech_ms=None):
        raise RuntimeError("vad caído")

    seg._speech_spans = rota
    assert seg.finish() == "algo"
