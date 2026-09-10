"""MoonshineSession sin Moonshine: un stream falso dispara los eventos."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moonshine_engine import (  # noqa: E402
    MoonshineSession,
    resolve_language,
    split_vocabulary,
)
import text_injector  # noqa: E402


class FakeStream:
    def __init__(self):
        self.listener = None
        self.started = False
        self.stopped = False
        self.closed = False
        self.audio = []
        self.on_stop = None

    def add_listener(self, listener):
        self.listener = listener

    def start(self):
        self.started = True

    def add_audio(self, data, sample_rate=16000):
        assert sample_rate == 16000
        assert isinstance(data, list)
        self.audio.extend(data)

    def stop(self):
        self.stopped = True
        if self.on_stop:
            self.on_stop()

    def close(self):
        self.closed = True

    # helpers para simular a Moonshine
    def partial(self, line_id, text):
        self.listener.on_line_text_changed(SimpleNamespace(line=SimpleNamespace(line_id=line_id, text=text)))

    def complete(self, line_id, text):
        self.listener.on_line_completed(SimpleNamespace(line=SimpleNamespace(line_id=line_id, text=text)))


def _session(**cb):
    stream = FakeStream()
    return stream, MoonshineSession(stream, **cb)


def test_los_parciales_se_muestran_pero_no_se_entregan_como_linea():
    partials, lines = [], []
    stream, s = _session(on_partial=partials.append, on_line=lines.append)
    stream.partial(1, "y el agua")
    stream.partial(1, "tenés que denunciar")
    assert partials == ["y el agua", "tenés que denunciar"]
    assert lines == []
    assert s.pending_text() == "tenés que denunciar"


def test_una_linea_completa_se_entrega_una_sola_vez():
    lines = []
    stream, s = _session(on_line=lines.append)
    stream.partial(1, "hola")
    stream.complete(1, "Hola, ¿qué tal?")
    stream.complete(1, "Hola, ¿qué tal?")      # Moonshine puede repetir el evento
    assert lines == ["Hola, ¿qué tal?"]
    assert s.pending_text() == ""
    assert s.text() == "Hola, ¿qué tal?"


def test_finish_devuelve_todo_y_pending_solo_lo_que_no_salio_por_on_line():
    lines = []
    stream, s = _session(on_line=lines.append)
    stream.complete(1, "Primera frase.")
    stream.partial(2, "segunda a medio")
    # Al parar, Moonshine cierra lo que puede. Acá simulamos que la 2 NO cierra
    # y aparece una 3 nueva que sí.
    stream.on_stop = lambda: stream.complete(3, "Tercera.")
    full = s.finish()
    assert full == "Primera frase. segunda a medio Tercera."
    assert lines == ["Primera frase.", "Tercera."]
    assert s.pending_text() == "segunda a medio"
    assert stream.stopped and stream.closed


def test_insert_audio_convierte_a_lista_float_y_se_ignora_despues_de_finish():
    stream, s = _session()
    s.insert_audio(np.array([0.1, -0.2], dtype=np.float32))
    assert len(stream.audio) == 2
    s.finish()
    s.insert_audio(np.zeros(10, dtype=np.float32))
    assert len(stream.audio) == 2


def test_un_error_del_stream_no_rompe_la_toma():
    errors = []
    stream, s = _session(on_error=errors.append)
    stream.listener.on_error(SimpleNamespace(error=RuntimeError("boom")))
    assert len(errors) == 1 and s.errors


def test_resolve_language_cae_a_espanol_cuando_no_hay_modelo():
    assert resolve_language("es") == "es"
    assert resolve_language("en") == "en"
    assert resolve_language("auto") == "es"
    assert resolve_language(None) == "es"
    assert resolve_language("pt") == "es"
    assert resolve_language("es-AR") == "es"


def test_split_vocabulary_separa_por_comas_saltos_y_puntoycoma_sin_duplicar():
    assert split_vocabulary("Nobi, LayerCake\ninox 316; Nobi,,") == ["Nobi", "LayerCake", "inox 316"]
    assert split_vocabulary("") == []


def test_wait_modifiers_released_espera_y_da_false_si_se_agota():
    state = {"n": 3}

    def down():
        state["n"] -= 1
        return state["n"] > 0

    assert text_injector.wait_modifiers_released(timeout=1.0, poll=0.0, _down=down) is True
    assert text_injector.wait_modifiers_released(timeout=0.01, poll=0.0, _down=lambda: True) is False
