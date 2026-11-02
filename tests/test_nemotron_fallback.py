"""NemotronSession: la lectura automática manda, salvo que se vaya a otro alfabeto."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nemotron_engine as ne  # noqa: E402


@pytest.mark.parametrize("texto, extranjero", [
    ("Hola, ¿cómo andás? Subí el backup", False),
    ("Then I ran the installer — 3 times!", False),
    ("Чика десі во- gasto podjalar los resultados", True),
    ("まあ、これを", True),
    ("مرحبا", True),
    ("", False),
])
def test_foreign_script(texto, extranjero):
    assert ne.foreign_script(texto) is extranjero


class _Stream:
    def __init__(self):
        self.options, self.fed = {}, 0

    def set_option(self, k, v):
        self.options[k] = v

    def accept_waveform(self, sr, a):
        self.fed += len(a)

    def input_finished(self):
        pass


class _Rec:
    """Devuelve un texto por stream: el primero creado es el automático."""
    def __init__(self, auto, fijado):
        self.textos, self.streams = [auto, fijado], []

    def create_stream(self):
        s = _Stream()
        self.streams.append(s)
        return s

    def is_ready(self, s):
        return False

    def decode_streams(self, ss):
        pass

    def get_result(self, s):
        return self.textos[self.streams.index(s)]


def _toma(rec, language):
    s = ne.NemotronSession(rec, language=language)
    s.insert_audio(np.zeros(1600, dtype=np.float32))
    return s, s.finish()


def test_usa_la_automatica_si_es_latina_y_no_abre_otra():
    rec = _Rec("Then you can run the installer", "Den llu can ran de instaler")
    s, texto = _toma(rec, "es")
    assert texto == "Then you can run the installer" and not s.used_fallback
    assert len(rec.streams) == 1                  # sin costo extra
    assert rec.streams[0].options == {}           # la automática va sin prompt


def test_cae_al_idioma_fijado_si_sale_otro_alfabeto():
    rec = _Rec("Чика десі во- gasto", "Ojalá que esto ayude a mejorar")
    s, texto = _toma(rec, "es")
    assert texto == "Ojalá que esto ayude a mejorar" and s.used_fallback
    assert rec.streams[1].options == {"language": "es-US"}
    # se puso al día: oyó lo mismo que la automática, cola de silencio incluida
    assert rec.streams[0].fed == rec.streams[1].fed


def test_respaldo_abierto_recien_al_cerrar_tambien_se_pone_al_dia():
    class _RecTarde(_Rec):
        def get_result(self, s):    # se va de alfabeto recién con el final
            return "Чика" if self.streams.index(s) == 0 and s.fed > 1600 else \
                ("Ojalá" if self.streams.index(s) == 1 else "Oja")
    rec = _RecTarde("", "")
    s, texto = _toma(rec, "es")
    assert texto == "Ojalá" and rec.streams[0].fed == rec.streams[1].fed


def test_auto_no_crea_segunda_lectura():
    rec = _Rec("まあ、これを", "nunca")
    s, texto = _toma(rec, "auto")
    assert len(rec.streams) == 1 and texto == "まあ、これを"
