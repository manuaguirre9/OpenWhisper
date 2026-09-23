"""question_marks.fix: casos sacados del log real y de las pruebas con TTS."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from question_marks import fix  # noqa: E402


@pytest.mark.parametrize("dicho, esperado", [
    # Del log: abrió y no cerró.
    ("Hola, hola amigo, ¿cómo estás", "Hola, hola amigo, ¿cómo estás?"),
    # Del log: sí/no sin ningún signo.
    ("Te parece una buena idea", "¿Te parece una buena idea?"),
    ("y por qué?", "¿y por qué?"),
    # Interrogativo después de una coma: el signo abre ahí, no al principio.
    ("Probé las dos opciones, cuál te gusta más", "Probé las dos opciones, ¿cuál te gusta más?"),
    ("Quiero cambiar el color. Qué pensás", "Quiero cambiar el color. ¿Qué pensás?"),
    ("Cuándo vamos a terminar esto.", "¿Cuándo vamos a terminar esto?"),
    ("Podés revisar el código del globo", "¿Podés revisar el código del globo?"),
    ("Y cómo lo hago", "¿Y cómo lo hago?"),
    ("Lo hice así, te parece", "Lo hice así, ¿te parece?"),
    ("Hay alguna forma de hacerlo más rápido", "¿Hay alguna forma de hacerlo más rápido?"),
])
def test_completa_preguntas(dicho, esperado):
    assert fix(dicho) == esperado


@pytest.mark.parametrize("dicho", [
    "¿Cómo estás?",                                 # ya está bien
    "No sé qué hacer con esto",                     # indirecta: sin signos
    "Qué bueno que viniste",                        # exclamación
    "Qué lindo día.",
    "Podemos hacer que toda la corriente de aire pase por acá",
    "Tenés que denunciar el accidente para que lo cubra el seguro",
    "Lo hago como vos digas, cuando puedas",        # sin tilde: no pregunta
    "Qué increíble!",
    "Hay algunas cosas que no me cierran",
    "",
])
def test_no_toca_lo_que_no_es_pregunta(dicho):
    assert fix(dicho) == dicho


def test_solo_la_frase_que_pregunta():
    assert fix("Ya lo probé. Funciona bien. Dónde lo subo") == \
        "Ya lo probé. Funciona bien. ¿Dónde lo subo?"
