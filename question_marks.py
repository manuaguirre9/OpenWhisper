"""
Completar los signos de pregunta que el modelo no pone.

POR QUÉ EXISTE
--------------
Nemotron puntúa, pero en frases cortas casi nunca cierra una pregunta. MEDIDO
con ocho preguntas dichas por TTS y 1s de silencio al final: solo una salió
con signos (`¿cómo estás?`). Más silencio no cambió nada, así que no es que le
falte contexto: el modelo no lo decide. En el log real pasa lo mismo:
`Te parece una buena idea` sin nada, `¿cómo estás` abierta y sin cerrar.

Lo que SÍ hace bien es acentuar: escribe `qué pensás` y `cuál te gusta` con
tilde, y `que`/`cual` sin tilde cuando no preguntan. Ese acento es la señal
que se usa acá, además de unos pocos arranques que en castellano rioplatense
son casi siempre pregunta (`te parece`, `se entiende`, `podés`).

Deliberadamente CONSERVADOR: una pregunta sin signos se lee igual; un signo
puesto donde no va se nota mucho más. Por eso:
  - solo mira el comienzo de una frase o de lo que viene después de una coma
    (`No sé qué hacer` es pregunta indirecta y no lleva signos);
  - `qué bueno`, `qué lindo`… son exclamaciones, no preguntas;
  - no toca frases que ya terminan en `!` o `…`.
"""
import re

_INTERROGATIVOS = {
    "qué", "cómo", "cuándo", "dónde", "adónde", "cuál", "cuáles",
    "quién", "quiénes", "cuánto", "cuánta", "cuántos", "cuántas",
}

# Arranques de pregunta de sí/no. Sin palabra interrogativa, lo único que
# distingue `te parece bien` de `¿te parece bien?` es la entonación, que el
# texto no tiene. Estos casi nunca son afirmación. `podemos` NO está: en el
# log aparece afirmando (`podemos hacer que toda la corriente de aire…`).
_ARRANQUES_SI_NO = (
    "te parece", "les parece", "no te parece", "se entiende",
    "me podés", "me podrías", "podés", "podrías", "sabés", "tenés idea",
    "hay alguna", "hay algún",
)

# `qué bueno que viniste` es exclamación. Lista corta: las que salen dictando.
_EXCLAMATIVAS = {
    "bueno", "buena", "buenísimo", "lindo", "linda", "bien", "mal", "raro",
    "loco", "locura", "suerte", "pena", "lástima", "genial", "horror",
    "divertido", "increíble", "difícil", "fácil", "manera", "cantidad",
}

# Muletillas que pueden ir antes de la pregunta: `y por qué`, `pero cómo`.
_PREVIAS = {"y", "pero", "entonces", "che", "o"}

_FIN_DE_FRASE = re.compile(r"(?<=[.!?…])\s+")


def _es_pregunta(clausula: str) -> bool:
    palabras = clausula.lower().split()
    if palabras and palabras[0] in _PREVIAS:
        palabras = palabras[1:]
    if not palabras:
        return False
    if palabras[0] == "por" and len(palabras) > 1 and palabras[1] == "qué":
        return True
    if palabras[0] in _INTERROGATIVOS:
        siguiente = palabras[1] if len(palabras) > 1 else ""
        return not (palabras[0] == "qué" and siguiente in _EXCLAMATIVAS)
    inicio = " ".join(palabras)
    return any(inicio == a or inicio.startswith(a + " ") for a in _ARRANQUES_SI_NO)


def _inicios_de_clausula(frase: str):
    """Posiciones donde empieza la frase o algo después de una coma."""
    yield 0
    for m in re.finditer(r",\s+", frase):
        yield m.end()


def _cerrar(frase: str) -> str:
    if frase.endswith("."):
        return frase[:-1] + "?"
    return frase + "?"


def _arreglar_frase(frase: str) -> str:
    if not frase or frase[-1] in "!…":
        return frase
    abre, cierra = "¿" in frase, frase.endswith("?")
    if abre and not cierra:
        return _cerrar(frase)
    if abre or ("?" in frase and not cierra):
        return frase
    for i in _inicios_de_clausula(frase):
        if _es_pregunta(frase[i:]):
            return frase[:i] + "¿" + (_cerrar(frase[i:]) if not cierra else frase[i:])
    if cierra:
        return "¿" + frase     # tiene `?` y ninguna pista de dónde empieza
    return frase


def fix(text: str) -> str:
    """Agrega `¿`/`?` donde el texto deja claro que hay una pregunta."""
    if not text or not text.strip():
        return text
    return " ".join(_arreglar_frase(f) for f in _FIN_DE_FRASE.split(text.strip()))
