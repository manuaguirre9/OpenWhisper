"""
Vocabulario aprendido: los pares (lo que escribió mal → lo que dijiste) que
el usuario corrige a mano en el globo, al terminar una toma.

POR QUÉ EXISTE
--------------
MEDIDO sobre las 1023 frases de openwhisper.log: las palabras propias de este
usuario el modelo NUNCA las escribe bien. "ReSpeaker" salió 0 veces correcta
(`re speaker`, `retspicher`), "Faraday" 0 veces (`farada` x2), "frontend" 0
veces (`fronten`), "barge-in" 0 veces (`barch` x2). Solo "LinkedIn" acertó
algunas (4 bien, 5 `inkedin`).

Por eso NO alcanza con minar el historial: no hay ninguna grafía correcta de
donde aprender, y los errores consistentes (`farada` dos veces igual) se
fosilizarían. Hace falta que el usuario diga cuál es la palabra, una vez.

Una corrección sirve para dos cosas, y las dos importan:
  1. Sesgar el decoder (`set_keyterms`): la próxima vez es más probable que
     salga bien de entrada. Por eso se guarda con las mayúsculas exactas: los
     docs de Moonshine dicen "match the capitalization you want to see".
  2. Arreglar el texto de esta toma en adelante (`apply`): aunque el modelo
     vuelva a errarle, el texto que se inyecta ya sale corregido.

Es deliberadamente VISIBLE y REVERSIBLE: se listan en Configuración y se
borran de a una. Nada se aprende a tus espaldas.
"""
import json
import os
import re
import threading

from config_manager import CONFIG_DIR

CORRECTIONS_FILE = os.path.join(CONFIG_DIR, "corrections.json")

# Más que esto empieza a costar exactitud en las palabras que NO pediste
# (docs de set_context: "a long list costs accuracy"). Se descartan las más
# viejas, que es lo mismo que decir las que menos usás últimamente.
MAX_TERMS = 120

_lock = threading.Lock()


def _normalize(word: str) -> str:
    """La clave de búsqueda: sin puntuación de los bordes y en minúscula.

    El usuario hace clic sobre `farada` en "jaula de farada y", y el token que
    llega puede traer una coma o un punto pegados. Puede ser también más de una
    palabra ("a cerca"), y entonces los espacios de adentro se normalizan a uno
    solo: al buscarla después se acepta cualquier separación.
    """
    clean = (word or "").strip().strip(".,;:¿?¡!()[]\"'«»…").lower()
    return " ".join(clean.split())


def load() -> dict:
    """{forma_errada_normalizada: forma_correcta}. Orden = antigüedad."""
    try:
        with open(CORRECTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(v, str) and k and v}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - un archivo roto no mata la app
        print(f"[Correcciones] No pude leer {CORRECTIONS_FILE}: {exc}")
        return {}


def save(pairs: dict):
    with _lock:
        try:
            with open(CORRECTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"[Correcciones] No pude guardar: {exc}")


def add(wrong: str, right: str) -> dict:
    """Guarda un par y devuelve el diccionario completo ya actualizado.

    Devuelve el diccionario (en vez de None) para que quien llama pueda
    refrescar el vocabulario del motor sin volver a leer el archivo.
    """
    key = _normalize(wrong)
    value = (right or "").strip()
    if not key or not value or key == value.lower():
        return load()
    pairs = load()
    pairs.pop(key, None)          # reinsertar al final: la más reciente sobrevive
    pairs[key] = value
    while len(pairs) > MAX_TERMS:
        pairs.pop(next(iter(pairs)))
    save(pairs)
    print(f"[Correcciones] Aprendido: {key!r} → {value!r} ({len(pairs)} en total)")
    return pairs


def remove(wrong: str) -> dict:
    pairs = load()
    if pairs.pop(_normalize(wrong), None) is not None:
        save(pairs)
    return pairs


def terms(pairs: dict = None) -> list:
    """Las grafías CORRECTAS, que es lo que quiere set_keyterms."""
    pairs = load() if pairs is None else pairs
    out = []
    for value in pairs.values():
        if value not in out:
            out.append(value)
    return out


def vocabulary_string(manual: str = "", pairs: dict = None) -> str:
    """Une el vocabulario escrito a mano con el aprendido, sin repetir.

    El motor recibe un solo string separado por comas (ver
    moonshine_engine.split_vocabulary), así que acá se arma uno solo.
    """
    seen, out = set(), []
    for chunk in list(_split(manual)) + terms(pairs):
        low = chunk.lower()
        if chunk and low not in seen:
            seen.add(low)
            out.append(chunk)
    return ", ".join(out)


def _split(text: str):
    for raw in (text or "").replace(";", ",").replace("\n", ",").split(","):
        term = raw.strip()
        if term:
            yield term


def apply(text: str, pairs: dict = None) -> str:
    """Reemplaza las formas erradas conocidas por la correcta, palabra entera.

    Sin distinguir mayúsculas al buscar, pero escribiendo SIEMPRE la grafía
    guardada: si corregiste `inkedin` → `LinkedIn`, también arregla `Inkedin`.

    Las claves de varias palabras ("a cerca" → "acerca") se buscan aceptando
    cualquier cantidad de espacios entre ellas, y se prueban ANTES que las de
    una sola: si no, una regla corta podría comerse parte de una larga y la
    larga ya no encontraría su texto.
    """
    pairs = load() if pairs is None else pairs
    if not text or not pairs:
        return text
    for wrong in sorted(pairs, key=lambda k: (-len(k.split()), -len(k))):
        right = pairs[wrong]
        pattern = r"\s+".join(re.escape(part) for part in wrong.split())
        try:
            text = re.sub(rf"(?<!\w){pattern}(?!\w)", right.replace("\\", "\\\\"),
                          text, flags=re.IGNORECASE)
        except re.error:
            continue
    return text
