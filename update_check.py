"""
Avisa cuando sale algo nuevo del lado del modelo de voz. Sin UI.

QUÉ MIRA
--------
Tres fuentes públicas, sin cuenta ni clave:

1. Conversiones para sherpa-onnx en HuggingFace, solo de los mantenedores de
   k2-fsa (csukuangfj, csukuangfj2). Es lo que IMPORTA: un modelo recién se
   puede usar en OpenWhisper cuando aparece acá. Las variantes de una misma
   conversión (80/160/560/1120 ms, int8 o no) cuentan como UNA novedad.
2. Modelos de reconocimiento de voz que publica NVIDIA. Todavía no se pueden
   usar, pero anticipan lo que va a venir (así apareció el "unified" en 2026-08).
3. Versiones nuevas de sherpa-onnx en PyPI (de ahí vienen cosas como el beam
   search que este modelo no tenía en 1.13).

CÓMO AVISA
----------
Una vez por semana como mucho. La PRIMERA vez que ve una fuente solo anota lo
que ya existe y no avisa nada: si no, el primer arranque mostraría años de
historia. Después avisa cada cosa una sola vez. El estado vive en
`novedades.json`, al lado de config.json.

No descarga ni instala nada: avisar y decidir es del usuario.
"""
import json
import os
import re
import time
import urllib.request
from typing import Callable, List, Optional

from config_manager import CONFIG_DIR

STATE_FILE = os.path.join(CONFIG_DIR, "novedades.json")
INTERVALO_S = 7 * 24 * 3600

_HF = "https://huggingface.co/api/models"
_CONVERSORES = ("csukuangfj", "csukuangfj2", "k2-fsa")
_BUSQUEDAS = ("sherpa-onnx-nemotron", "sherpa-onnx-parakeet", "sherpa-onnx-nemo")
# Solo las familias de NVIDIA que pueden reemplazar a la actual.
_FAMILIAS = re.compile(r"nemotron|parakeet|canary|unified", re.I)

Fetch = Callable[[str], object]


def _fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "OpenWhisper"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def familia(repo_id: str):
    """`csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11`
    → (`nemotron-3.5-asr-streaming-0.6b`, `2026-06-11`).

    Así las ocho variantes de una conversión son una sola novedad."""
    name = repo_id.split("/", 1)[-1]
    name = re.sub(r"^sherpa-onnx-", "", name)
    m = re.search(r"-(\d{4}-\d{2}-\d{2})$", name)
    fecha = m.group(1) if m else ""
    if m:
        name = name[:m.start()]
    name = re.sub(r"-(int8|fp16|fp32|\d+ms)(?=-|$)", "", name)
    return name, fecha


def _conversiones(fetch: Fetch) -> dict:
    out = {}
    for q in _BUSQUEDAS:
        for m in fetch(f"{_HF}?search={q}&sort=createdAt&direction=-1&limit=100") or []:
            rid = m.get("id", "")
            if rid.split("/")[0] not in _CONVERSORES:
                continue
            fam, fecha = familia(rid)
            if not _FAMILIAS.search(fam):    # p. ej. los giga-am rusos: no nos sirven
                continue
            key = f"{fam}@{fecha}"
            out.setdefault(key, {
                "titulo": "Hay una versión de modelo lista para sherpa-onnx",
                "texto": f"{fam} ({fecha or 'sin fecha'})",
                "url": f"https://huggingface.co/{rid}",
            })
    return out


def _nvidia(fetch: Fetch) -> dict:
    url = (f"{_HF}?author=nvidia&pipeline_tag=automatic-speech-recognition"
           "&sort=createdAt&direction=-1&limit=50")
    return {m["id"]: {
        "titulo": "NVIDIA publicó un modelo de voz nuevo",
        "texto": f"{m['id'].split('/')[-1]} — todavía no se puede usar hasta que "
                 "salga convertido para sherpa-onnx",
        "url": f"https://huggingface.co/{m['id']}",
    } for m in (fetch(url) or []) if m.get("id")}


def _sherpa(fetch: Fetch) -> dict:
    data = fetch("https://pypi.org/pypi/sherpa-onnx/json") or {}
    v = (data.get("info") or {}).get("version")
    if not v:
        return {}
    try:
        import sherpa_onnx
        tengo = getattr(sherpa_onnx, "__version__", "?")
    except Exception:  # noqa: BLE001
        tengo = "?"
    return {v: {
        "titulo": f"Salió sherpa-onnx {v}",
        "texto": f"OpenWhisper usa {tengo}. Puede traer soporte para modelos nuevos.",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases",
    }}


FUENTES = {"conversiones": _conversiones, "nvidia": _nvidia, "sherpa": _sherpa}


def load_state(path: str = STATE_FILE) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state: dict, path: str = STATE_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def check(force: bool = False, fetch: Fetch = _fetch_json,
          path: str = STATE_FILE, now: Optional[float] = None) -> Optional[List[dict]]:
    """Devuelve las novedades nuevas (lista, quizá vacía), o None si no tocaba
    revisar todavía. Si una fuente falla se la saltea: la próxima vez vuelve a
    intentarse, y lo que ya se vio no se repite."""
    now = time.time() if now is None else now
    state = load_state(path)
    ultima = state.get("ultima_revision")
    if not force and ultima is not None and now - ultima < INTERVALO_S:
        return None

    nuevas, vistos = [], state.setdefault("vistos", {})
    fallas = 0
    for nombre, fuente in FUENTES.items():
        try:
            items = fuente(fetch)
        except Exception as e:  # noqa: BLE001 - sin red no es un error de la app
            print(f"[Novedades] no pude consultar {nombre}: {e}")
            fallas += 1
            continue
        if nombre not in vistos:
            vistos[nombre] = sorted(items)    # primera vez: línea de base, sin avisar
            continue
        conocidos = set(vistos[nombre])
        for key in sorted(items):
            if key not in conocidos:
                nuevas.append(items[key])
                vistos[nombre].append(key)

    if fallas < len(FUENTES):
        state["ultima_revision"] = now
    save_state(state, path)
    for n in nuevas:
        print(f"[Novedades] {n['titulo']}: {n['texto']} → {n['url']}")
    if fallas == len(FUENTES):
        raise ConnectionError("no pude consultar ninguna fuente")
    return nuevas
