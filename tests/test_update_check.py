"""update_check: línea de base silenciosa, avisar una sola vez, agrupar variantes."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import update_check as uc  # noqa: E402

CONV = "https://huggingface.co/api/models?search="


def make_fetch(convs, nvidia=(), pypi="1.13.8"):
    def fetch(url):
        if "pypi.org" in url:
            return {"info": {"version": pypi}}
        if "author=nvidia" in url:
            return [{"id": i} for i in nvidia]
        if url.startswith(CONV + "sherpa-onnx-nemotron"):
            return [{"id": i} for i in convs]
        return []
    return fetch


def test_familia_agrupa_variantes():
    a = uc.familia("csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11")
    b = uc.familia("csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-2026-06-11")
    assert a == b == ("nemotron-3.5-asr-streaming-0.6b", "2026-06-11")


def test_primera_vez_no_avisa_y_despues_avisa_una_vez(tmp_path):
    p = str(tmp_path / "n.json")
    base = ["csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"]
    assert uc.check(fetch=make_fetch(base, ["nvidia/a"]), path=p, now=0) == []

    nuevo = base + [
        "csukuangfj2/sherpa-onnx-nemotron-labs-unified-asr-0.6b-160ms-2026-11-20",
        "csukuangfj2/sherpa-onnx-nemotron-labs-unified-asr-0.6b-560ms-int8-2026-11-20",
        "alguien/sherpa-onnx-nemotron-copia-2026-11-21",        # no es de k2-fsa
    ]
    f = make_fetch(nuevo, ["nvidia/a", "nvidia/b"], pypi="2.0.1")
    assert uc.check(fetch=f, path=p, now=1) is None             # no pasó una semana
    out = uc.check(fetch=f, path=p, now=uc.INTERVALO_S + 1)
    textos = [n["texto"] for n in out]
    assert len(out) == 3                                        # 1 conversión + 1 nvidia + sherpa
    assert any("nemotron-labs-unified-asr-0.6b (2026-11-20)" in t for t in textos)
    assert not any("copia" in t for t in textos)
    assert uc.check(fetch=f, path=p, force=True, now=uc.INTERVALO_S + 2) == []


def test_sin_red_no_marca_revisado(tmp_path):
    p = str(tmp_path / "n.json")

    def caido(url):
        raise OSError("sin red")
    with pytest.raises(ConnectionError):
        uc.check(fetch=caido, path=p, now=0)
    assert "ultima_revision" not in uc.load_state(p)
