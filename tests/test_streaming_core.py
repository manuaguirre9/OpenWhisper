"""
Unit tests for the pure streaming logic and the benchmark's WER.

These are the functions that break in a refactor, not at birth: the
LocalAgreement committer, the word normalizer, the buffer trim, and the
error guard that keeps one bad decoding pass from killing the dictation
thread. None of them need a microphone or a model.

    pip install -r requirements-dev.txt
    pytest
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

from streaming_core import (  # noqa: E402
    SAMPLE_RATE,
    HypothesisBuffer,
    OnlineASR,
    base_prompt_for,
    join_words,
    norm_word,
)
import bench_dictation as bench  # noqa: E402


# ----------------------------------------------------------- fake model --

class FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class FakeSegment:
    def __init__(self, words):
        self.words = [FakeWord(*w) for w in words]


class FakeModel:
    """Returns a scripted word list per call; raises if a script item is an Exception."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        if isinstance(script, Exception):
            raise script
        return iter([FakeSegment(script)]), None


# ------------------------------------------------------------ normalizer --

@pytest.mark.parametrize("raw,expected", [
    (" Hola,", "hola"),
    ("¿Cómo?", "cómo"),          # accents survive, punctuation does not
    ("Nobi.", "nobi"),
    ("—", ""),
    ("31", "31"),
])
def test_norm_word(raw, expected):
    assert norm_word(raw) == expected


def test_join_words_keeps_leading_spaces_but_collapses_runs():
    words = [(0.0, 0.1, " Hola"), (0.1, 0.2, " mundo")]
    assert join_words(words) == "Hola mundo"


def test_base_prompt_falls_back_to_spanish():
    assert base_prompt_for("en") != base_prompt_for("es")
    assert base_prompt_for(None) == base_prompt_for("es")
    assert base_prompt_for("de") == base_prompt_for("es")   # unknown -> es


# ---------------------------------------------------- LocalAgreement-2 --

def test_flush_commits_only_the_agreeing_prefix():
    buf = HypothesisBuffer()
    first = [(0.0, 0.5, " hola"), (0.5, 1.0, " mundo"), (1.0, 1.5, " cruuu")]
    assert buf.flush(first) == []          # nothing to agree with yet
    second = [(0.0, 0.5, " hola"), (0.5, 1.0, " mundo"), (1.0, 1.6, " cruel")]
    committed = buf.flush(second)
    assert [w[2] for w in committed] == [" hola", " mundo"]
    assert [w[2] for w in buf.buffer] == [" cruel"]   # tail stays tentative


def test_flush_ignores_case_and_punctuation_when_agreeing():
    buf = HypothesisBuffer()
    buf.flush([(0.0, 0.5, " Hola,")])
    committed = buf.flush([(0.0, 0.5, " hola")])
    assert len(committed) == 1


def test_flush_stops_at_first_disagreement():
    buf = HypothesisBuffer()
    buf.flush([(0.0, 0.5, " a"), (0.5, 1.0, " b"), (1.0, 1.5, " c")])
    committed = buf.flush([(0.0, 0.5, " a"), (0.5, 1.0, " X"), (1.0, 1.5, " c")])
    assert [w[2] for w in committed] == [" a"]   # " c" must NOT be committed


def test_insert_shifts_to_absolute_time():
    buf = HypothesisBuffer()
    got = buf.insert([(1.0, 1.4, " nuevo")], offset=4.5)
    assert got[0][0] == pytest.approx(5.5)


def test_insert_drops_only_what_quedo_MUY_atras():
    """El filtro por tiempo es GRUESO a propósito.

    Antes cortaba pegado a la frontera (start > last_committed - 0.1) y eso
    perdía palabras: los timestamps de una misma palabra se corren hasta ~1s
    entre pasadas, porque cada pasada decodifica una ventana distinta, así que
    la palabra siguiente a la última comiteada caía del lado equivocado y
    quedaba descartada en todas las pasadas siguientes. Medido sobre el clip
    largo es-AR: 21 palabras perdidas, los huecos del texto.
    """
    buf = HypothesisBuffer()
    buf.last_committed_time = 10.0
    # frontera 10.0 → piso = 10.0 - MAX_TS_DRIFT_S = 8.5
    got = buf.insert([(0.0, 0.4, " remoto"), (9.0, 9.4, " cerca")], offset=0.0)
    assert [w[2] for w in got] == [" cerca"]   # 0.4 quedó atrás del piso; 9.4 no


def test_insert_conserva_la_palabra_que_cruza_la_frontera():
    """El caso exacto del bug: 'puedo' [8.88-9.84] con la frontera en 9.88."""
    buf = HypothesisBuffer()
    buf.last_committed_time = 9.88
    got = buf.insert([(8.88, 9.84, " puedo")], offset=0.0)
    assert [w[2] for w in got] == [" puedo"]


def test_insert_saca_por_TEXTO_lo_que_ya_se_comiteo():
    """Lo que evita el duplicado es el n-grama, no el reloj."""
    buf = HypothesisBuffer()
    buf.committed_tail = [(8.0, 8.4, " bajo"), (8.4, 8.9, " la"), (8.9, 9.4, " sombra")]
    buf.last_committed_time = 9.4
    got = buf.insert([(8.5, 9.0, " la"), (9.0, 9.5, " sombra"), (9.5, 9.9, " y")],
                     offset=0.0)
    assert [w[2] for w in got] == [" y"]


def test_flush_alimenta_la_cola_para_el_dedup():
    buf = HypothesisBuffer()
    buf.buffer = [(0.0, 0.5, " hola")]
    assert [w[2] for w in buf.flush([(0.0, 0.5, " hola")])] == [" hola"]
    assert [w[2] for w in buf.committed_tail] == [" hola"]


# ------------------------------------------------------------- OnlineASR --

def _silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_process_iter_commits_across_two_agreeing_passes():
    model = FakeModel([[(0.0, 0.5, " hola")], [(0.0, 0.5, " hola")]])
    asr = OnlineASR(model, "es")
    asr.insert_audio(_silence(1.0))
    assert asr.process_iter() == []
    assert asr.process_iter()          # second pass agrees -> commits
    assert asr.committed_text() == "hola"


def test_a_failing_pass_does_not_raise_and_is_counted():
    seen = []
    model = FakeModel([RuntimeError("boom")])
    asr = OnlineASR(model, "es", on_error=seen.append)
    asr.insert_audio(_silence(1.0))
    assert asr.process_iter() == []     # loop survives
    assert asr.failed_passes == 1
    assert isinstance(seen[0], RuntimeError)


def test_short_buffer_is_not_decoded_at_all():
    model = FakeModel([[(0.0, 0.1, " x")]])
    asr = OnlineASR(model, "es")
    asr.insert_audio(_silence(0.1))     # below MIN_DECODE_S
    asr.process_iter()
    assert model.calls == 0


def test_trim_drops_committed_audio_and_advances_the_offset():
    model = FakeModel([[(0.0, 1.0, " ya")]])
    asr = OnlineASR(model, "es", trim_buffer_s=5.0)
    asr.insert_audio(_silence(10.0))
    asr.committed = [(0.0, 4.0, " ya")]   # 4s already committed
    asr._maybe_trim()
    # keeps KEEP_CONTEXT_S(1s) of left context -> cuts 3s
    assert asr.time_offset == pytest.approx(3.0)
    assert asr.buffer_seconds == pytest.approx(7.0)


def test_trim_is_a_noop_without_committed_words():
    model = FakeModel([[]])
    asr = OnlineASR(model, "es", trim_buffer_s=5.0)
    asr.insert_audio(_silence(10.0))
    asr._maybe_trim()
    assert asr.time_offset == 0.0
    assert asr.buffer_seconds == pytest.approx(10.0)


def test_finish_accepts_the_tentative_tail():
    model = FakeModel([[(0.0, 0.5, " hola")], [(0.0, 0.5, " hola"), (0.5, 1.0, " che")]])
    asr = OnlineASR(model, "es")
    asr.insert_audio(_silence(1.0))
    asr.process_iter()
    assert asr.finish() == "hola che"     # " che" was still tentative
    assert asr.tentative_tail() == ""


def test_prompt_carries_the_committed_tail_forward():
    model = FakeModel([[]])
    asr = OnlineASR(model, "es")
    asr.committed = [(0.0, 1.0, " molde de aluminio")]
    assert asr._prompt().endswith("molde de aluminio")
    assert base_prompt_for("es") in asr._prompt()


# ------------------------------------------------------------------ WER --

def test_wer_is_zero_for_an_exact_match_modulo_case_and_punctuation():
    assert bench.word_error_rate("Hola, mundo.", "hola mundo") == 0.0


def test_wer_counts_substitution_insertion_and_deletion():
    assert bench.word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)
    assert bench.word_error_rate("a b c", "a b c d") == pytest.approx(1 / 3)
    assert bench.word_error_rate("a b c", "a c") == pytest.approx(1 / 3)


def test_wer_edge_cases():
    assert bench.word_error_rate("", "") == 0.0
    assert bench.word_error_rate("", "algo") == 1.0
    assert bench.word_error_rate("hola", "") == 1.0


def test_normalize_keeps_accents_and_enye():
    assert bench.normalize_text("¡Diseño ÁGIL!") == "diseño ágil"


# ------------------------------------------------- ventana del encoder --
# El recorte de ventana es un PARCHE sobre faster_whisper.transcribe.pad_or_trim
# (la biblioteca no expone otro punto). Lo que hay que garantizar es que sea
# inerte fuera del `with`, que no recorte audio nunca, y que un hilo no le
# cambie la ventana a otro — un parche global mal hecho acá se lleva puesto el
# pipeline batched, que apila features y necesita que todas midan lo mismo.

def _fw():
    fw = pytest.importorskip("faster_whisper.transcribe",
                             reason="el parche se verifica contra la biblioteca real")
    if not hasattr(fw, "pad_or_trim"):
        pytest.skip("faster_whisper stubeado en esta corrida")
    return fw


def _pad_or_trim():
    return _fw().pad_or_trim


def test_short_window_es_inerte_fuera_del_with():
    import streaming_core as sc
    assert sc._install_dynamic_pad()
    out = _pad_or_trim()(np.zeros((80, 400), dtype=np.float32))
    assert out.shape[-1] == sc.FULL_WINDOW_FRAMES


def test_short_window_padea_solo_lo_necesario():
    import streaming_core as sc
    with sc.short_window(pad_s=2.0, min_s=5.0) as active:
        assert active
        assert _pad_or_trim()(np.zeros((80, 800), np.float32)).shape[-1] == 1000
        # piso: un audio de 1s no baja de la ventana mínima
        assert _pad_or_trim()(np.zeros((80, 100), np.float32)).shape[-1] == 500
    assert _pad_or_trim()(np.zeros((80, 800), np.float32)).shape[-1] == sc.FULL_WINDOW_FRAMES


def test_short_window_nunca_recorta_mas_que_el_original():
    """Una ventana más corta que el audio perdería palabras: el tope sigue en 3000."""
    import streaming_core as sc
    with sc.short_window(pad_s=2.0, min_s=5.0):
        assert _pad_or_trim()(np.zeros((80, 2900), np.float32)).shape[-1] == 3000
        assert _pad_or_trim()(np.zeros((80, 3000), np.float32)).shape[-1] == 3000
        # y una petición explícita de otro largo (el batched pide el suyo) no se toca
        assert _pad_or_trim()(np.zeros((80, 400), np.float32), 700).shape[-1] == 700


def test_short_window_se_restaura_con_excepcion():
    import streaming_core as sc
    with pytest.raises(RuntimeError):
        with sc.short_window():
            raise RuntimeError("boom")
    assert _pad_or_trim()(np.zeros((80, 400), np.float32)).shape[-1] == sc.FULL_WINDOW_FRAMES


def test_short_window_no_se_filtra_a_otro_hilo():
    import threading
    import streaming_core as sc
    visto = []
    with sc.short_window(pad_s=2.0, min_s=5.0):
        t = threading.Thread(
            target=lambda: visto.append(
                _pad_or_trim()(np.zeros((80, 800), np.float32)).shape[-1]))
        t.start(); t.join()
    assert visto == [sc.FULL_WINDOW_FRAMES]


def test_one_shot_pasa_la_ventana_corta(monkeypatch):
    """transcribe_one_shot tiene que decodificar ADENTRO del with, no afuera."""
    import streaming_core as sc
    sc._install_dynamic_pad()
    visto = {}

    class M:
        def transcribe(self, audio, **kw):
            visto["len"] = _pad_or_trim()(np.zeros((80, 800), np.float32)).shape[-1]
            seg = FakeSegment([]); seg.text = "hola"
            return [seg], None

    assert sc.transcribe_one_shot(M(), np.zeros(16000, np.float32), "es") == "hola"
    assert visto["len"] == 1000
    visto.clear()
    sc.transcribe_one_shot(M(), np.zeros(16000, np.float32), "es", short_window_enabled=False)
    assert visto["len"] == sc.FULL_WINDOW_FRAMES


def test_streaming_no_recorta_la_ventana_por_default():
    """El recorte es 3x en el one-shot y ROMPE en el streaming (una pasada de 3s
    pasó a tardar 41s). La asimetría es deliberada; si un refactor la da vuelta,
    esto tiene que caer."""
    import streaming_core as sc
    sc._install_dynamic_pad()
    visto = []

    class M:
        def transcribe(self, audio, **kw):
            visto.append(_pad_or_trim()(np.zeros((80, 800), np.float32)).shape[-1])
            return [], None

    online = sc.OnlineASR(M(), "es")
    assert online.short_window_enabled is False
    online.insert_audio(np.zeros(SAMPLE_RATE, np.float32))
    online.process_iter()
    assert visto == [sc.FULL_WINDOW_FRAMES]
