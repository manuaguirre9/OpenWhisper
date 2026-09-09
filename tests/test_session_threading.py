"""
Tests for the push-to-talk driver's threading, in streaming_prototype.Session.

The bug these lock down: _loop used to drain audio_q *and* run the Whisper
pass on the same thread, so for the whole duration of a pass nobody was
pulling from the queue. The queue is unbounded, so no audio was lost — but
the pass cadence went irregular, and on a Pi (where a pass can exceed
MIN_CHUNK_S) the drift compounds instead of self-correcting.

No mic and no model: sounddevice/pynput/faster_whisper are stubbed, and the
pass itself is either a no-op decode or blocked on an Event.
"""
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _install_platform_stubs():
    """streaming_prototype imports these at module level; none exist on CI."""
    sd = types.ModuleType("sounddevice")

    class InputStream:
        def __init__(self, **kw):
            self.kw = kw

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    sd.InputStream = InputStream
    sys.modules.setdefault("sounddevice", sd)

    keyboard = types.ModuleType("pynput.keyboard")

    class Key:
        f8 = "f8"
        esc = "esc"

    keyboard.Key = Key
    keyboard.Listener = object
    pynput = types.ModuleType("pynput")
    pynput.keyboard = keyboard
    sys.modules.setdefault("pynput", pynput)
    sys.modules.setdefault("pynput.keyboard", keyboard)

    fw = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, *a, **k):
            pass

    fw.WhisperModel = WhisperModel
    sys.modules.setdefault("faster_whisper", fw)


_install_platform_stubs()

import streaming_prototype as sp  # noqa: E402
from streaming_core import SAMPLE_RATE  # noqa: E402


class SilentModel:
    """Decodes to nothing, instantly. Enough for the real _run_pass to run."""

    def transcribe(self, audio, **kw):
        info = types.SimpleNamespace(language="es", language_probability=1.0,
                                     duration=0.0)
        return iter(()), info


def _chunk(seconds=0.5):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def _new_session(trim=8.0):
    return sp.Session(SilentModel(), "es", None, trim_buffer_s=trim)


# ---------- CLI ----------

def test_parse_args_defaults_match_the_documented_behaviour():
    args = sp.parse_args([])
    assert args.model == "small"
    assert args.threads is None          # → physical_core_count()
    assert args.trim == sp.TRIM_BUFFER_S


def test_parse_args_keeps_the_old_positional_invocations():
    # `python streaming_prototype.py small 3` was documented before argparse.
    args = sp.parse_args(["small", "3"])
    assert (args.model, args.threads) == ("small", 3)


def test_trim_is_overridable_for_the_pi_run():
    args = sp.parse_args(["base", "--trim", "8"])
    assert args.model == "base"
    assert args.trim == pytest.approx(8.0)


def test_trim_reaches_the_online_asr():
    session = _new_session(trim=8.0)
    session._finalize = lambda: None
    session.start()
    try:
        assert session.online.trim_buffer_s == pytest.approx(8.0)
    finally:
        session.stop()


# ---------- the actual threading fix ----------

def test_queue_keeps_draining_while_a_pass_is_running():
    """The regression. On the old single-threaded loop this leaves audio
    sitting in audio_q for the whole duration of the pass."""
    session = _new_session()
    session._finalize = lambda: None

    pass_started = threading.Event()
    let_pass_finish = threading.Event()

    def blocking_pass():
        pass_started.set()
        let_pass_finish.wait(5.0)

    session._run_pass = blocking_pass
    session.start()
    try:
        for _ in range(3):                       # 1.5s ≥ MIN_CHUNK_S → fires a pass
            session.audio_q.put(_chunk())
        assert pass_started.wait(5.0), "la pasada nunca arrancó"

        for _ in range(5):                       # audio nuevo, con la pasada bloqueada
            session.audio_q.put(_chunk())

        deadline = time.time() + 3.0
        while session.audio_q.qsize() and time.time() < deadline:
            time.sleep(0.01)

        assert session.audio_q.qsize() == 0, (
            "audio_q no se drenó mientras corría la pasada: el productor "
            "quedó bloqueado detrás del consumidor"
        )
    finally:
        let_pass_finish.set()
        session.stop()


def test_no_audio_is_lost_in_the_handoff():
    """Everything put on the queue must reach OnlineASR before _finalize."""
    session = _new_session()
    finalized = threading.Event()
    session._finalize = finalized.set

    session.start()
    n_chunks, seconds = 6, 0.5
    for _ in range(n_chunks):
        session.audio_q.put(_chunk(seconds))
    time.sleep(0.15)
    session.stop()

    assert finalized.wait(5.0), "el hilo de pasadas no terminó"
    expected = int(SAMPLE_RATE * seconds) * n_chunks
    assert len(session.online.audio) == expected
    assert len(np.concatenate(session.full_chunks)) == expected


def test_start_is_ignored_while_a_take_is_still_finalizing():
    """A second F8 must not swap self.online out from under the thread that
    is still printing the previous report."""
    session = _new_session()
    release_finalize = threading.Event()
    in_finalize = threading.Event()

    def slow_finalize():
        in_finalize.set()
        release_finalize.wait(5.0)

    session._finalize = slow_finalize
    session.start()
    first_online = session.online
    session.audio_q.put(_chunk())
    session.stop()

    assert in_finalize.wait(5.0)
    session.start()                     # debería ser un no-op: busy sigue True
    try:
        assert session.online is first_online
        assert session.recording is False
    finally:
        release_finalize.set()
