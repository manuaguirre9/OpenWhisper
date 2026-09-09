"""
Tests de la máquina de estados de la cola en BatchTranscriptionWindow.

Lo que fijan:
  - Cancelar frena la COLA, no sólo el archivo en curso. Antes
    _on_worker_finished encadenaba con _process_next() pasara lo que pasara,
    así que cancelar una cola de 3 cancelaba uno y arrancaba el siguiente.
  - Los archivos que quedaron sin procesar siguen en "En cola", para poder
    retomar con Transcribir.
  - shutdown() cancela y espera los QThread; Qt aborta el proceso si los
    destruye corriendo.

Corre sin modelo y sin pantalla (QT_QPA_PLATFORM=offscreen lo pone el propio
test). sherpa_onnx y faster_whisper se stubean: sólo se importan.
"""
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _install_stubs():
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = type("WhisperModel", (), {"__init__": lambda self, *a, **k: None})
    sys.modules.setdefault("faster_whisper", fw)

    sherpa = types.ModuleType("sherpa_onnx")
    for name in ("OfflineSpeakerDiarizationConfig",
                 "OfflineSpeakerSegmentationModelConfig",
                 "OfflineSpeakerSegmentationPyannoteModelConfig",
                 "SpeakerEmbeddingExtractorConfig", "FastClusteringConfig",
                 "OfflineSpeakerDiarization"):
        setattr(sherpa, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules.setdefault("sherpa_onnx", sherpa)


_install_stubs()

pytest.importorskip("PyQt6", reason="la ventana batch es sólo de escritorio")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import batch_window as bw  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    w = bw.BatchTranscriptionWindow(
        dictation_transcriber_provider=lambda: None,
        config_provider=lambda: {"language": "es", "cpu_threads": 0,
                                 "custom_vocabulary": ""},
    )
    yield w
    w.deleteLater()


class StubWorker:
    """Suficiente para las transiciones de estado: la cola sólo le pide cancel()."""

    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def wait(self, ms=0):
        return True


def _queue(window, n=3):
    window._on_files_added([f"/tmp/audio{i}.mp3" for i in range(n)])
    return window.entries


# ---------- cancelar frena la cola ----------

def test_cancel_stops_the_whole_queue(window):
    entries = _queue(window, 3)
    window.current_index = 0
    entries[0].status = bw.ST_RUNNING
    worker = StubWorker()
    window.worker = worker             # simula un worker vivo

    window._on_cancel_clicked()
    assert worker.cancelled is True
    assert window._queue_cancelled is True

    window._on_worker_finished()      # lo emite el QThread al terminar

    assert entries[0].status == bw.ST_CANCELLED
    assert entries[1].status == bw.ST_PENDING, "arrancó el siguiente archivo"
    assert entries[2].status == bw.ST_PENDING
    assert window.current_index is None
    assert window._active_transcriber is None


def test_pending_files_can_be_resumed_after_a_cancel(window):
    entries = _queue(window, 3)
    window.current_index = 0
    entries[0].status = bw.ST_RUNNING
    window.worker = StubWorker()
    window._on_cancel_clicked()
    window._on_worker_finished()

    # El flag no queda pegado: la próxima corrida tiene que poder encadenar.
    assert window._queue_cancelled is False
    assert any(e.status == bw.ST_PENDING for e in entries)
    window._refresh_buttons()
    assert window.transcribe_btn.isEnabled()


def test_normal_finish_still_chains_to_the_next_file(window):
    """El arreglo no debe romper el encadenado normal de la cola."""
    entries = _queue(window, 2)
    window.current_index = 0
    entries[0].status = bw.ST_DONE
    window.worker = None
    started = []
    window._start_worker = lambda entry: started.append(entry)
    window._active_transcriber = object()

    window._on_worker_finished()

    assert entries[1].status == bw.ST_RUNNING
    assert started and started[0] is entries[1]


# ---------- teardown ----------

def test_shutdown_cancels_and_waits_on_the_worker(window):
    calls = []

    class FakeWorker:
        def cancel(self):
            calls.append("cancel")

        def wait(self, ms):
            calls.append(("wait", ms))
            return True

    window.worker = FakeWorker()
    window.shutdown(timeout_ms=1234)

    assert calls == ["cancel", ("wait", 1234)]
    assert window._queue_cancelled is True


def test_shutdown_waits_on_loader_and_downloader_too(window):
    waited = []

    class FakeThread:
        def __init__(self, name):
            self.name = name

        def wait(self, ms):
            waited.append(self.name)
            return True

    window.worker = None
    window.loader = FakeThread("loader")
    window.diar_downloader = FakeThread("downloader")
    window.shutdown()

    assert waited == ["loader", "downloader"]


def test_shutdown_is_safe_with_nothing_running(window):
    window.worker = None
    window.loader = None
    window.diar_downloader = None
    window.shutdown()          # no debe tirar
