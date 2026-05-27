"""
Batch transcription window: drop one or many long audio files in, get clean
transcripts out. Transcripts stream into the UI as segments arrive instead
of waiting for the file to finish.

Visual style is aligned with the floating overlay — same dark matte
palette, coral/lavender accents, Segoe UI typography, rounded surfaces.
"""

import os
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QPlainTextEdit, QFrame, QMessageBox,
    QSizePolicy, QListWidget, QListWidgetItem, QComboBox, QCheckBox,
    QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QFont, QDragEnterEvent, QDropEvent, QTextCursor,
    QBrush,
)

from theme import WINDOW_QSS, apply_windows_dark_titlebar
from transcription_engine import Transcriber
from diarization_engine import (
    Diarizer, ensure_models, models_ready, assign_speakers_to_segments,
)


# Palette aligned with the floating overlay.
WF_SURFACE_ELEVATED = QColor(22, 22, 28)
WF_SURFACE_HOVER = QColor(32, 32, 40)
WF_ON_SURFACE = QColor(232, 232, 240)
WF_ON_SURFACE_MUTED = QColor(170, 170, 182)
WF_ON_SURFACE_DIM = QColor(120, 120, 130)
WF_ACCENT_RECORDING = QColor(255, 112, 122)   # coral
WF_ACCENT_PROCESSING = QColor(180, 160, 255)  # lavender
WF_ACCENT_DONE = QColor(140, 220, 170)        # soft green
WF_ACCENT_ERROR = QColor(255, 120, 130)       # coral red

SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".oga", ".flac",
                  ".webm", ".aac", ".opus"}

# Whisper model sizes available in the batch window picker.
# tiny/base/small/medium are the safe defaults; large-v3 is the best quality
# but is heavy (~3GB RAM) and slow on CPU — leave it in for users who want it.
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

# Status values for a queued file.
ST_PENDING = "pending"
ST_RUNNING = "running"
ST_DONE = "done"
ST_ERROR = "error"
ST_CANCELLED = "cancelled"


# ---------- data ----------

@dataclass
class FileEntry:
    path: str
    status: str = ST_PENDING
    progress: float = 0.0   # 0..1 — overall (Whisper + optional diarization)
    segments: List[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    error: str = ""
    # Speaker diarization (only populated when the file was processed with
    # diarize=True). Keys: speaker_id (int) → display name (str). Defaults
    # to "Speaker 1", "Speaker 2"… and the user can rename them.
    speaker_names: Dict[int, str] = field(default_factory=dict)
    diarized: bool = False
    diar_phase: str = ""   # "" / "transcribing" / "diarizing"


# ---------- post-processing helpers ----------

PARAGRAPH_GAP = 1.2  # seconds of silence between segments → new paragraph


def group_into_paragraphs(segments, gap_threshold=PARAGRAPH_GAP):
    if not segments:
        return []
    paragraphs = []
    current = [segments[0]["text"].strip()]
    prev_end = segments[0]["end"]
    for seg in segments[1:]:
        gap = seg["start"] - prev_end
        text = seg["text"].strip()
        if gap >= gap_threshold and current:
            paragraphs.append(" ".join(current).strip())
            current = [text]
        else:
            current.append(text)
        prev_end = seg["end"]
    if current:
        paragraphs.append(" ".join(current).strip())

    cleaned = []
    for p in paragraphs:
        if not p:
            continue
        if p[0].islower():
            p = p[0].upper() + p[1:]
        cleaned.append(p)
    return cleaned


def format_plain_text(segments):
    return "\n\n".join(group_into_paragraphs(segments))


def format_markdown(segments, source_name=None, info=None):
    lines = []
    if source_name:
        lines.append(f"# {source_name}")
        lines.append("")
    if info and info.get("duration"):
        mins = int(info["duration"] // 60)
        secs = int(info["duration"] % 60)
        lang = info.get("language", "?")
        lines.append(f"_Duración: {mins} min {secs:02d} s · idioma: {lang}_")
        lines.append("")
    for p in group_into_paragraphs(segments):
        lines.append(p)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _srt_timestamp(seconds):
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt(segments, speaker_names=None):
    """SubRip subtitles. If segments include speaker labels and speaker_names
    is provided, each line is prefixed with the speaker name."""
    out = []
    for i, seg in enumerate(segments, start=1):
        out.append(str(i))
        out.append(f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}")
        text = seg["text"].strip()
        if speaker_names is not None and "speaker" in seg:
            sp = seg["speaker"]
            name = speaker_names.get(sp, f"Speaker {sp + 1}")
            text = f"{name}: {text}"
        out.append(text)
        out.append("")
    return "\n".join(out)


def format_markdown_with_speakers(segments, speaker_names, source_name=None, info=None):
    """Markdown export with speaker blocks (uses format_with_speakers body)."""
    lines = []
    if source_name:
        lines.append(f"# {source_name}")
        lines.append("")
    if info and info.get("duration"):
        mins = int(info["duration"] // 60)
        secs = int(info["duration"] % 60)
        lang = info.get("language", "?")
        lines.append(f"_Duración: {mins} min {secs:02d} s · idioma: {lang}_")
        lines.append("")
    lines.append(format_with_speakers(segments, speaker_names))
    return "\n".join(lines).rstrip() + "\n"


# ---------- worker thread ----------

class TranscriptionWorker(QThread):
    # Whisper streaming phase signals.
    progress = pyqtSignal(float)              # 0..1 (relative to current phase)
    segment_ready = pyqtSignal(dict, float)   # segment dict, fraction
    # Diarization phase signals.
    phase_changed = pyqtSignal(str)           # "transcribing" / "diarizing"
    diar_progress = pyqtSignal(float)         # 0..1 within diarization
    # Terminal signals.
    finished_ok = pyqtSignal(list, dict)      # segments (with speaker key if diarized), info
    failed = pyqtSignal(str)

    def __init__(self, transcriber, audio_path, language=None,
                 diarize=False, num_speakers=None, diarizer=None, parent=None):
        super().__init__(parent)
        self.transcriber = transcriber
        self.audio_path = audio_path
        self.language = language
        self.diarize = diarize
        self.num_speakers = num_speakers
        self.diarizer = diarizer  # Diarizer instance (pre-built and cached)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            # --- Phase 1: Whisper transcription with streaming ---
            self.phase_changed.emit("transcribing")

            def on_segment(seg, frac):
                self.segment_ready.emit(seg, frac)
                self.progress.emit(frac)

            segments, info = self.transcriber.transcribe_file(
                self.audio_path,
                language=self.language,
                segment_cb=on_segment,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                return

            # --- Phase 2: speaker diarization (optional) ---
            if self.diarize and self.diarizer is not None:
                self.phase_changed.emit("diarizing")
                diar_segments = self.diarizer.diarize(
                    self.audio_path,
                    num_speakers=self.num_speakers,
                    progress_cb=lambda f: self.diar_progress.emit(f),
                    cancel_check=lambda: self._cancelled,
                )
                if self._cancelled:
                    return
                segments = assign_speakers_to_segments(segments, diar_segments)

            self.finished_ok.emit(segments, info)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---------- speaker-aware formatting ----------

def _capitalize_first(text: str) -> str:
    text = text.lstrip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def format_with_speakers(segments, speaker_names):
    """
    Render diarized segments as blocks grouped by consecutive speaker turns.
    Each block: `**SpeakerName:**\\n<joined text>` with blank line separators.
    """
    if not segments:
        return ""
    blocks = []
    current_speaker = segments[0].get("speaker", 0)
    current_text_parts = [_capitalize_first(segments[0]["text"])]

    for seg in segments[1:]:
        sp = seg.get("speaker", 0)
        if sp != current_speaker:
            blocks.append((current_speaker, " ".join(current_text_parts).strip()))
            current_speaker = sp
            current_text_parts = [_capitalize_first(seg["text"])]
        else:
            current_text_parts.append(seg["text"].strip())

    blocks.append((current_speaker, " ".join(current_text_parts).strip()))

    def _name(sp):
        return speaker_names.get(sp, f"Speaker {sp + 1}")

    out_lines = []
    for sp, text in blocks:
        out_lines.append(f"**{_name(sp)}:**")
        out_lines.append(text)
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"


# ---------- model loader threads ----------

class DiarModelDownloadWorker(QThread):
    """Downloads the sherpa-onnx diarization models in the background."""
    status = pyqtSignal(str, float)   # label, fraction
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        try:
            ensure_models(status_cb=lambda label, frac: self.status.emit(label, frac))
            self.finished_ok.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")




class ModelLoaderWorker(QThread):
    """
    Builds a fresh Transcriber off the UI thread. Used when the batch
    window needs a model size that the dictation orchestrator isn't
    currently holding, so loading doesn't freeze the window.
    """
    loaded = pyqtSignal(object)   # the Transcriber instance
    failed = pyqtSignal(str)
    # (downloaded_mb, total_mb) reported during first-time download.
    download_progress = pyqtSignal(float, float)

    def __init__(self, model_size, cpu_threads, vocabulary, parent=None):
        super().__init__(parent)
        self.model_size = model_size
        self.cpu_threads = cpu_threads
        self.vocabulary = vocabulary

    def run(self):
        try:
            t = Transcriber(
                model_size=self.model_size,
                cpu_threads=self.cpu_threads,
                vocabulary=self.vocabulary,
                progress_cb=lambda done, total: self.download_progress.emit(done, total),
            )
            self.loaded.emit(t)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---------- drop zone ----------

class DropZone(QFrame):
    files_selected = pyqtSignal(list)   # list of absolute paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._hover = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_picker()

    def _open_picker(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Elegir archivos de audio",
            "",
            "Audio (*.mp3 *.wav *.m4a *.mp4 *.ogg *.flac *.webm *.aac *.opus)",
        )
        if paths:
            self.files_selected.emit(paths)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._has_supported_url(event.mimeData()):
            self._hover = True
            self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._hover = False
        self.update()

    def dropEvent(self, event: QDropEvent):
        self._hover = False
        self.update()
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                p = url.toLocalFile()
                if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                    paths.append(p)
        if paths:
            self.files_selected.emit(paths)

    @staticmethod
    def _has_supported_url(mime):
        if not mime.hasUrls():
            return False
        for url in mime.urls():
            if url.isLocalFile():
                if os.path.splitext(url.toLocalFile())[1].lower() in SUPPORTED_EXTS:
                    return True
        return False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        fill = WF_SURFACE_HOVER if self._hover else WF_SURFACE_ELEVATED
        painter.setBrush(fill)
        border_color = WF_ACCENT_PROCESSING if self._hover else WF_ON_SURFACE_DIM
        pen = QPen(border_color, 1.4)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern([6, 5])
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 16, 16)

        cx = rect.center().x()
        cy = rect.center().y() - 14
        self._draw_mic(painter, cx, cy, size=22, color=WF_ON_SURFACE_MUTED)

        painter.setPen(WF_ON_SURFACE)
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        l1 = QRectF(rect.left(), cy + 16, rect.width(), 20)
        painter.drawText(l1, int(Qt.AlignmentFlag.AlignCenter),
                         "Arrastrá uno o varios audios")

        painter.setPen(WF_ON_SURFACE_MUTED)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
        l2 = QRectF(rect.left(), cy + 36, rect.width(), 18)
        painter.drawText(l2, int(Qt.AlignmentFlag.AlignCenter),
                         "o hacé clic para examinar")
        painter.end()

    def _draw_mic(self, painter, cx, cy, size, color):
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        body_w = size * 0.48
        body_h = size * 0.58
        body = QRectF(cx - body_w / 2, cy - size * 0.45, body_w, body_h)
        painter.drawRoundedRect(body, body_w / 2, body_w / 2)
        arc = QRectF(cx - size * 0.38, cy - size * 0.08, size * 0.76, size * 0.52)
        painter.drawArc(arc, 200 * 16, 140 * 16)
        painter.drawLine(QPointF(cx, cy + size * 0.38),
                         QPointF(cx, cy + size * 0.50))


# ---------- main window ----------

class BatchTranscriptionWindow(QMainWindow):
    def __init__(self, dictation_transcriber_provider, config_provider, parent=None):
        super().__init__(parent)
        # Provider that returns the Transcriber used for live dictation.
        # We reuse it whenever the user picks the same model in this window;
        # otherwise we load a dedicated batch transcriber.
        self.dictation_transcriber_provider = dictation_transcriber_provider
        self.config_provider = config_provider

        self.entries: List[FileEntry] = []
        self.current_index: Optional[int] = None    # being processed
        self.selected_index: Optional[int] = None   # shown in the text area
        self.worker: Optional[TranscriptionWorker] = None

        # Batch-specific Transcriber cache. Persists across queue runs so the
        # user doesn't re-pay the model load cost when picking the same size.
        self.batch_transcriber: Optional[Transcriber] = None
        self.loader: Optional[ModelLoaderWorker] = None
        # Transcriber the queue is currently using (set when transcription
        # starts; cleared when the queue drains).
        self._active_transcriber: Optional[Transcriber] = None

        # Speaker diarization state (lazy-initialized — models download +
        # pipeline construction only happen the first time the user clicks
        # Transcribir with the checkbox enabled).
        self.diarizer: Optional[Diarizer] = None
        self.diar_downloader: Optional[DiarModelDownloadWorker] = None
        # The speaker rename widgets currently mounted in the panel; kept
        # so we can read their values when the user types.
        self._speaker_inputs: Dict[int, QLineEdit] = {}

        self.setWindowTitle("Transcribir archivos")
        self.resize(680, 790)
        self._build_ui()
        self.setStyleSheet(WINDOW_QSS)

    def showEvent(self, event):
        super().showEvent(event)
        apply_windows_dark_titlebar(self)

    # ----- UI construction -----

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        title = QLabel("Transcribir archivos")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Pegá audios largos y obtené el texto formateado, con párrafos y puntuación.")
        subtitle.setObjectName("subtle")
        layout.addWidget(subtitle)

        # Model selector
        model_row = QHBoxLayout()
        model_row.setSpacing(10)
        model_label = QLabel("Modelo de IA")
        model_label.setMinimumWidth(100)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_SIZES)
        # Default to whatever dictation is using, falling back to config, then "base".
        default_model = "base"
        dict_trans = self.dictation_transcriber_provider()
        if dict_trans is not None:
            default_model = dict_trans.model_size
        else:
            default_model = (self.config_provider() or {}).get("model_size", "base")
        if default_model in MODEL_SIZES:
            self.model_combo.setCurrentText(default_model)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("subtle")
        model_row.addWidget(model_label)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(self.model_status_label, stretch=1)
        layout.addLayout(model_row)
        self._refresh_model_status()

        # Diarization controls
        diar_row = QHBoxLayout()
        diar_row.setSpacing(10)
        self.diar_checkbox = QCheckBox("Identificar hablantes")
        self.diar_checkbox.setChecked(False)
        self.diar_checkbox.toggled.connect(self._on_diar_toggled)
        diar_row.addWidget(self.diar_checkbox)

        diar_row.addSpacing(20)
        self.speakers_label = QLabel("Cantidad:")
        self.speakers_label.setEnabled(False)
        diar_row.addWidget(self.speakers_label)

        self.speakers_combo = QComboBox()
        self.speakers_combo.addItem("Auto", userData=None)
        for n in (2, 3, 4, 5):
            self.speakers_combo.addItem(f"{n} hablantes", userData=n)
        self.speakers_combo.setEnabled(False)
        diar_row.addWidget(self.speakers_combo)

        self.diar_status_label = QLabel("")
        self.diar_status_label.setObjectName("subtle")
        diar_row.addWidget(self.diar_status_label, stretch=1)
        layout.addLayout(diar_row)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_selected.connect(self._on_files_added)
        layout.addWidget(self.drop_zone)

        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(110)
        self.queue_list.setMaximumHeight(150)
        self.queue_list.currentRowChanged.connect(self._on_row_selected)
        layout.addWidget(self.queue_list)

        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        # Action buttons
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.transcribe_btn = QPushButton("Transcribir")
        self.transcribe_btn.setObjectName("primary")
        self.transcribe_btn.setEnabled(False)
        self.transcribe_btn.clicked.connect(self._on_transcribe_clicked)
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        self.remove_btn = QPushButton("Quitar")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._on_remove_clicked)
        self.clear_btn = QPushButton("Limpiar cola")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        actions.addWidget(self.transcribe_btn)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.remove_btn)
        actions.addWidget(self.clear_btn)
        actions.addStretch()
        layout.addLayout(actions)

        # Output area
        self.output_edit = QPlainTextEdit()
        self.output_edit.setPlaceholderText("La transcripción aparecerá acá mientras se procesa…")
        layout.addWidget(self.output_edit, stretch=1)

        # Speaker rename panel (only visible when the selected file has been
        # diarized). One QLineEdit per detected speaker; editing re-renders
        # the transcript with the new label.
        self.speakers_panel = QWidget()
        self.speakers_panel_layout = QHBoxLayout(self.speakers_panel)
        self.speakers_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.speakers_panel_layout.setSpacing(8)
        self.speakers_panel.setVisible(False)
        layout.addWidget(self.speakers_panel)

        # Export row
        export = QHBoxLayout()
        export.setSpacing(10)
        self.export_txt_btn = QPushButton("Exportar .txt")
        self.export_srt_btn = QPushButton("Exportar .srt")
        self.export_md_btn = QPushButton("Exportar .md")
        for btn, fmt in (
            (self.export_txt_btn, "txt"),
            (self.export_srt_btn, "srt"),
            (self.export_md_btn, "md"),
        ):
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked=False, f=fmt: self._on_export(f))
            export.addWidget(btn)
        export.addStretch()
        layout.addLayout(export)

    # ----- queue helpers -----

    def _on_files_added(self, paths):
        existing = {e.path for e in self.entries}
        added_any = False
        for p in paths:
            if p in existing:
                continue
            entry = FileEntry(path=p)
            self.entries.append(entry)
            self._append_list_item(entry)
            added_any = True
        if added_any:
            self._refresh_buttons()
            if self.selected_index is None and self.entries:
                self.queue_list.setCurrentRow(0)

    def _append_list_item(self, entry):
        item = QListWidgetItem(self._row_text(entry))
        item.setForeground(QBrush(self._row_color(entry)))
        self.queue_list.addItem(item)

    def _row_text(self, entry):
        name = os.path.basename(entry.path)
        if entry.status == ST_PENDING:
            return f"○   {name}   ·   En cola"
        if entry.status == ST_RUNNING:
            pct = int(entry.progress * 100)
            if entry.diar_phase == "diarizing":
                return f"●   {name}   ·   Identificando hablantes {pct}%"
            return f"●   {name}   ·   Transcribiendo {pct}%"
        if entry.status == ST_DONE:
            suffix = " · con hablantes" if entry.diarized else ""
            return f"✓   {name}   ·   Completo{suffix}"
        if entry.status == ST_CANCELLED:
            return f"○   {name}   ·   Cancelado"
        if entry.status == ST_ERROR:
            return f"✕   {name}   ·   Error"
        return name

    def _row_color(self, entry):
        if entry.status == ST_RUNNING:
            return WF_ACCENT_PROCESSING
        if entry.status == ST_DONE:
            return WF_ACCENT_DONE
        if entry.status == ST_ERROR:
            return WF_ACCENT_ERROR
        if entry.status == ST_CANCELLED:
            return WF_ON_SURFACE_DIM
        return WF_ON_SURFACE

    def _refresh_row(self, idx):
        if not (0 <= idx < self.queue_list.count()):
            return
        entry = self.entries[idx]
        item = self.queue_list.item(idx)
        item.setText(self._row_text(entry))
        item.setForeground(QBrush(self._row_color(entry)))

    def _refresh_buttons(self):
        has_entries = bool(self.entries)
        has_pending = any(e.status == ST_PENDING for e in self.entries)
        running = self.worker is not None
        loading = self.loader is not None
        downloading_diar = self.diar_downloader is not None
        busy = running or loading or downloading_diar
        self.transcribe_btn.setEnabled(has_pending and not busy)
        if downloading_diar:
            self.transcribe_btn.setText("Descargando hablantes…")
        elif loading:
            self.transcribe_btn.setText("Cargando modelo…")
        else:
            self.transcribe_btn.setText("Transcribir")
        self.cancel_btn.setEnabled(running)
        self.remove_btn.setEnabled(
            self.selected_index is not None
            and not (running and self.selected_index == self.current_index)
        )
        self.clear_btn.setEnabled(has_entries and not busy)
        # Lock model + diar selectors while anything is in flight.
        self.model_combo.setEnabled(not busy)
        self.diar_checkbox.setEnabled(not busy)
        active_diar = self.diar_checkbox.isChecked() and not busy
        self.speakers_label.setEnabled(active_diar)
        self.speakers_combo.setEnabled(active_diar)
        sel = self._selected_entry()
        has_segments = sel is not None and bool(sel.segments)
        for b in (self.export_txt_btn, self.export_srt_btn, self.export_md_btn):
            b.setEnabled(has_segments)

    def _selected_entry(self):
        if self.selected_index is None:
            return None
        if not (0 <= self.selected_index < len(self.entries)):
            return None
        return self.entries[self.selected_index]

    # ----- selection / display -----

    def _on_row_selected(self, row):
        if row < 0 or row >= len(self.entries):
            self.selected_index = None
            self.output_edit.clear()
            self.speakers_panel.setVisible(False)
            self._refresh_buttons()
            return
        self.selected_index = row
        entry = self.entries[row]
        # Render the file's current text. If it's already diarized, format
        # with speaker blocks and show the rename panel. Otherwise just
        # plain paragraphs (works mid-stream too).
        if entry.diarized:
            self.output_edit.setPlainText(
                format_with_speakers(entry.segments, entry.speaker_names)
            )
            self._rebuild_speakers_panel(entry)
        else:
            self.output_edit.setPlainText(format_plain_text(entry.segments))
            self.speakers_panel.setVisible(False)
        # If this is the file being transcribed, position cursor at the end
        # so future streamed segments append visibly.
        cursor = self.output_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_edit.setTextCursor(cursor)
        # Progress reflects the currently selected file's progress.
        self.progress.setValue(int(entry.progress * 1000))
        self._refresh_buttons()

    def _on_remove_clicked(self):
        if self.selected_index is None:
            return
        if self.worker is not None and self.selected_index == self.current_index:
            return  # can't remove the file currently being transcribed
        idx = self.selected_index
        del self.entries[idx]
        self.queue_list.takeItem(idx)
        # Adjust indexes
        if self.current_index is not None and self.current_index > idx:
            self.current_index -= 1
        self.selected_index = self.queue_list.currentRow()
        if self.selected_index >= 0:
            self._on_row_selected(self.selected_index)
        else:
            self.output_edit.clear()
            self.progress.setValue(0)
        self._refresh_buttons()

    def _on_clear_clicked(self):
        if self.worker is not None:
            return
        self.entries.clear()
        self.queue_list.clear()
        self.current_index = None
        self.selected_index = None
        self.output_edit.clear()
        self.progress.setValue(0)
        self._refresh_buttons()

    # ----- diarization helpers -----

    def _on_diar_toggled(self, checked: bool):
        self.speakers_label.setEnabled(checked)
        self.speakers_combo.setEnabled(checked)
        self._refresh_diar_status()
        self._refresh_buttons()

    def _refresh_diar_status(self):
        if not self.diar_checkbox.isChecked():
            self.diar_status_label.setText("")
            return
        if self.diar_downloader is not None:
            return  # download status already shown
        if self.diarizer is not None or models_ready():
            self.diar_status_label.setText("✓ Modelos listos")
            self.diar_status_label.setStyleSheet("color: rgb(140, 220, 170);")
        else:
            self.diar_status_label.setText("Se descargarán al transcribir (~35MB)")
            self.diar_status_label.setStyleSheet("color: rgb(170, 170, 182);")

    def _ensure_diarizer_ready(self, on_ready):
        """Make sure the Diarizer is instantiated. Downloads models first if
        necessary. Calls on_ready() once the diarizer is usable, or surfaces
        an error via QMessageBox."""
        if self.diarizer is not None:
            on_ready()
            return
        if models_ready():
            try:
                self.diarizer = Diarizer()
                on_ready()
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"No se pudo iniciar el motor de hablantes:\n\n{e}"
                )
            return
        # Models missing — kick off background download.
        self.diar_status_label.setStyleSheet("color: rgb(180, 160, 255);")
        self.diar_downloader = DiarModelDownloadWorker()
        self.diar_downloader.status.connect(self._on_diar_download_progress)
        self.diar_downloader.finished_ok.connect(lambda: self._on_diar_download_done(on_ready))
        self.diar_downloader.failed.connect(self._on_diar_download_failed)
        self.diar_downloader.finished.connect(self._on_diar_downloader_finished)
        self.diar_downloader.start()
        self._refresh_buttons()

    def _on_diar_download_progress(self, label, frac):
        pct = int(frac * 100)
        self.diar_status_label.setText(f"{label}… {pct}%")

    def _on_diar_download_done(self, on_ready):
        try:
            self.diarizer = Diarizer()
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Modelos descargados pero no se pudo iniciar el motor:\n\n{e}"
            )
            return
        self._refresh_diar_status()
        on_ready()

    def _on_diar_download_failed(self, message):
        self.diar_status_label.setText("")
        QMessageBox.critical(
            self, "Error",
            f"No se pudieron descargar los modelos de hablantes:\n\n{message}"
        )

    def _on_diar_downloader_finished(self):
        self.diar_downloader = None
        self._refresh_buttons()

    # ----- speaker rename panel -----

    def _rebuild_speakers_panel(self, entry: FileEntry):
        """Show one labeled QLineEdit per detected speaker for `entry`."""
        # Clear previous widgets
        while self.speakers_panel_layout.count():
            item = self.speakers_panel_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._speaker_inputs.clear()

        if not entry.diarized or not entry.segments:
            self.speakers_panel.setVisible(False)
            return

        speaker_ids = sorted({s.get("speaker", 0) for s in entry.segments
                              if "speaker" in s})
        if not speaker_ids:
            self.speakers_panel.setVisible(False)
            return

        for sp in speaker_ids:
            row = QVBoxLayout()
            row.setSpacing(4)
            label = QLabel(f"Speaker {sp + 1}")
            label.setObjectName("subtle")
            row.addWidget(label)
            edit = QLineEdit()
            edit.setPlaceholderText(f"Speaker {sp + 1}")
            edit.setText(entry.speaker_names.get(sp, ""))
            edit.textChanged.connect(lambda txt, s=sp: self._on_speaker_renamed(s, txt))
            row.addWidget(edit)
            container = QWidget()
            container.setLayout(row)
            self.speakers_panel_layout.addWidget(container)
            self._speaker_inputs[sp] = edit

        self.speakers_panel_layout.addStretch()
        self.speakers_panel.setVisible(True)

    def _on_speaker_renamed(self, speaker_id: int, new_name: str):
        entry = self._selected_entry()
        if entry is None or not entry.diarized:
            return
        name = new_name.strip()
        if name:
            entry.speaker_names[speaker_id] = name
        else:
            entry.speaker_names.pop(speaker_id, None)
        # Re-render the transcript with the new label.
        self.output_edit.setPlainText(format_with_speakers(entry.segments, entry.speaker_names))

    # ----- transcription pipeline -----

    def _on_transcribe_clicked(self):
        if self.worker is not None or self.loader is not None or self.diar_downloader is not None:
            return
        selected = self.model_combo.currentText()

        def proceed_with_model():
            transcriber = self._resolve_transcriber(selected)
            if transcriber is not None:
                self._begin_processing_with(transcriber)
            else:
                self._load_model_then_process(selected)

        # If diarization is enabled, make sure the diarizer is ready before
        # we start the Whisper job — we need it the moment Whisper finishes.
        if self.diar_checkbox.isChecked():
            self._ensure_diarizer_ready(proceed_with_model)
        else:
            proceed_with_model()

    def _resolve_transcriber(self, model_size):
        """Return a ready Transcriber for `model_size`, or None if we need
        to load one. Prefers the dictation instance to save RAM."""
        dict_trans = self.dictation_transcriber_provider()
        if dict_trans is not None and dict_trans.model_size == model_size:
            return dict_trans
        if self.batch_transcriber is not None and self.batch_transcriber.model_size == model_size:
            return self.batch_transcriber
        return None

    def _load_model_then_process(self, model_size):
        config = self.config_provider() or {}
        self.model_status_label.setText(f"Cargando modelo «{model_size}»…")
        self.model_status_label.setStyleSheet("color: rgb(180, 160, 255);")
        self._refresh_buttons()
        self.loader = ModelLoaderWorker(
            model_size=model_size,
            cpu_threads=config.get("cpu_threads", 0),
            vocabulary=config.get("custom_vocabulary", ""),
        )
        self.loader.loaded.connect(self._on_model_loaded)
        self.loader.failed.connect(self._on_model_load_failed)
        self.loader.download_progress.connect(
            lambda done, total: self._on_model_download_progress(model_size, done, total)
        )
        self.loader.finished.connect(self._on_loader_finished)
        self.loader.start()

    def _on_model_download_progress(self, model_size, done_mb, total_mb):
        if total_mb <= 0:
            return
        pct = int(min(100, max(0, (done_mb / total_mb) * 100)))
        self.model_status_label.setText(
            f"Descargando «{model_size}»  {done_mb:.0f} / {total_mb:.0f} MB · {pct}%"
        )

    def _on_model_loaded(self, transcriber):
        # Cache it so future jobs at this size are instant.
        self.batch_transcriber = transcriber
        self._refresh_model_status()
        self._begin_processing_with(transcriber)

    def _on_model_load_failed(self, message):
        self.model_status_label.setText("")
        QMessageBox.critical(self, "Error", f"No se pudo cargar el modelo:\n\n{message}")
        self._refresh_buttons()

    def _on_loader_finished(self):
        self.loader = None
        self._refresh_buttons()

    def _begin_processing_with(self, transcriber):
        # Keep vocabulary in sync with the latest config in case the user
        # changed it in Settings while the batch transcriber was cached.
        config = self.config_provider() or {}
        transcriber.set_vocabulary(config.get("custom_vocabulary", ""))
        self._active_transcriber = transcriber
        self._refresh_model_status()
        self._process_next()

    def _process_next(self):
        for i, entry in enumerate(self.entries):
            if entry.status == ST_PENDING:
                self.current_index = i
                entry.status = ST_RUNNING
                entry.progress = 0.0
                entry.segments = []
                entry.info = {}
                entry.error = ""
                self._refresh_row(i)
                # Auto-select the file being processed so streaming is visible.
                self.queue_list.setCurrentRow(i)
                self._start_worker(entry)
                self._refresh_buttons()
                return
        # Nothing pending → queue done.
        self.current_index = None
        self.worker = None
        self._active_transcriber = None
        self.progress.setValue(0)
        self._refresh_buttons()

    def _start_worker(self, entry):
        transcriber = self._active_transcriber
        if transcriber is None:
            return
        config = self.config_provider() or {}
        lang = config.get("language") or None
        if lang == "auto":
            lang = None

        diarize = self.diar_checkbox.isChecked() and self.diarizer is not None
        num_speakers = self.speakers_combo.currentData() if diarize else None

        self.progress.setValue(0)
        self.output_edit.clear()

        self.worker = TranscriptionWorker(
            transcriber,
            entry.path,
            language=lang,
            diarize=diarize,
            num_speakers=num_speakers,
            diarizer=self.diarizer if diarize else None,
        )
        self.worker.phase_changed.connect(self._on_phase_changed)
        self.worker.progress.connect(self._on_progress)
        self.worker.segment_ready.connect(self._on_segment_ready)
        self.worker.diar_progress.connect(self._on_diar_progress)
        self.worker.finished_ok.connect(self._on_file_finished)
        self.worker.failed.connect(self._on_file_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_model_changed(self, _new_text):
        self._refresh_model_status()
        self._refresh_buttons()

    def _refresh_model_status(self):
        selected = self.model_combo.currentText()
        ready = self._resolve_transcriber(selected) is not None
        if ready:
            self.model_status_label.setText("✓ Cargado")
            self.model_status_label.setStyleSheet("color: rgb(140, 220, 170);")
        else:
            self.model_status_label.setText("Se cargará al transcribir")
            self.model_status_label.setStyleSheet("color: rgb(170, 170, 182);")

    def _on_phase_changed(self, phase: str):
        if self.current_index is None:
            return
        entry = self.entries[self.current_index]
        entry.diar_phase = phase
        self._refresh_row(self.current_index)

    def _on_progress(self, fraction):
        if self.current_index is None:
            return
        entry = self.entries[self.current_index]
        entry.progress = fraction
        # Only update the progress bar if the user is looking at the
        # running file. Otherwise progress reflects whatever is selected.
        if self.selected_index == self.current_index:
            self.progress.setValue(int(fraction * 1000))
        self._refresh_row(self.current_index)

    def _on_diar_progress(self, fraction: float):
        if self.current_index is None:
            return
        if self.selected_index == self.current_index:
            self.progress.setValue(int(fraction * 1000))
        # Show "Identificando hablantes…" in the queue row.
        self._refresh_row(self.current_index)

    def _on_segment_ready(self, segment, fraction):
        """Stream the new segment's text into the file's transcript."""
        if self.current_index is None:
            return
        entry = self.entries[self.current_index]
        entry.segments.append(segment)
        # If the user is viewing this file, append text live.
        if self.selected_index == self.current_index:
            self._append_segment_to_view(entry, segment)

    def _append_segment_to_view(self, entry, segment):
        """Insert the new segment's text at the end of the output area,
        adding a paragraph break if the silence gap warrants it."""
        cursor = self.output_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        seg_count = len(entry.segments)
        text = segment["text"]

        def _capitalize_first(s):
            s = s.lstrip()
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            return s

        if seg_count == 1:
            cursor.insertText(_capitalize_first(text))
        else:
            prev = entry.segments[-2]
            gap = segment["start"] - prev["end"]
            if gap >= PARAGRAPH_GAP:
                cursor.insertText("\n\n" + _capitalize_first(text))
            else:
                cursor.insertText(text)  # keep leading space from Whisper

        # Auto-scroll to the bottom so the latest text is always visible.
        self.output_edit.setTextCursor(cursor)
        bar = self.output_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_file_finished(self, segments, info):
        if self.current_index is None:
            return
        entry = self.entries[self.current_index]
        entry.segments = segments
        entry.info = info
        entry.status = ST_DONE
        entry.progress = 1.0
        entry.diar_phase = ""
        # Detect diarization: any segment with a "speaker" key.
        entry.diarized = any("speaker" in s for s in segments)
        self._refresh_row(self.current_index)
        # Re-render fully-formatted text if user is on this file.
        if self.selected_index == self.current_index:
            if entry.diarized:
                self.output_edit.setPlainText(
                    format_with_speakers(entry.segments, entry.speaker_names)
                )
                self._rebuild_speakers_panel(entry)
            else:
                self.output_edit.setPlainText(format_plain_text(segments))
                self.speakers_panel.setVisible(False)
            self.progress.setValue(1000)
        self._refresh_buttons()

    def _on_file_failed(self, message):
        if self.current_index is None:
            return
        entry = self.entries[self.current_index]
        entry.status = ST_ERROR
        entry.error = message
        entry.progress = 0.0
        self._refresh_row(self.current_index)
        if self.selected_index == self.current_index:
            self.output_edit.setPlainText(f"[Error transcribiendo este archivo]\n\n{message}")
        # Surface the error once but keep the queue moving.
        print(f"[BatchWindow] Transcription failed for {entry.path}: {message}")

    def _on_worker_finished(self):
        """Called when the QThread emits its finished signal, regardless
        of success/failure/cancel. We chain into the next pending file."""
        self.worker = None
        # If the current file is still RUNNING here it means cancel was hit
        # (no finished_ok or failed emitted).
        if self.current_index is not None:
            entry = self.entries[self.current_index]
            if entry.status == ST_RUNNING:
                entry.status = ST_CANCELLED
                self._refresh_row(self.current_index)
        # Move on to the next pending file.
        self._process_next()

    def _on_cancel_clicked(self):
        if self.worker is not None:
            self.worker.cancel()
        self.cancel_btn.setEnabled(False)

    # ----- export -----

    def _on_export(self, fmt):
        entry = self._selected_entry()
        if entry is None or not entry.segments:
            return
        default_name = os.path.splitext(os.path.basename(entry.path))[0]
        filters = {
            "txt": ("Texto plano (*.txt)", ".txt"),
            "srt": ("Subtítulos SubRip (*.srt)", ".srt"),
            "md":  ("Markdown (*.md)",          ".md"),
        }
        label, ext = filters[fmt]
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar transcripción", default_name + ext, label
        )
        if not path:
            return

        if fmt == "txt":
            # If the file is currently shown, respect manual edits in the textarea.
            if self.selected_index is not None and self.entries[self.selected_index] is entry:
                content = self.output_edit.toPlainText().strip() + "\n"
            elif entry.diarized:
                content = format_with_speakers(entry.segments, entry.speaker_names)
            else:
                content = format_plain_text(entry.segments) + "\n"
        elif fmt == "srt":
            if entry.diarized:
                content = format_srt(entry.segments, entry.speaker_names)
            else:
                content = format_srt(entry.segments)
        else:
            if entry.diarized:
                content = format_markdown_with_speakers(
                    entry.segments, entry.speaker_names,
                    source_name=os.path.basename(entry.path),
                    info=entry.info,
                )
            else:
                content = format_markdown(
                    entry.segments,
                    source_name=os.path.basename(entry.path),
                    info=entry.info,
                )

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo:\n{e}")
