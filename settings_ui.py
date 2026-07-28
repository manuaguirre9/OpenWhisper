from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QPlainTextEdit, QSpinBox, QFrame,
)
from PyQt6.QtCore import pyqtSignal
import sounddevice as sd

from config_manager import load_config, save_config
from system_info import physical_core_count, logical_core_count, resolve_cpu_threads
from theme import WINDOW_QSS, apply_windows_dark_titlebar


class SettingsWindow(QWidget):
    settings_saved = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración")
        self.setObjectName("root")
        self.setMinimumSize(500, 560)

        self.config = load_config()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Configuración")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Ajustá el micrófono, idioma, motor y vocabulario de Whisper Dictation.")
        subtitle.setObjectName("subtle")
        layout.addWidget(subtitle)

        # --- Micrófono ---
        self.mic_combo = QComboBox()
        self.populate_microphones()
        layout.addLayout(self._labeled_row("Micrófono", self.mic_combo))

        # --- Idioma ---
        self.lang_combo = QComboBox()
        self.languages = {"es": "Español", "en": "Inglés", "auto": "Autodetectar"}
        for code, name in self.languages.items():
            self.lang_combo.addItem(name, userData=code)
        current_lang = self.config.get("language", "es")
        idx = self.lang_combo.findData(current_lang)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        layout.addLayout(self._labeled_row("Idioma", self.lang_combo))

        # --- Modelo ---
        self.model_combo = QComboBox()
        self.models = ["tiny", "base", "small", "medium"]
        self.model_combo.addItems(self.models)
        current_model = self.config.get("model_size", "base")
        if current_model in self.models:
            self.model_combo.setCurrentText(current_model)
        layout.addLayout(self._labeled_row("Modelo de IA", self.model_combo))

        # --- Ducking ---
        self.duck_combo = QComboBox()
        for i in range(0, 101, 10):
            self.duck_combo.addItem(f"{i}%", userData=i)
        current_duck = self.config.get("ducking_percentage", 30)
        idx = self.duck_combo.findData(current_duck)
        if idx >= 0:
            self.duck_combo.setCurrentIndex(idx)
        layout.addLayout(self._labeled_row("Bajar volumen al grabar", self.duck_combo))

        # --- CPU threads ---
        # 0 = auto = physical cores. Positive values are capped at the number
        # of logical cores, so the spinbox itself can't be pushed past what the
        # CPU actually has (more threads than that only adds contention).
        self._phys_cores = physical_core_count()
        self._logi_cores = logical_core_count()
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, self._logi_cores)
        self.threads_spin.setValue(int(self.config.get("cpu_threads", 0)))
        threads_row = self._labeled_row("Hilos de CPU (0 = auto)", self.threads_spin)
        layout.addLayout(threads_row)

        # Live hint showing the concrete thread count that will actually be
        # used, so "0 = auto" isn't a black box.
        self.threads_hint = QLabel()
        self.threads_hint.setObjectName("subtle")
        self.threads_spin.valueChanged.connect(self._update_threads_hint)
        self._update_threads_hint()
        layout.addWidget(self.threads_hint)

        # --- Divider ---
        layout.addSpacing(4)
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: rgb(36, 36, 44);")
        layout.addWidget(divider)
        layout.addSpacing(4)

        # --- Vocabulario ---
        vocab_label = QLabel("Vocabulario personalizado")
        vocab_label.setObjectName("section")
        layout.addWidget(vocab_label)

        vocab_hint = QLabel(
            "Nombres propios, jerga, términos técnicos. Se inyectan al modelo "
            "como contexto para que los reconozca mejor."
        )
        vocab_hint.setObjectName("subtle")
        vocab_hint.setWordWrap(True)
        layout.addWidget(vocab_hint)

        self.vocab_edit = QPlainTextEdit()
        self.vocab_edit.setPlaceholderText(
            "Ej: OpenWhisper, Ryzen, faster-whisper, PyQt, Manuel Aguirre, Kubernetes, Postgres…"
        )
        self.vocab_edit.setPlainText(self.config.get("custom_vocabulary", ""))
        self.vocab_edit.setFixedHeight(100)
        layout.addWidget(self.vocab_edit)

        layout.addStretch()

        note = QLabel("Cambiar el modelo o los hilos recarga la IA.")
        note.setObjectName("subtle")
        layout.addWidget(note)

        # --- Save button ---
        button_row = QHBoxLayout()
        button_row.addStretch()
        self.save_btn = QPushButton("Guardar")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save_settings)
        button_row.addWidget(self.save_btn)
        layout.addLayout(button_row)

        self.setStyleSheet(WINDOW_QSS)

    def showEvent(self, event):
        super().showEvent(event)
        apply_windows_dark_titlebar(self)

    # ----- helpers -----

    def _update_threads_hint(self):
        """Reflect the effective thread count for the current spinbox value."""
        val = self.threads_spin.value()
        eff = resolve_cpu_threads(val)
        prefix = "Auto → " if val == 0 else ""
        self.threads_hint.setText(
            f"{prefix}usará {eff} hilos  "
            f"(núcleos físicos: {self._phys_cores}, lógicos: {self._logi_cores})"
        )

    def _labeled_row(self, label_text, widget):
        row = QHBoxLayout()
        row.setSpacing(12)
        label = QLabel(label_text)
        label.setMinimumWidth(220)
        row.addWidget(label)
        row.addWidget(widget, stretch=1)
        return row

    def populate_microphones(self):
        try:
            devices = sd.query_devices()
            self.mic_combo.addItem("Predeterminado del sistema", userData="default")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = f"{dev['name']} (API: {sd.query_hostapis(dev['hostapi'])['name']})"
                    self.mic_combo.addItem(name, userData=i)
            current_mic = self.config.get("microphone", "default")
            if current_mic == "default":
                self.mic_combo.setCurrentIndex(0)
            else:
                try:
                    idx = self.mic_combo.findData(int(current_mic))
                    if idx >= 0:
                        self.mic_combo.setCurrentIndex(idx)
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            print(f"Error enumerating audio devices: {e}")
            self.mic_combo.addItem("Error detectando micrófonos", userData="default")

    def save_settings(self):
        self.config["microphone"] = self.mic_combo.currentData()
        self.config["language"] = self.lang_combo.currentData()
        self.config["model_size"] = self.model_combo.currentText()
        self.config["ducking_percentage"] = self.duck_combo.currentData()
        self.config["cpu_threads"] = self.threads_spin.value()
        self.config["custom_vocabulary"] = self.vocab_edit.toPlainText().strip()

        save_config(self.config)
        self.settings_saved.emit(self.config)
        QMessageBox.information(self, "Guardado", "Configuración guardada exitosamente.")
        self.close()
