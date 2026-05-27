import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QMessageBox, QPlainTextEdit, QSpinBox
)
from PyQt6.QtCore import pyqtSignal
import sounddevice as sd
from config_manager import load_config, save_config


class SettingsWindow(QWidget):
    # Signal emitted when settings are saved to notify the orchestrator
    settings_saved = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración de Whisper Dictation")
        self.setMinimumSize(460, 480)

        self.config = load_config()

        layout = QVBoxLayout()

        # 1. Microphone Selection
        mic_layout = QHBoxLayout()
        mic_label = QLabel("Micrófono:")
        self.mic_combo = QComboBox()
        self.populate_microphones()
        mic_layout.addWidget(mic_label)
        mic_layout.addWidget(self.mic_combo)
        layout.addLayout(mic_layout)

        # 2. Language Selection
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Idioma:")
        self.lang_combo = QComboBox()
        self.languages = {"es": "Español", "en": "Inglés", "auto": "Autodetectar"}
        for code, name in self.languages.items():
            self.lang_combo.addItem(name, userData=code)

        current_lang = self.config.get("language", "es")
        index = self.lang_combo.findData(current_lang)
        if index >= 0:
            self.lang_combo.setCurrentIndex(index)

        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        layout.addLayout(lang_layout)

        # 3. Model Size Selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Modelo de IA:")
        self.model_combo = QComboBox()
        self.models = ["tiny", "base", "small", "medium"]
        self.model_combo.addItems(self.models)

        current_model = self.config.get("model_size", "base")
        if current_model in self.models:
            self.model_combo.setCurrentText(current_model)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # 4. Audio Ducking
        duck_layout = QHBoxLayout()
        duck_label = QLabel("Bajar volumen (Ducking):")
        self.duck_combo = QComboBox()
        for i in range(0, 101, 10):
            self.duck_combo.addItem(f"{i}%", userData=i)

        current_duck = self.config.get("ducking_percentage", 30)
        index = self.duck_combo.findData(current_duck)
        if index >= 0:
            self.duck_combo.setCurrentIndex(index)

        duck_layout.addWidget(duck_label)
        duck_layout.addWidget(self.duck_combo)
        layout.addLayout(duck_layout)

        # 5. CPU threads (0 = auto)
        threads_layout = QHBoxLayout()
        threads_label = QLabel("Hilos de CPU (0 = auto):")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setValue(int(self.config.get("cpu_threads", 0)))
        cpu_hint = QLabel(f"<small><i>detectados: {os.cpu_count() or '?'}</i></small>")
        threads_layout.addWidget(threads_label)
        threads_layout.addWidget(self.threads_spin)
        threads_layout.addWidget(cpu_hint)
        threads_layout.addStretch()
        layout.addLayout(threads_layout)

        # 6. Custom vocabulary
        vocab_label = QLabel("Vocabulario personalizado (nombres propios, jerga, términos técnicos):")
        layout.addWidget(vocab_label)
        self.vocab_edit = QPlainTextEdit()
        self.vocab_edit.setPlaceholderText(
            "Ej: OpenWhisper, Ryzen, faster-whisper, PyQt, Manuel Aguirre, Kubernetes, Postgres..."
        )
        self.vocab_edit.setPlainText(self.config.get("custom_vocabulary", ""))
        self.vocab_edit.setFixedHeight(90)
        layout.addWidget(self.vocab_edit)

        vocab_hint = QLabel(
            "<small><i>Estas palabras se inyectan al modelo como contexto para que las reconozca mejor. "
            "Separá por comas o espacios.</i></small>"
        )
        vocab_hint.setWordWrap(True)
        layout.addWidget(vocab_hint)

        # Note
        note_label = QLabel("<small><i>*Cambiar el modelo o los hilos requiere recargar la IA.</i></small>")
        layout.addWidget(note_label)

        # Save Button
        self.save_btn = QPushButton("Guardar Configuración")
        self.save_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def populate_microphones(self):
        """Fetches available input devices and populates the dropdown."""
        try:
            devices = sd.query_devices()
            self.mic_combo.addItem("Predeterminado del Sistema", userData="default")

            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = f"{dev['name']} (API: {sd.query_hostapis(dev['hostapi'])['name']})"
                    self.mic_combo.addItem(name, userData=i)

            current_mic = self.config.get("microphone", "default")
            if current_mic == "default":
                self.mic_combo.setCurrentIndex(0)
            else:
                try:
                    index = self.mic_combo.findData(int(current_mic))
                    if index >= 0:
                        self.mic_combo.setCurrentIndex(index)
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            print(f"Error enumerating audio devices: {e}")
            self.mic_combo.addItem("Error detectando micrófonos", userData="default")

    def save_settings(self):
        mic_id = self.mic_combo.currentData()
        lang_code = self.lang_combo.currentData()
        model = self.model_combo.currentText()
        duck = self.duck_combo.currentData()
        threads = self.threads_spin.value()
        vocab = self.vocab_edit.toPlainText().strip()

        self.config["microphone"] = mic_id
        self.config["language"] = lang_code
        self.config["model_size"] = model
        self.config["ducking_percentage"] = duck
        self.config["cpu_threads"] = threads
        self.config["custom_vocabulary"] = vocab

        save_config(self.config)
        self.settings_saved.emit(self.config)

        QMessageBox.information(self, "Guardado", "Configuración guardada exitosamente.")
        self.close()
