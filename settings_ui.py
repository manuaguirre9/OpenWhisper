from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QMessageBox, QPlainTextEdit, QSpinBox, QFrame, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import pyqtSignal, Qt
import sounddevice as sd

from config_manager import load_config, save_config
from system_info import physical_core_count, logical_core_count, resolve_cpu_threads
from theme import WINDOW_QSS, apply_windows_dark_titlebar
import corrections


class SettingsWindow(QWidget):
    settings_saved = pyqtSignal(dict)
    # Pedido de abrir la ventana de transcripción de archivos (app.py la conecta).
    open_batch = pyqtSignal()

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

        # --- Motor ---
        # Moonshine es streaming nativo: muestra texto mientras hablás y la
        # espera al terminar es ~0,1s. Whisper es el camino MIT, sin texto en
        # vivo (cada frase se decodifica cuando cierra). El híbrido usa los dos
        # a la vez: Moonshine llena el globo mientras hablás, Whisper decodifica
        # en paralelo y es SU texto el que se escribe. Cuesta los dos modelos en
        # memoria y CPU. Nemotron 3.5 hace las dos cosas con UN solo modelo:
        # muestra en vivo, puntúa, la espera al soltar es 0,00s y detecta el
        # idioma solo (castellano con términos en inglés, o inglés entero).
        # Es además el más barato de CPU: RTF 0,19 con 4 hilos.
        self.engine_combo = QComboBox()
        self.engines = {
            "nemotron": "Nemotron 3.5 (vivo + puntuación)",
            "moonshine": "Moonshine (texto en vivo)",
            "whisper": "Whisper (faster-whisper)",
            "hibrido": "Híbrido (Moonshine en vivo + Whisper escribe)",
        }
        for code, name in self.engines.items():
            self.engine_combo.addItem(name, userData=code)
        idx = self.engine_combo.findData(self.config.get("engine", "moonshine"))
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
        layout.addLayout(self._labeled_row("Motor", self.engine_combo))

        # --- Modo del hotkey ---
        # Escribir en la app destino MIENTRAS dictás solo es posible si no hay
        # modificadores apretados (Chromium descarta letras con Ctrl abajo), por
        # eso el modo "mantener" muestra el texto en el widget y pega al soltar.
        self.hotkey_mode_combo = QComboBox()
        self.hotkey_modes = {
            "hold": "Mantener apretado (pega al soltar)",
            "toggle": "Alternar: pulsar para empezar y terminar (escribe en vivo)",
        }
        for code, name in self.hotkey_modes.items():
            self.hotkey_mode_combo.addItem(name, userData=code)
        idx = self.hotkey_mode_combo.findData(self.config.get("hotkey_mode", "hold"))
        if idx >= 0:
            self.hotkey_mode_combo.setCurrentIndex(idx)
        layout.addLayout(self._labeled_row("Hotkey Ctrl+Win", self.hotkey_mode_combo))

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

        # --- Correcciones aprendidas ---
        # Se muestran para que nada quede aprendido a espaldas del usuario, y
        # porque un error consistente (el modelo escribe SIEMPRE "fronten")
        # queda guardado igual de bien que un acierto: hay que poder sacarlo.
        corr_label = QLabel("Correcciones aprendidas")
        corr_label.setObjectName("section")
        layout.addWidget(corr_label)

        corr_hint = QLabel(
            "Las palabras que corregiste en el globo al terminar una toma. "
            "Doble clic para olvidar una."
        )
        corr_hint.setObjectName("subtle")
        corr_hint.setWordWrap(True)
        layout.addWidget(corr_hint)

        self.corr_list = QListWidget()
        self.corr_list.setFixedHeight(96)
        self.corr_list.itemDoubleClicked.connect(self._forget_correction)
        layout.addWidget(self.corr_list)
        self._reload_corrections()

        layout.addStretch()

        note = QLabel("Cambiar el motor, el modelo, el idioma o los hilos recarga la IA.")
        note.setObjectName("subtle")
        layout.addWidget(note)

        # --- Save button ---
        button_row = QHBoxLayout()
        self.batch_btn = QPushButton("Transcribir archivo…")
        self.batch_btn.setToolTip("Transcribir un archivo de audio o video, con identificación de hablantes")
        self.batch_btn.clicked.connect(self.open_batch.emit)
        button_row.addWidget(self.batch_btn)
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

    def _reload_corrections(self):
        self.corr_list.clear()
        pairs = corrections.load()
        if not pairs:
            item = QListWidgetItem("Todavía ninguna. Corregí una palabra en el globo.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.corr_list.addItem(item)
            return
        # Al revés: lo último que corregiste es lo que querés ver primero.
        for wrong, right in reversed(list(pairs.items())):
            item = QListWidgetItem(f"{wrong}  →  {right}")
            item.setData(Qt.ItemDataRole.UserRole, wrong)
            self.corr_list.addItem(item)

    def _forget_correction(self, item):
        wrong = item.data(Qt.ItemDataRole.UserRole)
        if not wrong:
            return
        corrections.remove(wrong)
        self._reload_corrections()
        # El motor tiene que dejar de sesgar hacia ese término ya mismo.
        self.settings_saved.emit(self.config)

    def showEvent(self, event):
        # Pudo haber corregido palabras desde la última vez que abrió esto.
        self._reload_corrections()
        super().showEvent(event)

    def save_settings(self):
        self.config["microphone"] = self.mic_combo.currentData()
        self.config["language"] = self.lang_combo.currentData()
        self.config["model_size"] = self.model_combo.currentText()
        self.config["engine"] = self.engine_combo.currentData()
        self.config["hotkey_mode"] = self.hotkey_mode_combo.currentData()
        self.config["ducking_percentage"] = self.duck_combo.currentData()
        self.config["cpu_threads"] = self.threads_spin.value()
        self.config["custom_vocabulary"] = self.vocab_edit.toPlainText().strip()

        save_config(self.config)
        self.settings_saved.emit(self.config)
        QMessageBox.information(self, "Guardado", "Configuración guardada exitosamente.")
        self.close()
