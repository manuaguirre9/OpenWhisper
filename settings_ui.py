from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox
from PyQt6.QtCore import pyqtSignal
import sounddevice as sd
from config_manager import load_config, save_config

class SettingsWindow(QWidget):
    # Signal emitted when settings are saved to notify the orchestrator
    settings_saved = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuración de Whisper Dictation")
        self.setFixedSize(400, 250)
        
        self.config = load_config()
        
        # Main layout
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
            
        # Set current language
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
        
        # Note
        note_label = QLabel("<small><i>*Cambiar el modelo requiere reiniciar la carga de IA.</i></small>")
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
                # Only add devices with input channels
                if dev['max_input_channels'] > 0:
                    # Windows usually has weird hostapi names, but we keep it simple
                    name = f"{dev['name']} (API: {sd.query_hostapis(dev['hostapi'])['name']})"
                    self.mic_combo.addItem(name, userData=i)
                    
            # Set current selection
            current_mic = self.config.get("microphone", "default")
            if current_mic == "default":
                self.mic_combo.setCurrentIndex(0)
            else:
                index = self.mic_combo.findData(int(current_mic))
                if index >= 0:
                    self.mic_combo.setCurrentIndex(index)
        except Exception as e:
            print(f"Error enumerating audio devices: {e}")
            self.mic_combo.addItem("Error detectando micrófonos", userData="default")

    def save_settings(self):
        # Gather data
        mic_id = self.mic_combo.currentData()
        lang_code = self.lang_combo.currentData()
        model = self.model_combo.currentText()
        
        # Update config dictionary
        self.config["microphone"] = mic_id
        self.config["language"] = lang_code
        self.config["model_size"] = model
        
        # Save to file
        save_config(self.config)
        
        # Notify orchestrator
        self.settings_saved.emit(self.config)
        
        QMessageBox.information(self, "Guardado", "Configuración guardada exitosamente.")
        self.close()
