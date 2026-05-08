import sys
import threading
import time
import pyperclip
import pyautogui
from pynput import keyboard

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor

from audio_capture import AudioRecorder
from transcription_engine import Transcriber
from config_manager import load_config
from settings_ui import SettingsWindow

def create_tray_icon_pixmap():
    """Create a simple dynamic icon for the system tray if no .ico file exists."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setBrush(QColor("red"))
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()
    return QIcon(pixmap)

# --- UI Component ---
class FloatingWidget(QWidget):
    update_ui_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Removed WA_TransparentForMouseEvents to allow dragging

        self.config = load_config()
        # Restore saved position or default to 100, 100
        x = self.config.get("pos_x", 100)
        y = self.config.get("pos_y", 100)
        self.setGeometry(x, y, 100, 50)
        
        self.label = QLabel("🎙️ Loading AI...", self)
        # Reduced font size
        self.label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.update_style("loading")
        
        self.update_ui_signal.connect(self.handle_state_change)
        self.oldPos = None

    def update_style(self, state):
        # Reduced padding and border-radius to make it smaller
        if state == "recording":
            self.label.setText("🔴 Recording...")
            self.label.setStyleSheet("color: white; background-color: rgba(255, 0, 0, 180); padding: 5px 10px; border-radius: 8px;")
        elif state == "processing":
            self.label.setText("⏳ Processing...")
            self.label.setStyleSheet("color: white; background-color: rgba(0, 0, 255, 180); padding: 5px 10px; border-radius: 8px;")
        elif state == "ready":
            self.label.setText("🎙️ Ready")
            self.label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 5px 10px; border-radius: 8px;")
        else: # loading
            self.label.setStyleSheet("color: white; background-color: rgba(100, 100, 100, 150); padding: 5px 10px; border-radius: 8px;")
            
        self.label.adjustSize()
        self.resize(self.label.size())

    def handle_state_change(self, state):
        self.update_style(state)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.oldPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.oldPos is not None:
            delta = event.globalPosition().toPoint() - self.oldPos
            self.move(self.pos() + delta)
            self.oldPos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.oldPos = None
            # Save position to config
            from config_manager import save_config
            self.config["pos_x"] = self.pos().x()
            self.config["pos_y"] = self.pos().y()
            save_config(self.config)

# --- Background Orchestrator ---
class Orchestrator(QObject):
    def __init__(self, ui_widget):
        super().__init__()
        self.ui_widget = ui_widget
        self.config = load_config()
        
        self.ctrl_pressed = False
        self.cmd_pressed = False
        self.is_recording = False
        
        self.recorder = AudioRecorder()
        self.transcriber = None

    def load_model(self):
        model_size = self.config.get("model_size", "base")
        self.transcriber = Transcriber(model_size=model_size)
        self.ui_widget.update_ui_signal.emit("ready")

    def apply_new_config(self, new_config):
        print("[Orchestrator] Applying new config...")
        old_model = self.config.get("model_size")
        self.config = new_config
        
        if new_config.get("model_size") != old_model:
            self.ui_widget.update_ui_signal.emit("loading")
            # Reloading model takes time, should ideally be in thread but we do it simply here
            threading.Thread(target=self.load_model, daemon=True).start()

    def inject_text(self, text):
        if not text:
            return
        print(f"[Orchestrator] Injecting text: {text}")
        original_clipboard = pyperclip.paste()
        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.2)
        pyperclip.copy(original_clipboard)

    def check_state(self):
        if self.ctrl_pressed and self.cmd_pressed:
            if not self.is_recording and self.transcriber is not None:
                self.is_recording = True
                self.ui_widget.update_ui_signal.emit("recording")
                mic = self.config.get("microphone", "default")
                self.recorder.start_recording(device_id=mic)
        else:
            if self.is_recording:
                self.is_recording = False
                self.ui_widget.update_ui_signal.emit("processing")
                
                audio_data = self.recorder.stop_recording()
                
                lang = self.config.get("language")
                if lang == "auto":
                    lang = None
                    
                text = self.transcriber.transcribe(audio_data, language=lang)
                self.inject_text(text)
                
                self.ui_widget.update_ui_signal.emit("ready")

    def on_press(self, key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self.ctrl_pressed = True
        elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
            self.cmd_pressed = True
        self.check_state()

    def on_release(self, key):
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            self.ctrl_pressed = False
        elif key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
            self.cmd_pressed = False
        elif key == keyboard.Key.esc:
            print("Exiting application...")
            # For a proper tray app, esc shouldn't kill it. We remove this.
            pass
        self.check_state()

    def start_keep_alive(self):
        """Periodically hits the model with silence so Windows doesn't page it out of RAM."""
        def keep_alive_loop():
            while True:
                # Sleep for 3 minutes
                time.sleep(180)
                if self.transcriber is not None and not self.is_recording:
                    self.transcriber.warmup()
                    
        threading.Thread(target=keep_alive_loop, daemon=True).start()

    def run(self):
        self.load_model()
        self.start_keep_alive()
        print("[Orchestrator] Ready. Press Ctrl+Windows to record.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

if __name__ == '__main__':
    # Ensure PyQt doesn't quit if settings window closes
    QApplication.setQuitOnLastWindowClosed(False)
    app = QApplication(sys.argv)
    
    # UI
    ui = FloatingWidget()
    ui.show()
    
    # Orchestrator
    orchestrator = Orchestrator(ui)
    bg_thread = threading.Thread(target=orchestrator.run, daemon=True)
    bg_thread.start()
    
    # Settings Window
    settings_win = SettingsWindow()
    settings_win.settings_saved.connect(orchestrator.apply_new_config)
    
    # System Tray
    tray_icon = QSystemTrayIcon(create_tray_icon_pixmap(), app)
    tray_menu = QMenu()
    
    config_action = tray_menu.addAction("Configuración")
    config_action.triggered.connect(settings_win.show)
    
    quit_action = tray_menu.addAction("Salir")
    quit_action.triggered.connect(app.quit)
    
    tray_icon.setContextMenu(tray_menu)
    tray_icon.setToolTip("Whisper Dictation")
    tray_icon.show()
    
    sys.exit(app.exec())
