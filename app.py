import sys
import threading
import time
import pyperclip
import pyautogui
from pynput import keyboard

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor

from audio_capture import AudioRecorder
from transcription_engine import Transcriber
from config_manager import load_config
from settings_ui import SettingsWindow
from audio_ducking import AudioDucker
from floating_widget import FloatingWidget

def create_tray_icon_pixmap():
    """Create a simple dynamic icon for the system tray if no .ico file exists."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setBrush(QColor("red"))
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()
    return QIcon(pixmap)

# --- Background Orchestrator ---
class Orchestrator(QObject):
    def __init__(self, ui_widget):
        super().__init__()
        self.ui_widget = ui_widget
        self.config = load_config()
        
        self.is_recording = False
        
        self.recorder = AudioRecorder()
        self.transcriber = None
        self.audio_ducker = AudioDucker()

    def load_model(self):
        model_size = self.config.get("model_size", "base")
        cpu_threads = self.config.get("cpu_threads", 0)
        vocabulary = self.config.get("custom_vocabulary", "")
        self.transcriber = Transcriber(
            model_size=model_size,
            cpu_threads=cpu_threads,
            vocabulary=vocabulary,
        )
        self.ui_widget.update_ui_signal.emit("ready")

    def apply_new_config(self, new_config):
        print("[Orchestrator] Applying new config...")
        old_model = self.config.get("model_size")
        old_threads = self.config.get("cpu_threads", 0)
        self.config = new_config

        needs_reload = (
            new_config.get("model_size") != old_model or
            new_config.get("cpu_threads", 0) != old_threads
        )

        if needs_reload:
            self.ui_widget.update_ui_signal.emit("loading")
            threading.Thread(target=self.load_model, daemon=True).start()
        elif self.transcriber is not None:
            # Vocabulary can be updated live without reloading the model
            self.transcriber.set_vocabulary(new_config.get("custom_vocabulary", ""))

    def inject_text(self, text):
        if not text:
            return
        print(f"[Orchestrator] Injecting text: {text}")

        # paste() can raise if the clipboard holds a non-text payload
        # (e.g. an image copied from a browser). In that case we just skip
        # restoring it instead of crashing the whole injection.
        original_clipboard = None
        try:
            original_clipboard = pyperclip.paste()
        except Exception as e:
            print(f"[Orchestrator] Could not read original clipboard: {e}")

        try:
            pyperclip.copy(text)
            time.sleep(0.1)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.2)
        finally:
            if original_clipboard is not None:
                try:
                    pyperclip.copy(original_clipboard)
                except Exception:
                    pass

    def is_exact_hotkey_pressed(self):
        import ctypes
        VK_CONTROL = 0x11
        VK_LWIN = 0x5B
        VK_RWIN = 0x5C
        VK_MENU = 0x12  # Alt
        VK_SHIFT = 0x10 # Shift
        
        is_ctrl = bool(ctypes.windll.user32.GetAsyncKeyState(VK_CONTROL) & 0x8000)
        is_win = bool((ctypes.windll.user32.GetAsyncKeyState(VK_LWIN) & 0x8000) or 
                      (ctypes.windll.user32.GetAsyncKeyState(VK_RWIN) & 0x8000))
        
        if not (is_ctrl and is_win):
            return False
            
        is_alt = bool(ctypes.windll.user32.GetAsyncKeyState(VK_MENU) & 0x8000)
        is_shift = bool(ctypes.windll.user32.GetAsyncKeyState(VK_SHIFT) & 0x8000)
        
        if is_alt or is_shift:
            return False
            
        # Check A-Z (0x41-0x5A), 0-9 (0x30-0x39) to avoid collisions with other shortcuts
        keys_to_check = list(range(0x41, 0x5B)) + list(range(0x30, 0x3A))
        for i in keys_to_check:
            if ctypes.windll.user32.GetAsyncKeyState(i) & 0x8000:
                return False
                
        return True

    def check_state(self):
        if self.is_exact_hotkey_pressed():
            if not self.is_recording and self.transcriber is not None:
                self.is_recording = True
                self.ui_widget.update_ui_signal.emit("recording")
                
                # Duck volume
                duck_perc = self.config.get("ducking_percentage", 30)
                if duck_perc > 0:
                    self.audio_ducker.duck(duck_perc)
                    
                mic = self.config.get("microphone", "default")
                self.recorder.start_recording(device_id=mic)
        else:
            if self.is_recording:
                self.is_recording = False
                self.ui_widget.update_ui_signal.emit("processing")
                
                # Restore volume
                if self.config.get("ducking_percentage", 30) > 0:
                    self.audio_ducker.restore()
                
                audio_data = self.recorder.stop_recording()
                
                lang = self.config.get("language")
                if lang == "auto":
                    lang = None
                    
                # Run transcription in a background thread to prevent blocking the listener
                threading.Thread(target=self._process_audio_and_inject, args=(audio_data, lang), daemon=True).start()

    def _process_audio_and_inject(self, audio_data, lang):
        try:
            text = self.transcriber.transcribe(audio_data, language=lang)
            self.inject_text(text)
        except Exception as e:
            print(f"[Orchestrator] Error during transcription/injection: {e}")
        finally:
            self.ui_widget.update_ui_signal.emit("ready")

    def on_press(self, key):
        try:
            self.check_state()
        except Exception as e:
            print(f"Error in on_press: {e}")

    def on_release(self, key):
        try:
            self.check_state()
        except Exception as e:
            print(f"Error in on_release: {e}")

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
    # Wire the recorder into the widget so it can poll live RMS levels
    # for the waveform visualization while recording.
    ui.set_recorder(orchestrator.recorder)
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
