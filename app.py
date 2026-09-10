import queue
import sys
import threading
import time
from pynput import keyboard

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor

from audio_capture import AudioRecorder
from segment_asr import SegmentASR
from transcription_engine import Transcriber
from config_manager import load_config
from settings_ui import SettingsWindow
from audio_ducking import AudioDucker
from floating_widget import FloatingWidget
from dictation_bubble import DictationBubble
from batch_window import BatchTranscriptionWindow
from text_injector import paste_text, type_text, wait_modifiers_released

def create_tray_icon_pixmap():
    """Create a simple dynamic icon for the system tray if no .ico file exists."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setBrush(QColor("red"))
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()
    return QIcon(pixmap)


class _SegmentLive:
    """Adapta SegmentASR (whisper) a la misma interfaz que MoonshineSession.

    Todo lo que SegmentASR comitea sale por on_commit → on_line, incluida la
    cola que decodifica finish(). Por eso pending_text() es siempre vacío: no
    queda nada que on_line no haya entregado.
    """

    def __init__(self, segmenter: SegmentASR):
        self.seg = segmenter

    def insert_audio(self, chunk):
        self.seg.insert_audio(chunk)

    def poll(self):
        self.seg.poll()

    def finish(self) -> str:
        return self.seg.finish()

    def pending_text(self) -> str:
        return ""


# --- Background Orchestrator ---
class Orchestrator(QObject):
    def __init__(self, ui_widget):
        super().__init__()
        self.ui_widget = ui_widget
        self.config = load_config()

        self.is_recording = False
        # Mientras la toma anterior se está cerrando (decodificando la cola y
        # escribiendo), no se puede arrancar otra: compartirían la sesión en
        # vivo. La pulsación se ignora, igual que en streaming_prototype.
        self._finalizing = False
        # Para el modo "toggle": el hotkey dispara en el FLANCO de apretar, y
        # los autorepeat del teclado mientras se mantiene no cuentan.
        self._hotkey_down = False

        self.recorder = AudioRecorder()
        # Motores. `transcriber` (whisper) es el que comparte la ventana de
        # batch; si el motor activo es Moonshine queda en None y la ventana de
        # batch carga el suyo cuando lo necesita.
        self.transcriber = None
        self.moonshine = None
        self.engine_name = None
        self.audio_ducker = AudioDucker()

        # Sesión en vivo de la toma actual (SegmentASR o MoonshineSession), el
        # hilo que la alimenta, y cuántas muestras ya le dio. Ver _start_live.
        self._live = None
        self._live_fed = 0
        self._live_stop = None
        self._live_thread = None
        self._live_lines = []     # frases ya cerradas en esta toma (para el widget)
        self._live_partial = ""
        self._live_typing = False  # ¿esta toma escribe en vivo en el destino?

        # Todo lo que se escribe en la app destino pasa por UNA cola y UN hilo:
        # así las frases en vivo y el texto final salen en orden, y la espera a
        # que el usuario suelte los modificadores no frena el audio.
        self._inject_q = queue.Queue()

    # ------------------------------------------------------------ motores --

    def _on_download_progress(self, done_mb, total_mb):
        if total_mb <= 0:
            return
        pct = int(min(100, max(0, (done_mb / total_mb) * 100)))
        self.ui_widget.update_loading_progress_signal.emit(pct)

    def load_model(self):
        engine = self.config.get("engine", "moonshine")
        if engine == "moonshine":
            try:
                self._load_moonshine()
            except Exception as e:  # noqa: BLE001 - sin Moonshine, Whisper sigue sirviendo
                print(f"[Orchestrator] Moonshine no cargó ({e}); uso Whisper.")
                engine = "whisper"
        if engine == "whisper":
            self._load_whisper()
        self.ui_widget.update_ui_signal.emit("ready")

    def _load_moonshine(self):
        from moonshine_engine import MoonshineEngine

        lang = self.config.get("language")
        self.moonshine = MoonshineEngine(
            language=lang,
            model_size=self.config.get("model_size", "small"),
            vocabulary=self.config.get("custom_vocabulary", ""),
            progress_cb=self._on_download_progress,
        )
        self.transcriber = None
        self.engine_name = "moonshine"

    def _load_whisper(self):
        self.transcriber = Transcriber(
            model_size=self.config.get("model_size", "small"),
            cpu_threads=self.config.get("cpu_threads", 0),
            vocabulary=self.config.get("custom_vocabulary", ""),
            beam_size=self.config.get("beam_size", 1),
            progress_cb=self._on_download_progress,
        )
        self.moonshine = None
        self.engine_name = "whisper"

    def _engine_ready(self) -> bool:
        if self._finalizing:
            return False
        if self.engine_name == "moonshine":
            return self.moonshine is not None
        return self.transcriber is not None

    def apply_new_config(self, new_config):
        print("[Orchestrator] Applying new config...")
        old = self.config
        self.config = new_config

        needs_reload = (
            new_config.get("engine", "moonshine") != old.get("engine", "moonshine") or
            new_config.get("model_size") != old.get("model_size") or
            new_config.get("cpu_threads", 0) != old.get("cpu_threads", 0) or
            (self.engine_name == "moonshine" and
             new_config.get("language") != old.get("language"))
        )

        if needs_reload:
            self.ui_widget.update_ui_signal.emit("loading")
            threading.Thread(target=self.load_model, daemon=True).start()
            return
        # Vocabulary and beam_size can be updated live without reloading.
        vocab = new_config.get("custom_vocabulary", "")
        if self.transcriber is not None:
            self.transcriber.set_vocabulary(vocab)
            self.transcriber.beam_size = new_config.get("beam_size", 1)
        if self.moonshine is not None:
            self.moonshine.set_vocabulary(vocab)

    # ------------------------------------------------------------- hotkey --

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
        pressed = self.is_exact_hotkey_pressed()
        edge = pressed and not self._hotkey_down
        self._hotkey_down = pressed

        if self.config.get("hotkey_mode", "hold") == "toggle":
            # Una pulsación empieza, la siguiente termina. Entre las dos el
            # usuario no tiene nada apretado, y por eso acá sí se puede
            # escribir en el destino mientras habla.
            if edge:
                if self.is_recording:
                    self._stop_take()
                elif self._engine_ready():
                    self._start_take()
            return

        if pressed:
            if not self.is_recording and self._engine_ready():
                self._start_take()
        elif self.is_recording:
            self._stop_take()

    def _start_take(self):
        self.is_recording = True
        self._live_typing = self.config.get("hotkey_mode", "hold") == "toggle"
        self._live_lines = []
        self._live_partial = ""
        self.ui_widget.update_text_signal.emit("")
        self.ui_widget.update_ui_signal.emit("recording")

        # Duck volume
        duck_perc = self.config.get("ducking_percentage", 30)
        if duck_perc > 0:
            self.audio_ducker.duck(duck_perc)

        mic = self.config.get("microphone", "default")
        self.recorder.start_recording(device_id=mic)
        self._start_live()

    def _stop_take(self):
        self.is_recording = False
        self._finalizing = True
        self.ui_widget.update_ui_signal.emit("processing")

        # Restore volume
        if self.config.get("ducking_percentage", 30) > 0:
            self.audio_ducker.restore()

        audio_data = self.recorder.stop_recording()

        lang = self.config.get("language")
        if lang == "auto":
            lang = None

        # Run transcription in a background thread to prevent blocking the listener
        threading.Thread(target=self._finish_take, args=(audio_data, lang), daemon=True).start()

    # --------------------------------------------------------------- toma --

    def _language(self):
        lang = self.config.get("language")
        return None if lang == "auto" else lang

    def _start_live(self):
        """Arranca el consumo en vivo del audio de esta toma.

        Moonshine: el stream decodifica de forma incremental y emite parciales
        (→ widget) y frases cerradas (→ on_line). Whisper en modo segmento:
        cada frase que cierra por VAD se decodifica mientras seguís hablando y
        sale por on_commit (→ on_line). Whisper one-shot: nada en vivo.
        """
        self._live = None
        if self.engine_name == "moonshine" and self.moonshine is not None:
            self._live = self.moonshine.start_session(
                on_partial=self._on_partial,
                on_line=self._on_line,
                on_error=lambda e: print(f"[Orchestrator] Moonshine: {e}"),
            )
            period = 0.1
        elif (self.engine_name == "whisper" and self.transcriber is not None
              and getattr(self.transcriber, "model", None) is not None
              and self.config.get("dictation_mode", "segment") == "segment"):
            lang = self._language()
            try:
                prompt = self.transcriber._build_prompt(lang)
            except Exception:
                prompt = None
            self._live = _SegmentLive(SegmentASR(
                self.transcriber.model, lang,
                beam_size=self.transcriber.beam_size,
                base_prompt=prompt,
                on_error=lambda e: print(f"[Orchestrator] Segmento falló: {e}"),
                on_commit=self._on_line,
            ))
            period = 0.25
        else:
            return

        live = self._live
        self._live_fed = 0
        self._live_stop = threading.Event()

        def loop():
            while not self._live_stop.is_set():
                try:
                    chunk = self.recorder.drain_new()
                    if len(chunk):
                        live.insert_audio(chunk)
                        self._live_fed += len(chunk)
                    if hasattr(live, "poll"):
                        live.poll()
                except Exception as e:  # noqa: BLE001 - nunca matar el hilo
                    print(f"[Orchestrator] Sesión en vivo: {e}")
                self._live_stop.wait(period)

        self._live_thread = threading.Thread(target=loop, daemon=True)
        self._live_thread.start()

    def _on_partial(self, text):
        """Parcial de la frase en curso. Solo se MUESTRA: Moonshine lo reescribe."""
        self._live_partial = text
        self._push_live_text()

    def _on_line(self, text):
        """Una frase cerró. Es definitiva: va al widget y, en modo toggle, al destino."""
        self._live_lines.append(text)
        self._live_partial = ""
        self._push_live_text()
        if self._live_typing:
            self._inject_q.put(("type", text + " "))

    def _push_live_text(self):
        shown = " ".join(self._live_lines + ([self._live_partial] if self._live_partial else []))
        self.ui_widget.update_text_signal.emit(shown)

    def _finalize_live(self, audio_data, lang):
        """El texto de la toma, y cuánto de eso todavía no se escribió en vivo."""
        if self._live is None:
            if self.transcriber is None:
                return "", ""
            text = self.transcriber.transcribe(audio_data, language=lang)
            return text, text

        # Esperar al hilo ANTES de tocar la sesión: puede estar en medio de una
        # decodificación y ni SegmentASR ni el stream de Moonshine son
        # thread-safe. No es tiempo perdido: es justo el trabajo adelantado.
        self._live_stop.set()
        if self._live_thread is not None:
            self._live_thread.join()

        # Lo que el hilo no llegó a consumir. Contado en muestras, así que no
        # puede haber ni un bloque repetido ni uno perdido.
        residual = audio_data[self._live_fed:]
        if len(residual):
            self._live.insert_audio(residual)
        text = self._live.finish()
        pending = self._live.pending_text()
        self._live = None
        return text, pending

    def _finish_take(self, audio_data, lang):
        try:
            text, pending = self._finalize_live(audio_data, lang)
            # En modo toggle las frases ya salieron por on_line a medida que
            # cerraban; solo falta lo que quedó a medio (Moonshine) o nada
            # (whisper). En modo hold va todo ahora.
            to_inject = pending if self._live_typing else text
            if to_inject:
                print(f"[Orchestrator] Injecting text: {to_inject}")
                self._inject_q.put(("paste", to_inject))
            self._inject_q.join()
        except Exception as e:
            print(f"[Orchestrator] Error during transcription/injection: {e}")
        finally:
            self._finalizing = False
            self.ui_widget.update_text_signal.emit("")
            self.ui_widget.update_ui_signal.emit("ready")

    # ----------------------------------------------------------- inyección --

    def _injector_loop(self):
        while True:
            kind, text = self._inject_q.get()
            try:
                # Con Ctrl/Win apretados las apps descartan lo inyectado (ver
                # text_injector.py). En modo hold ya están sueltos; en toggle el
                # usuario los suelta enseguida después de pulsar.
                if not wait_modifiers_released(timeout=3.0):
                    print("[Orchestrator] Modificadores apretados 3s: inyecto igual.")
                if kind == "type":
                    type_text(text)
                else:
                    paste_text(text)
            except Exception as e:  # noqa: BLE001
                print(f"[Orchestrator] Error inyectando texto: {e}")
            finally:
                self._inject_q.task_done()

    def inject_text(self, text):
        """Compatibilidad: pegar un texto completo (camino viejo)."""
        if text:
            self._inject_q.put(("paste", text))

    # ---------------------------------------------------------------- loop --

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
        threading.Thread(target=self._injector_loop, daemon=True).start()
        self.load_model()
        self.start_keep_alive()
        mode = self.config.get("hotkey_mode", "hold")
        print(f"[Orchestrator] Ready ({self.engine_name}, hotkey {mode}). Ctrl+Windows para dictar.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

if __name__ == '__main__':
    # Ensure PyQt doesn't quit if settings window closes
    QApplication.setQuitOnLastWindowClosed(False)
    app = QApplication(sys.argv)

    # UI
    ui = FloatingWidget()
    ui.show()

    # Globo de dictado: aparece abajo al centro cuando empieza la toma, muestra
    # el texto a medida que se reconoce y desaparece cuando termina. Escucha
    # las mismas señales que la píldora.
    bubble = DictationBubble()
    ui.update_ui_signal.connect(bubble.handle_state_change)
    ui.update_text_signal.connect(bubble.set_text)

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

    # Batch transcription window — shares the orchestrator's Transcriber
    # via a lambda so it always sees the current instance (after model
    # reloads triggered by settings changes).
    batch_win = BatchTranscriptionWindow(
        dictation_transcriber_provider=lambda: orchestrator.transcriber,
        config_provider=lambda: orchestrator.config,
    )

    # System Tray
    tray_icon = QSystemTrayIcon(create_tray_icon_pixmap(), app)
    tray_menu = QMenu()

    batch_action = tray_menu.addAction("Transcribir archivo…")
    batch_action.triggered.connect(lambda: (batch_win.show(), batch_win.raise_(), batch_win.activateWindow()))

    config_action = tray_menu.addAction("Configuración")
    config_action.triggered.connect(settings_win.show)

    tray_menu.addSeparator()

    quit_action = tray_menu.addAction("Salir")
    quit_action.triggered.connect(app.quit)

    tray_icon.setContextMenu(tray_menu)
    tray_icon.setToolTip("Whisper Dictation")
    tray_icon.show()

    sys.exit(app.exec())
