import queue
import sys
import numpy as np
import threading
import time
from pynput import keyboard

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QCursor

from audio_capture import AudioRecorder
from segment_asr import SegmentASR
import corrections
from spellcheck_win import SystemSpellChecker
from transcription_engine import Transcriber
from config_manager import load_config
from settings_ui import SettingsWindow
from audio_ducking import AudioDucker
from dictation_bubble import DictationBubble
from batch_window import BatchTranscriptionWindow
from text_injector import paste_text, type_text, wait_modifiers_released
import app_log

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


class _HybridLive:
    """Moonshine para MOSTRAR, Whisper para ESCRIBIR, sobre el mismo audio.

    POR QUÉ: Moonshine es el único que muestra texto mientras hablás en esta
    CPU, pero es el menos exacto de los dos (MEDIDO en openwhisper.log: inventa
    `superiferia`, `varija`, `barch`). Whisper acierta más pero no puede
    mostrar nada en vivo. Acá cada uno hace lo que sabe: el globo se llena con
    Moonshine a los ~1,2s, y lo que termina escrito en tu aplicación es de
    Whisper.

    POR QUÉ WHISPER VA EN SU PROPIO HILO: `SegmentASR.poll()` BLOQUEA mientras
    decodifica un segmento (medido en la Pi: segundos). Si se lo llamara desde
    el hilo que alimenta a Moonshine, cada segmento de Whisper congelaría el
    texto en vivo justo lo que tarda en decodificar, que es exactamente lo que
    el híbrido viene a evitar. Así que el audio se le pasa por una cola y él
    decodifica a su ritmo, sin frenar a nadie.

    La espera al soltar NO vuelve a ser la del Whisper de antes: mientras
    hablabas ya decodificó todo lo que cerró por VAD, y al final solo le queda
    la cola.
    """

    def __init__(self, moonshine_session, segmenter: SegmentASR):
        self.ms = moonshine_session
        self.seg = segmenter
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                self.seg.insert_audio(chunk)
                self.seg.poll()
            except Exception as e:  # noqa: BLE001 - nunca matar el hilo
                print(f"[Hibrido] Whisper: {e}")
            finally:
                self._q.task_done()

    def insert_audio(self, chunk):
        # Moonshine primero y en este hilo: es incremental y barato, y es el
        # que le da la cara al usuario.
        self.ms.insert_audio(chunk)
        self._q.put(np.asarray(chunk, dtype=np.float32).copy())

    def poll(self):
        pass        # lo hace el worker

    def finish(self) -> str:
        """Devuelve el texto de WHISPER: es el que se escribe."""
        try:
            self.ms.finish()        # cierra el stream de Moonshine (ya no se usa)
        except Exception as e:  # noqa: BLE001
            print(f"[Hibrido] Moonshine al cerrar: {e}")
        self._q.join()              # que termine de tragar lo encolado
        self._stop.set()
        self._worker.join(timeout=30.0)
        return self.seg.finish()

    def pending_text(self) -> str:
        return ""   # SegmentASR entrega todo por on_commit, igual que _SegmentLive


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
        self.nemotron = None
        # Corrector del sistema: lo crea main() en el hilo de Qt (es COM) y
        # nos lo pasa para poder ignorar el vocabulario propio del usuario.
        self.speller = None
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
        self._t_take = self._t_release = 0.0
        self._t_first_text = None

        # Todo lo que se escribe en la app destino pasa por UNA cola y UN hilo:
        # así las frases en vivo y el texto final salen en orden, y la espera a
        # que el usuario suelte los modificadores no frena el audio.
        self._inject_q = queue.Queue()
        # Frases en vivo que NO se pudieron tipear porque el usuario seguía con
        # modificadores apretados. Se pegan junto con el texto final: nunca se
        # tipea con Ctrl abajo (sale basura y dispara atajos), nunca se pierde.
        self._deferred = []

    # ------------------------------------------------------------ motores --

    def _on_download_progress(self, done_mb, total_mb):
        if total_mb <= 0:
            return
        pct = int(min(100, max(0, (done_mb / total_mb) * 100)))
        self.ui_widget.update_loading_progress_signal.emit(pct)

    def load_model(self):
        engine = self.config.get("engine", "moonshine")
        if engine == "nemotron":
            try:
                self._load_nemotron()
                self.ui_widget.update_ui_signal.emit("ready")
                return
            except Exception as e:  # noqa: BLE001 - sin Nemotron, Moonshine sirve
                print(f"[Orchestrator] Nemotron no cargo ({e}); uso Moonshine.")
                engine = "moonshine"
        if engine == "hibrido":
            # El híbrido necesita LOS DOS. Si Moonshine no carga se degrada a
            # whisper solo (se pierde el vivo, no la exactitud); si el que no
            # carga es Whisper, queda Moonshine solo.
            try:
                self._load_moonshine()
            except Exception as e:  # noqa: BLE001
                print(f"[Orchestrator] Moonshine no cargó ({e}); híbrido → whisper.")
                engine = "whisper"
            else:
                try:
                    self._load_whisper(keep_moonshine=True)
                    self.engine_name = "hibrido"
                except Exception as e:  # noqa: BLE001
                    print(f"[Orchestrator] Whisper no cargó ({e}); híbrido → moonshine.")
                    self.engine_name = "moonshine"
        elif engine == "moonshine":
            try:
                self._load_moonshine()
            except Exception as e:  # noqa: BLE001 - sin Moonshine, Whisper sigue sirviendo
                print(f"[Orchestrator] Moonshine no cargó ({e}); uso Whisper.")
                engine = "whisper"
        if engine == "whisper":
            self._load_whisper()
        self.ui_widget.update_ui_signal.emit("ready")

    def _vocabulary(self) -> str:
        """Lo que escribiste en Configuración más lo que fuiste corrigiendo.

        Las correcciones entran como keyterms con su grafía exacta: los docs de
        Moonshine piden escribir el término tal como querés verlo salir, y eso
        es justo lo que el usuario tipeó en el globo."""
        return corrections.vocabulary_string(self.config.get("custom_vocabulary", ""))

    def _rebuild_speller(self):
        """Rehacer el corrector cuando cambia el idioma. Corre en el hilo de
        Qt (viene de settings_saved), que es donde vive su objeto COM."""
        try:
            self.speller = SystemSpellChecker(self.config.get("language", "es"))
            self.ui_widget.set_spellchecker(self.speller)
            self._teach_speller()
        except Exception as exc:  # noqa: BLE001 - sin corrector se dicta igual
            print(f"[Ortografia] No pude rehacer el corrector: {exc}")

    def _teach_speller(self, vocab=None):
        """Que el corrector deje de marcar lo que el usuario ya declaró suyo:
        si no, `nitoOS` y `ReSpeaker` quedarían subrayados para siempre."""
        if self.speller is None:
            return
        terms = corrections._split(vocab if vocab is not None else self._vocabulary())
        self.speller.ignore_all(terms)

    def _load_nemotron(self):
        """Nemotron 3.5 streaming: el unico que muestra en vivo Y puntua.
        Ver nemotron_engine.py para los numeros que lo justifican."""
        from nemotron_engine import NemotronEngine
        from system_info import resolve_cpu_threads

        self.nemotron = NemotronEngine(
            num_threads=resolve_cpu_threads(self.config.get("cpu_threads", 0)),
            progress_cb=self._on_download_progress,
        )
        self.nemotron.set_vocabulary(self._vocabulary())
        self.transcriber = None
        self.moonshine = None
        self.engine_name = "nemotron"

    def _load_moonshine(self):
        from moonshine_engine import MoonshineEngine

        lang = self.config.get("language")
        self.moonshine = MoonshineEngine(
            language=lang,
            model_size=self.config.get("model_size", "small"),
            vocabulary=self._vocabulary(),
            progress_cb=self._on_download_progress,
        )
        self.transcriber = None
        self.nemotron = None
        self.engine_name = "moonshine"

    def _load_whisper(self, keep_moonshine=False):
        self.transcriber = Transcriber(
            model_size=self.config.get("model_size", "small"),
            cpu_threads=self.config.get("cpu_threads", 0),
            vocabulary=self._vocabulary(),
            beam_size=self.config.get("beam_size", 1),
            progress_cb=self._on_download_progress,
        )
        self.nemotron = None
        if not keep_moonshine:
            self.moonshine = None
            self.engine_name = "whisper"

    def _engine_ready(self) -> bool:
        if self._finalizing:
            return False
        if self.engine_name == "nemotron":
            return self.nemotron is not None
        if self.engine_name == "hibrido":
            return self.moonshine is not None and self.transcriber is not None
        if self.engine_name == "moonshine":
            return self.moonshine is not None
        return self.transcriber is not None

    def apply_new_config(self, new_config):
        """Aplicar la config que guardó la ventana de Configuración.

        La COPIA es obligatoria, no una precaución: SettingsWindow muta su
        propio dict y emite ESE objeto. Si lo guardáramos por referencia, en el
        guardado siguiente `old` y `new_config` serían el mismo dict, toda
        comparación daría igual y `needs_reload` quedaría en False para
        siempre: cambiar de motor o de modelo no haría nada hasta reiniciar.
        """
        print("[Orchestrator] Applying new config...")
        old = self.config
        self.config = dict(new_config)
        new_config = self.config

        needs_reload = (
            new_config.get("engine", "moonshine") != old.get("engine", "moonshine") or
            new_config.get("model_size") != old.get("model_size") or
            new_config.get("cpu_threads", 0) != old.get("cpu_threads", 0) or
            (self.engine_name in ("moonshine", "hibrido") and
             new_config.get("language") != old.get("language"))
        )

        if needs_reload:
            self.ui_widget.update_ui_signal.emit("loading")
            threading.Thread(target=self.load_model, daemon=True).start()
            return
        # Vocabulary and beam_size can be updated live without reloading.
        if new_config.get("language") != old.get("language"):
            self._rebuild_speller()
        vocab = corrections.vocabulary_string(new_config.get("custom_vocabulary", ""))
        self._teach_speller(vocab)
        if self.transcriber is not None:
            self.transcriber.set_vocabulary(vocab)
            self.transcriber.beam_size = new_config.get("beam_size", 1)
        if self.moonshine is not None:
            self.moonshine.set_vocabulary(vocab)
        if self.nemotron is not None:
            self.nemotron.set_vocabulary(vocab)

    def on_correction(self, wrong: str, right: str):
        """El usuario corrigió una palabra en el globo. Llega del hilo de Qt."""
        corrections.add(wrong, right)
        vocab = self._vocabulary()
        self._teach_speller(vocab)
        # En caliente: cambiar el sesgo no recarga el modelo, así que la
        # palabra ya vale para la toma siguiente.
        if self.transcriber is not None:
            self.transcriber.set_vocabulary(vocab)
        if self.moonshine is not None:
            self.moonshine.set_vocabulary(vocab)
        if self.nemotron is not None:
            self.nemotron.set_vocabulary(vocab)

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
        self._t_take = time.perf_counter()
        self._t_first_text = None
        mic = self.config.get("microphone", "default")
        try:
            import sounddevice as sd
            mic_name = sd.query_devices(None if mic == "default" else mic, "input")["name"]
        except Exception:
            mic_name = "?"
        print(f"[Take] inicio (motor {self.engine_name}, modo {self.config.get('hotkey_mode', 'hold')}, mic {mic} = {mic_name!r})")
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
        self._t_release = time.perf_counter()
        print(f"[Take] soltar a los {self._t_release - self._t_take:.1f}s")
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
        if self.engine_name == "nemotron" and self.nemotron is not None:
            # Un solo stream por toma: los parciales van al globo y el texto
            # entero sale en finish(). `on_line` no dispara nunca (cortar por
            # frases le arruina la puntuacion, ver nemotron_engine.py).
            self._live = self.nemotron.start_session(
                on_partial=self._on_partial,
                on_error=lambda e: print(f"[Orchestrator] Nemotron: {e}"),
            )
            period = 0.1
        elif self.engine_name == "hibrido" and self.moonshine is not None                 and self.transcriber is not None:
            # Moonshine solo MUESTRA (por eso on_line no escribe nada), Whisper
            # solo ESCRIBE (por eso su on_commit no toca el globo). Al soltar,
            # el globo pasa a mostrar el texto de Whisper, que es el que se
            # inyectó: ver el `finally` de _finish_take.
            session = self.moonshine.start_session(
                on_partial=self._on_partial,
                on_line=self._on_line_display,
                on_error=lambda e: print(f"[Hibrido] Moonshine: {e}"),
            )
            self._live = _HybridLive(session, self._new_segmenter(self._on_line_type))
            period = 0.1
        elif self.engine_name == "moonshine" and self.moonshine is not None:
            self._live = self.moonshine.start_session(
                on_partial=self._on_partial,
                on_line=self._on_line,
                on_error=lambda e: print(f"[Orchestrator] Moonshine: {e}"),
            )
            period = 0.1
        elif (self.engine_name == "whisper" and self.transcriber is not None
              and getattr(self.transcriber, "model", None) is not None
              and self.config.get("dictation_mode", "segment") == "segment"):
            self._live = _SegmentLive(self._new_segmenter(self._on_line))
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

    def _new_segmenter(self, on_commit) -> SegmentASR:
        """SegmentASR listo para esta toma, con el idioma y prompt de la config."""
        lang = self._language()
        try:
            prompt = self.transcriber._build_prompt(lang)
        except Exception:
            prompt = None
        return SegmentASR(
            self.transcriber.model, lang,
            beam_size=self.transcriber.beam_size,
            base_prompt=prompt,
            on_error=lambda e: print(f"[Orchestrator] Segmento falló: {e}"),
            on_commit=on_commit,
        )

    def _on_line_display(self, text):
        """Frase cerrada que solo se MUESTRA (híbrido: viene de Moonshine)."""
        self._live_lines.append(text)
        self._live_partial = ""
        self._push_live_text()

    def _on_line_type(self, text):
        """Frase cerrada que solo se ESCRIBE (híbrido: viene de Whisper).

        No toca el globo a propósito: ahí ya está el texto de Moonshine, y
        mezclar los dos mostraría la misma frase dos veces, segmentada distinto.
        """
        text = corrections.apply(text)   # ver _on_line: en toggle esto ya se escribe
        print(f"[Take] whisper cerró a los {time.perf_counter() - self._t_take:.1f}s: {text[:80]!r}")
        if self._live_typing:
            self._inject_q.put(("type", text + " "))

    def _on_partial(self, text):
        """Parcial de la frase en curso. Solo se MUESTRA: Moonshine lo reescribe."""
        self._live_partial = text
        self._push_live_text()

    def _on_line(self, text):
        """Una frase cerró. Es definitiva: va al widget y, en modo toggle, al destino.

        Se corrige ACÁ y no solo en _finish_take porque en modo toggle esta
        frase se escribe ya mismo en la aplicación destino: si el vocabulario
        aprendido se aplicara únicamente al final, toggle nunca lo vería."""
        text = corrections.apply(text)
        print(f"[Take] frase a los {time.perf_counter() - self._t_take:.1f}s: {text[:80]!r}")
        self._live_lines.append(text)
        self._live_partial = ""
        self._push_live_text()
        if self._live_typing:
            self._inject_q.put(("type", text + " "))

    def _push_live_text(self):
        shown = " ".join(self._live_lines + ([self._live_partial] if self._live_partial else []))
        if shown and self._t_first_text is None:
            self._t_first_text = time.perf_counter()
            print(f"[Take] primer texto a los {self._t_first_text - self._t_take:.2f}s: {shown[:60]!r}")
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

    def _save_take_audio(self, audio_data):
        """Guarda el audio de la toma (last_take.wav, y el anterior como
        prev_take.wav) para poder diagnosticar un reconocimiento malo con el
        audio real, no con suposiciones. 12s de dictado son ~380 KB."""
        try:
            import os
            import wave
            from config_manager import CONFIG_DIR
            last = os.path.join(CONFIG_DIR, "last_take.wav")
            prev = os.path.join(CONFIG_DIR, "prev_take.wav")
            if os.path.exists(last):
                os.replace(last, prev)
            pcm = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(last, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            peak = float(np.max(np.abs(audio_data))) if len(audio_data) else 0.0
            rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0
            print(f"[Take] audio: pico {peak:.3f}, RMS {rms:.4f} → {last}")
        except Exception as e:  # noqa: BLE001 - el diagnóstico nunca rompe la toma
            print(f"[Take] no pude guardar el audio: {e}")

    def _finish_take(self, audio_data, lang):
        text = ""   # lo lee el `finally` para dejar el globo corregible
        try:
            t0 = time.perf_counter()
            print(f"[Take] audio grabado: {len(audio_data) / 16000:.1f}s")
            self._save_take_audio(audio_data)
            text, pending = self._finalize_live(audio_data, lang)
            print(f"[Take] decodificación final: {time.perf_counter() - t0:.2f}s, {len(text)} chars")
            # En modo toggle las frases ya salieron por on_line a medida que
            # cerraban; solo falta lo que quedó a medio (Moonshine) o nada
            # (whisper). En modo hold va todo ahora.
            to_inject = pending if self._live_typing else text
            # Aunque el modelo vuelva a errarle a una palabra ya corregida, el
            # texto que se escribe sale bien: el sesgo hace el reconocimiento
            # más probable, esto lo garantiza.
            to_inject = corrections.apply(to_inject)
            text = corrections.apply(text)
            t1 = time.perf_counter()
            # Vaciar la cola ANTES de leer _deferred: el injector difiere las
            # frases que caen con modificadores apretados, y si lo leyéramos
            # primero, las que difiera mientras drena quedarían fuera de esta
            # toma y aparecerían fuera de orden al principio de la siguiente.
            self._inject_q.join()
            if self._deferred:
                to_inject = ("".join(self._deferred) + to_inject).strip()
                self._deferred = []
            if to_inject:
                print(f"[Orchestrator] Injecting text: {to_inject}")
                self._inject_q.put(("paste", to_inject))
                self._inject_q.join()
            print(f"[Take] escritura: {time.perf_counter() - t1:.2f}s · espera total desde soltar: {time.perf_counter() - self._t_release:.2f}s")
        except Exception as e:
            print(f"[Orchestrator] Error during transcription/injection: {e}")
        finally:
            self._finalizing = False
            # El globo no se desvanece todavía: queda unos segundos mostrando
            # lo que entendió, para poder señalarle una palabra mal. Si no hay
            # texto, "correctable" se comporta como "ready" y se va igual.
            self.ui_widget.update_text_signal.emit(text)
            self.ui_widget.update_ui_signal.emit("correctable")

    # ----------------------------------------------------------- inyección --

    def _injector_loop(self):
        while True:
            kind, text = self._inject_q.get()
            try:
                # Con Ctrl/Win apretados las apps descartan lo inyectado y Win+letra
                # dispara atajos (ver text_injector.py). En modo hold ya están
                # sueltos; en toggle el usuario los suelta enseguida... salvo que
                # crea que es modo hold y los mantenga. Entonces NO se tipea: la
                # frase se guarda y sale con el pegado final.
                if kind == "type":
                    if wait_modifiers_released(timeout=2.0):
                        type_text(text)
                    else:
                        print("[Orchestrator] Modificadores apretados: difiero la frase al final.")
                        self._deferred.append(text)
                else:
                    if not wait_modifiers_released(timeout=15.0):
                        print("[Orchestrator] Modificadores apretados 15s: pego igual.")
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
        self.ui_widget.update_ui_signal.emit("loading")
        self.load_model()
        self.start_keep_alive()
        mode = self.config.get("hotkey_mode", "hold")
        print(f"[Orchestrator] Ready ({self.engine_name}, hotkey {mode}). Ctrl+Windows para dictar.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

def _install_crash_diagnostics():
    """Que un crash deje rastro en el log en vez de desaparecer.

    Dos agujeros, los dos por compilar con --noconsole:
      1. PyQt6 ABORTA el proceso (qFatal) si una excepción de Python queda sin
         atrapar dentro de un método virtual de Qt, como paintEvent. Sin un
         sys.excepthook que la escriba, el log corta en seco y solo queda un
         0xc0000409 en el Visor de eventos. Así perdimos el IndexError de
         `_flagged` el 22/09.
      2. qFatal/qWarning escriben desde C++, no por sys.stderr, así que el
         _Tee de app_log no los ve. qInstallMessageHandler los trae a Python.
    """
    import faulthandler
    import traceback

    from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

    def _excepthook(tipo, valor, tb):
        print("=" * 60)
        print("EXCEPCIÓN NO ATRAPADA (si viene de un virtual de Qt, ahora aborta):")
        traceback.print_exception(tipo, valor, tb)
        print("=" * 60)
        sys.stderr.flush()

    sys.excepthook = _excepthook

    _niveles = {
        QtMsgType.QtDebugMsg: "debug",
        QtMsgType.QtInfoMsg: "info",
        QtMsgType.QtWarningMsg: "AVISO",
        QtMsgType.QtCriticalMsg: "CRÍTICO",
        QtMsgType.QtFatalMsg: "FATAL",
    }

    def _qt_handler(modo, contexto, mensaje):
        etiqueta = _niveles.get(modo, "qt")
        donde = ""
        if contexto is not None and contexto.file:
            donde = f" ({contexto.file}:{contexto.line})"
        print(f"[Qt/{etiqueta}] {mensaje}{donde}")
        sys.stderr.flush()

    qInstallMessageHandler(_qt_handler)

    # Para una caída dura de verdad (segfault): vuelca las pilas al log.
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] faulthandler no arrancó: {exc}")


if __name__ == '__main__':
    app_log.install()
    _install_crash_diagnostics()
    # Ensure PyQt doesn't quit if settings window closes
    QApplication.setQuitOnLastWindowClosed(False)
    app = QApplication(sys.argv)

    # UI: el globo de dictado es la única ventana. Aparece abajo al centro
    # mientras carga la IA y durante cada toma, y desaparece el resto del
    # tiempo. Lo demás vive en el ícono de la bandeja.
    ui = DictationBubble()
    # El corrector es COM: se crea en el hilo de Qt, que es el único que lo usa
    # (marcar palabras al pintar el globo, e ignorar términos al corregir).
    speller = SystemSpellChecker(load_config().get("language", "es"))
    ui.set_spellchecker(speller)

    # Orchestrator
    orchestrator = Orchestrator(ui)
    bg_thread = threading.Thread(target=orchestrator.run, daemon=True)
    bg_thread.start()

    # Settings Window
    settings_win = SettingsWindow()
    settings_win.settings_saved.connect(orchestrator.apply_new_config)
    ui.correction_made.connect(orchestrator.on_correction)
    orchestrator.speller = speller
    orchestrator._teach_speller()

    # Batch transcription window — shares the orchestrator's Transcriber
    # via a lambda so it always sees the current instance (after model
    # reloads triggered by settings changes).
    batch_win = BatchTranscriptionWindow(
        dictation_transcriber_provider=lambda: orchestrator.transcriber,
        config_provider=lambda: orchestrator.config,
        nemotron_provider=lambda: orchestrator.nemotron,
    )

    # System Tray
    tray_icon = QSystemTrayIcon(create_tray_icon_pixmap(), app)
    tray_menu = QMenu()

    def show_batch():
        batch_win.show()
        batch_win.raise_()
        batch_win.activateWindow()

    batch_action = tray_menu.addAction("Transcribir archivo…")
    batch_action.triggered.connect(show_batch)
    # También desde Configuración, y con doble clic en el ícono: sin la píldora
    # flotante, el ícono de la bandeja es la única puerta y suele quedar
    # escondido en el desbordamiento de Windows.
    settings_win.open_batch.connect(show_batch)

    config_action = tray_menu.addAction("Configuración")
    config_action.triggered.connect(settings_win.show)

    tray_menu.addSeparator()

    quit_action = tray_menu.addAction("Salir")
    quit_action.triggered.connect(app.quit)

    tray_icon.setContextMenu(tray_menu)
    tray_icon.setToolTip("OpenWhisper — Ctrl+Win para dictar · doble clic: transcribir archivo · clic derecho: menú")

    def on_tray_activated(reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            show_batch()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            tray_menu.popup(QCursor.pos())

    tray_icon.activated.connect(on_tray_activated)
    tray_icon.show()

    if "--batch" in sys.argv[1:]:
        show_batch()

    sys.exit(app.exec())
