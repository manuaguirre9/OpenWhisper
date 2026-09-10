"""
Globo de dictado: aparece abajo al centro de la pantalla cuando empieza una
toma, muestra el texto a medida que se va reconociendo, y desaparece cuando
la toma termina. Es solo lectura: no acepta foco ni clics, para que el texto
que la app escribe siga yendo a la ventana que el usuario tenía activa.

Es la única UI de la app (además del ícono de la bandeja). El orquestador le
habla por señales, desde sus hilos:
    update_ui_signal("loading")          → aparece con "Cargando IA…"
    update_loading_progress_signal(pct)  → porcentaje de descarga del modelo
    update_ui_signal("ready")            → desaparece
    update_ui_signal("recording")        → aparece, vacío, punto rojo
    update_text_signal(texto)            → reemplaza el texto que muestra
    update_ui_signal("processing")       → sigue visible, con puntos animados
"""
import math

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import (
    Qt, QTimer, QRect, QRectF, QPointF, QPoint, QPropertyAnimation,
    QParallelAnimationGroup, QEasingCurve, QSize, pyqtSignal,
)
from PyQt6.QtGui import QFont, QFontMetrics, QPainter, QColor

# Paleta compartida con batch_window.py: negro mate, sin borde ni sombra.
WF_SURFACE = QColor(14, 14, 18, 255)
WF_ON_SURFACE = QColor(232, 232, 240, 235)
WF_ON_SURFACE_MUTED = QColor(170, 170, 182, 200)
WF_ACCENT_RECORDING = QColor(255, 112, 122)   # coral
WF_ACCENT_PROCESSING = QColor(180, 160, 255)  # lavanda


class DictationBubble(QWidget):
    update_ui_signal = pyqtSignal(str)
    update_loading_progress_signal = pyqtSignal(int)
    update_text_signal = pyqtSignal(str)

    MAX_W = 620          # ancho máximo del globo
    MIN_W = 260
    PAD_X = 22
    PAD_Y = 14
    RADIUS = 18
    INDICATOR_W = 26     # espacio para el punto rojo / los puntos de proceso
    BOTTOM_MARGIN = 56   # distancia al borde inferior del área útil (sobre la barra de tareas)
    MAX_CHARS = 320      # más que esto se muestra solo la cola, con "…" adelante
    FONT = ("Segoe UI", 13)
    PLACEHOLDER = "Escuchando…"
    LOADING_TEXT = "Cargando IA…"
    ANIM_MS = 170
    SLIDE_PX = 12

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        # No robar el foco al aparecer: el texto dictado va a la app activa.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.state = "hidden"
        self._text = ""
        self._phase = 0.0
        self._font = QFont(*self.FONT)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._anim = None
        self._loading_percent = None
        self._layout_text()

        self.update_ui_signal.connect(self.handle_state_change)
        self.update_loading_progress_signal.connect(self._handle_loading_progress)
        self.update_text_signal.connect(self.set_text)

    # ------------------------------------------------------------ señales --

    def handle_state_change(self, state):
        if state == "loading":
            self._text = ""
            self._loading_percent = None
            self.state = "loading"
            self._layout_text()
            if not self.isVisible():
                self._appear()
            self.update()
        elif state == "recording":
            self._text = ""
            self.state = "recording"
            self._layout_text()
            self._appear()
        elif state == "processing":
            if self.state == "hidden":
                return
            self.state = "processing"
            self.update()
        else:  # ready, loading
            if self.state != "hidden":
                self._vanish()

    def _handle_loading_progress(self, percent):
        try:
            self._loading_percent = max(0, min(100, int(percent)))
        except (TypeError, ValueError):
            return
        if self.state == "loading":
            self.update()

    def _placeholder(self) -> str:
        if self.state == "loading":
            if self._loading_percent is None:
                return self.LOADING_TEXT
            return f"{self.LOADING_TEXT} {self._loading_percent}%"
        return self.PLACEHOLDER

    def set_text(self, text):
        text = (text or "").strip()
        # Al cerrar la toma la app manda "" antes de "ready": no vaciar el
        # globo mientras se desvanece, que si no salta a su tamaño mínimo.
        if not text and self.state != "recording":
            return
        if len(text) > self.MAX_CHARS:
            text = "…" + text[-self.MAX_CHARS:]
        if text == self._text:
            return
        self._text = text
        if self.state != "hidden":
            self._layout_text()
            self.update()

    # ------------------------------------------------------------- layout --

    def _screen_rect(self) -> QRect:
        screen = QApplication.primaryScreen()
        return screen.availableGeometry() if screen else QRect(0, 0, 1280, 720)

    def _layout_text(self):
        """Recalcula el tamaño para el texto actual y recentra abajo."""
        fm = QFontMetrics(self._font)
        text = self._text or self._placeholder()
        text_max_w = self.MAX_W - 2 * self.PAD_X - self.INDICATOR_W
        bound = fm.boundingRect(QRect(0, 0, text_max_w, 10_000),
                                int(Qt.TextFlag.TextWordWrap), text)
        w = max(self.MIN_W, min(self.MAX_W, bound.width() + 2 * self.PAD_X + self.INDICATOR_W))
        h = bound.height() + 2 * self.PAD_Y
        self._text_rect = QRectF(self.PAD_X + self.INDICATOR_W, self.PAD_Y,
                                 w - 2 * self.PAD_X - self.INDICATOR_W, bound.height())
        self._place(QSize(int(w), int(h)))

    def _place(self, size: QSize):
        screen = self._screen_rect()
        x = screen.center().x() - size.width() // 2
        y = screen.bottom() - self.BOTTOM_MARGIN - size.height()
        self._rest_pos = QPoint(x, y)
        self.setGeometry(x, y, size.width(), size.height())

    # ---------------------------------------------------------- animación --

    def _stop_anim(self):
        if self._anim is not None:
            self._anim.stop()
            self._anim = None

    def _appear(self):
        self._stop_anim()
        self.setWindowOpacity(0.0)
        start = QPoint(self._rest_pos.x(), self._rest_pos.y() + self.SLIDE_PX)
        self.move(start)
        self.show()
        self._timer.start(50)

        fade = QPropertyAnimation(self, b"windowOpacity", self)
        fade.setDuration(self.ANIM_MS)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        slide = QPropertyAnimation(self, b"pos", self)
        slide.setDuration(self.ANIM_MS)
        slide.setStartValue(start)
        slide.setEndValue(self._rest_pos)
        slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        group = QParallelAnimationGroup(self)
        group.addAnimation(fade)
        group.addAnimation(slide)
        group.start()
        self._anim = group

    def _vanish(self):
        self._stop_anim()
        self.state = "hidden"
        end = QPoint(self._rest_pos.x(), self._rest_pos.y() + self.SLIDE_PX)
        fade = QPropertyAnimation(self, b"windowOpacity", self)
        fade.setDuration(self.ANIM_MS)
        fade.setStartValue(self.windowOpacity())
        fade.setEndValue(0.0)
        slide = QPropertyAnimation(self, b"pos", self)
        slide.setDuration(self.ANIM_MS)
        slide.setStartValue(self.pos())
        slide.setEndValue(end)
        slide.setEasingCurve(QEasingCurve.Type.InCubic)
        group = QParallelAnimationGroup(self)
        group.addAnimation(fade)
        group.addAnimation(slide)

        def done():
            if self.state == "hidden":
                self.hide()
                self._timer.stop()
                self._text = ""
        group.finished.connect(done)
        group.start()
        self._anim = group

    def _tick(self):
        self._phase = (self._phase + 0.25) % (2.0 * math.pi)
        self.update()

    # ------------------------------------------------------------ pintado --

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect())
        painter.setBrush(WF_SURFACE)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, self.RADIUS, self.RADIUS)

        # Indicador a la izquierda, alineado con la primera línea de texto.
        fm = QFontMetrics(self._font)
        cy = self.PAD_Y + fm.height() / 2.0
        cx = self.PAD_X + 6
        if self.state == "recording":
            pulse = (math.sin(self._phase) + 1.0) * 0.5
            c = QColor(WF_ACCENT_RECORDING)
            c.setAlpha(int(150 + 105 * pulse))
            painter.setBrush(c)
            painter.drawEllipse(QPointF(cx, cy), 4.5 + 1.5 * pulse, 4.5 + 1.5 * pulse)
        else:
            for i in range(3):
                phase = self._phase * 1.8 - i * 0.7
                t = (math.sin(phase) + 1.0) * 0.5
                c = QColor(WF_ACCENT_PROCESSING)
                c.setAlpha(int(170 + 80 * t))
                painter.setBrush(c)
                painter.drawEllipse(QPointF(cx - 6 + i * 6, cy - 2.5 * math.sin(phase)), 2.4, 2.4)

        painter.setFont(self._font)
        if self._text:
            painter.setPen(WF_ON_SURFACE)
            painter.drawText(self._text_rect,
                             int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                             self._text)
        else:
            painter.setPen(WF_ON_SURFACE_MUTED)
            painter.drawText(self._text_rect,
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             self._placeholder())
        painter.end()
