"""
Floating overlay pill that lives on top of every window and reflects the
current dictation state (loading / ready / recording / processing).

The widget paints itself with a custom paintEvent — no stylesheet, no child
widgets. State is set externally via the `update_ui_signal` Qt signal.
"""

import math
import sys

import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF
from PyQt6.QtGui import QFont, QPainter, QColor, QPen

from config_manager import load_config, save_config


# WhisperFlow-inspired palette: deep matte black pill, no border, no shadow.
WF_SURFACE = QColor(14, 14, 18, 255)
WF_ON_SURFACE = QColor(232, 232, 240, 235)
WF_ON_SURFACE_MUTED = QColor(170, 170, 182, 200)
WF_ACCENT_RECORDING = QColor(255, 112, 122)   # soft coral red
WF_ACCENT_PROCESSING = QColor(180, 160, 255)  # soft lavender


class FloatingWidget(QWidget):
    update_ui_signal = pyqtSignal(str)

    NUM_BARS = 26
    # Widget and pill share the exact same bounds — no transparent gap
    # around the pill that could be misread as a frame.
    WIDGET_W = 220
    WIDGET_H = 44
    PILL_INSET_X = 0
    PILL_TOP = 0
    PILL_HEIGHT = WIDGET_H

    _TICK_INTERVALS = {
        "loading": 80,
        "ready": 200,
        "recording": 28,
        "processing": 50,
    }

    def __init__(self, recorder=None):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Prevent Qt from drawing any system background behind the widget,
        # which on some Windows themes shows up as a faint outer frame.
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self.recorder = recorder
        self.state = "loading"

        self.config = load_config()
        x = self.config.get("pos_x", 100)
        y = self.config.get("pos_y", 100)
        self.setGeometry(x, y, self.WIDGET_W, self.WIDGET_H)

        self._bar_levels = np.zeros(self.NUM_BARS, dtype=np.float32)
        self._anim_phase = 0.0

        self.update_ui_signal.connect(self.handle_state_change)
        self.oldPos = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._TICK_INTERVALS["loading"])

    def set_recorder(self, recorder):
        self.recorder = recorder

    def showEvent(self, event):
        super().showEvent(event)
        self._disable_windows_corner_rounding()

    def _disable_windows_corner_rounding(self):
        """
        Windows 11 DWM auto-rounds every top-level window with ~8px radius,
        even when FramelessWindowHint is set. That clip is what produced the
        faint gray crescents between our 22px pill corners and the DWM clip.
        DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_DONOTROUND = 1.
        No-op on non-Windows or pre-Win11 (the API call just fails silently).
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = int(self.winId())
            pref = ctypes.c_int(1)  # DWMWCP_DONOTROUND
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 33, ctypes.byref(pref), ctypes.sizeof(pref)
            )
        except Exception as e:
            print(f"[FloatingWidget] DwmSetWindowAttribute failed: {e}")

    def handle_state_change(self, state):
        if state == self.state:
            return
        self.state = state
        interval = self._TICK_INTERVALS.get(state, 100)
        self._timer.start(interval)
        if state == "recording":
            self._bar_levels[:] = 0.0
        self.update()

    def _tick(self):
        self._anim_phase = (self._anim_phase + 0.22) % (2.0 * math.pi)

        if self.state == "recording" and self.recorder is not None:
            latest = self.recorder.get_latest_level()
            # Symmetric equalizer: bars are organized by distance to the
            # center of the pill. The fade factor below pushes the visual
            # focus into the middle of the waveform, just like WhisperFlow.
            n = self.NUM_BARS
            half = (n - 1) / 2.0
            for i in range(n):
                # 0 at center, 1 at edges
                d = abs(i - half) / half
                # Soft cosine-like fade so the wave tapers smoothly outward
                edge_fade = max(0.0, 1.0 - d ** 1.4)

                if latest < 0.04:
                    target = 0.0
                else:
                    phase = self._anim_phase * 1.6 + i * 0.4
                    wiggle = (math.sin(phase) + 1.0) * 0.5
                    target = latest * (0.55 + 0.45 * wiggle) * edge_fade

                if target > self._bar_levels[i]:
                    self._bar_levels[i] += 0.78 * (target - self._bar_levels[i])
                else:
                    self._bar_levels[i] *= 0.84
        else:
            self._bar_levels *= 0.80

        self.update()

    # ----- painting -----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pill_rect = QRectF(
            self.PILL_INSET_X,
            self.PILL_TOP,
            self.width() - 2 * self.PILL_INSET_X,
            self.PILL_HEIGHT,
        )
        radius = pill_rect.height() / 2.0

        # Flat dark surface (the pill itself). No border, no shadow.
        painter.setBrush(WF_SURFACE)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pill_rect, radius, radius)

        # State-specific foreground.
        if self.state == "loading":
            self._paint_loading(painter, pill_rect)
        elif self.state == "ready":
            self._paint_ready(painter, pill_rect)
        elif self.state == "recording":
            self._paint_recording(painter, pill_rect)
        elif self.state == "processing":
            self._paint_processing(painter, pill_rect)

        painter.end()

    def _paint_loading(self, painter, rect):
        cx = rect.center().x() - 10
        cy = rect.center().y()
        self._draw_pulsing_dots(painter, cx, cy, color=WF_ON_SURFACE_MUTED, spacing=10)

    def _paint_ready(self, painter, rect):
        # Center the [icon + spacing + text] block as a single visual unit
        # so the composition feels balanced inside the pill.
        text = "Ctrl + Win para hablar"
        icon_size = 14
        spacing = 9
        font = QFont("Segoe UI", 9, QFont.Weight.Normal)
        painter.setFont(font)
        text_w = painter.fontMetrics().horizontalAdvance(text)

        total_w = icon_size + spacing + text_w
        start_x = rect.left() + (rect.width() - total_w) / 2.0
        cy = rect.center().y()

        icon_cx = start_x + icon_size / 2.0
        self._draw_mic(painter, icon_cx, cy, size=icon_size, color=WF_ON_SURFACE_MUTED)
        self._draw_text(painter, rect, text,
                        left=start_x + icon_size + spacing,
                        color=WF_ON_SURFACE_MUTED, size=9)

    def _paint_recording(self, painter, rect):
        # No text, no icon — the symmetric waveform fills the pill and is
        # the only indicator. This is the WhisperFlow look.
        bars_left = rect.left() + 14
        bars_right = rect.right() - 14
        self._draw_bars(painter, bars_left, bars_right, rect, WF_ACCENT_RECORDING)

    def _paint_processing(self, painter, rect):
        # Three dots traveling in a small wave pattern.
        cx = rect.center().x() - 16
        cy = rect.center().y()
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            phase = self._anim_phase * 1.8 - i * 0.7
            t = (math.sin(phase) + 1.0) * 0.5
            r = 3.0 + 1.8 * t
            offset_y = -3 * math.sin(phase)
            c = QColor(WF_ACCENT_PROCESSING)
            c.setAlpha(int(190 + 60 * t))
            painter.setBrush(c)
            painter.drawEllipse(QPointF(cx + i * 14, cy + offset_y), r, r)

    # ----- helpers -----

    def _draw_text(self, painter, rect, text, left, color, size=10):
        painter.setPen(color)
        painter.setFont(QFont("Segoe UI", size, QFont.Weight.Normal))
        text_rect = QRectF(left, rect.top(), rect.right() - left - 14, rect.height())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            text,
        )

    def _draw_pulsing_dots(self, painter, cx, cy, color, spacing=10):
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            phase = self._anim_phase - i * 0.55
            t = (math.sin(phase) + 1.0) * 0.5
            r = 2.6 + 1.6 * t
            c = QColor(color)
            c.setAlpha(int(180 + 60 * t))
            painter.setBrush(c)
            painter.drawEllipse(QPointF(cx + i * spacing, cy), r, r)

    def _draw_mic(self, painter, cx, cy, size, color):
        # Minimalist monochromatic mic: outline-only strokes, rounded caps.
        pen = QPen(color, 1.3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Capsule body (outline, no fill)
        body_w = size * 0.48
        body_h = size * 0.58
        body = QRectF(cx - body_w / 2, cy - size * 0.45, body_w, body_h)
        painter.drawRoundedRect(body, body_w / 2, body_w / 2)

        # Soft arc cradling the body from below
        arc = QRectF(cx - size * 0.38, cy - size * 0.08, size * 0.76, size * 0.52)
        painter.drawArc(arc, 200 * 16, 140 * 16)

        # Short stem
        painter.drawLine(
            QPointF(cx, cy + size * 0.38),
            QPointF(cx, cy + size * 0.50),
        )

    def _draw_bars(self, painter, x0, x1, rect, accent_color):
        n = self.NUM_BARS
        width = max(0.0, x1 - x0)
        if width <= 0:
            return
        gap = 2.5
        bar_w = max(2.0, (width - gap * (n - 1)) / n)
        center_y = rect.center().y()
        max_bar_h = rect.height() * 0.78
        min_bar_h = 2.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent_color)
        for i in range(n):
            level = float(self._bar_levels[i])
            bar_h = max(min_bar_h, level * max_bar_h)
            bx = x0 + i * (bar_w + gap)
            by = center_y - bar_h / 2.0
            painter.drawRoundedRect(
                QRectF(bx, by, bar_w, bar_h), bar_w / 2.0, bar_w / 2.0
            )

    # ----- drag-to-move -----

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
            current_config = load_config()
            current_config["pos_x"] = self.pos().x()
            current_config["pos_y"] = self.pos().y()
            save_config(current_config)
            self.config = current_config
