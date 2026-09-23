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
    update_ui_signal("correctable")      → queda unos segundos aceptando clics

El estado "correctable" es la única vez que el globo toca el mouse. Durante la
toma NO acepta ni clics ni foco (WA_TransparentForMouseEvents), que es lo que
garantiza que el texto dictado vaya a la ventana que el usuario tenía activa.
Al terminar la toma se le habilita el mouse —pero NO el foco, que es otra
cosa— para que se pueda señalar una palabra mal reconocida. Recién cuando el
usuario hace clic se abre `_WordEditor`, que sí toma foco, y al cerrarse se lo
devuelve a la ventana de donde vino. Ver corrections.py para qué se hace con
el par resultante.
"""
import math

from PyQt6.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton
from PyQt6.QtCore import (
    Qt, QTimer, QRect, QRectF, QPointF, QPoint, QEvent, QPropertyAnimation,
    QParallelAnimationGroup, QEasingCurve, QSize, pyqtSignal,
)
from PyQt6.QtGui import QFont, QFontMetrics, QPainter, QColor, QCursor, QPen

import corrections

# Paleta compartida con batch_window.py: negro mate, sin borde ni sombra.
WF_SURFACE = QColor(14, 14, 18, 255)
WF_ON_SURFACE = QColor(232, 232, 240, 235)
WF_ON_SURFACE_MUTED = QColor(170, 170, 182, 200)
WF_ACCENT_RECORDING = QColor(255, 112, 122)   # coral
WF_ACCENT_PROCESSING = QColor(180, 160, 255)  # lavanda
WF_SURFACE_RAISED = QColor(28, 28, 36, 255)   # campo de texto, igual que theme.py


def _foreground_window():
    """HWND de la ventana activa, para poder devolverle el foco después."""
    try:
        import ctypes
        return ctypes.windll.user32.GetForegroundWindow()
    except Exception:
        return None


def _restore_foreground(hwnd):
    """Devolver el foco a `hwnd`. Windows solo lo permite si quien llama ya es
    foreground, que es justo el caso: lo llama el editor antes de cerrarse."""
    if not hwnd:
        return
    try:
        import ctypes
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


class DictationBubble(QWidget):
    update_ui_signal = pyqtSignal(str)
    update_loading_progress_signal = pyqtSignal(int)
    update_text_signal = pyqtSignal(str)
    # (forma errada, forma correcta). La escucha el orquestador para guardar
    # el par y refrescar el vocabulario del motor en caliente.
    correction_made = pyqtSignal(str, str)

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
    LINGER_MS = 4000     # cuánto queda el globo corregible antes de irse solo
    HINT = "clic para corregir · arrastrá para varias palabras"
    HINT_FONT = ("Segoe UI", 9)
    HINT_AFTER = 3       # con esta cantidad de correcciones ya no hace falta

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
        self._hint_font = QFont(*self.HINT_FONT)
        self._lines = []      # [[(token, x, ancho), …], …] por línea visual
        self._words = []      # [(token, QRectF)] para acertarle con el mouse
        self._hover = -1      # índice en _words bajo el cursor, -1 si ninguno
        self._sel = None      # (primero, último) inclusive, al arrastrar
        self._anchor = None   # palabra donde empezó el arrastre
        self._show_hint = False
        self._flagged = set()   # índices de _words que el corrector marcó
        self._speller = None
        self._editor = None
        self._linger = QTimer(self)
        self._linger.setSingleShot(True)
        self._linger.timeout.connect(self._linger_done)
        self.setMouseTracking(True)
        self._layout_text()

        self.update_ui_signal.connect(self.handle_state_change)
        self.update_loading_progress_signal.connect(self._handle_loading_progress)
        self.update_text_signal.connect(self.set_text)

    def set_spellchecker(self, speller):
        """Corrector del sistema (ver spellcheck_win.py). Opcional: sin él el
        globo funciona igual, solo que no señala nada por su cuenta."""
        self._speller = speller

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
            self._end_correctable()
            self._text = ""
            self.state = "recording"
            self._layout_text()
            self._appear()
        elif state == "processing":
            if self.state == "hidden":
                return
            self.state = "processing"
            self.update()
        elif state == "correctable":
            # Sin texto no hay nada que corregir: irse como siempre.
            if self.state == "hidden" or not self._text:
                self._end_correctable()
                if self.state != "hidden":
                    self._vanish()
                return
            self.state = "correctable"
            self._show_hint = len(corrections.load()) < self.HINT_AFTER
            self._hover = -1
            self._sel = None
            self._anchor = None
            # A partir de acá sí recibe clics. El foco sigue sin tocarse: la
            # ventana conserva WindowDoesNotAcceptFocus, así que el usuario no
            # pierde el cursor de texto donde estaba escribiendo.
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self._timer.stop()          # sin animación: el globo ya está quieto
            self._layout_text()
            self._flag_words()          # después del layout: usa _words ya armado
            self.update()
            self._linger.start(self.LINGER_MS)
        else:  # ready, loading
            self._end_correctable()
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

    def _wrap(self, text, max_w):
        """Corta el texto en líneas a mano, guardando dónde cae cada palabra.

        Qt sabe hacer el wrap solo (TextWordWrap), pero no dice dónde quedó
        cada palabra, y sin eso no se puede saber a cuál le hiciste clic. Las
        posiciones se calculan con el mismo ancho de espacio que usa Qt al
        dibujar la línea entera, así que coinciden con lo que se ve.
        """
        fm = QFontMetrics(self._font)
        space_w = fm.horizontalAdvance(" ")
        lines, cur, cur_w = [], [], 0.0
        for token in text.split():
            tw = fm.horizontalAdvance(token)
            x = cur_w + (space_w if cur else 0.0)
            if cur and x + tw > max_w:
                lines.append((cur, cur_w))
                cur, cur_w, x = [], 0.0, 0.0
            cur.append((token, x, tw))
            cur_w = x + tw
        if cur:
            lines.append((cur, cur_w))
        return lines

    def _layout_text(self):
        """Recalcula el tamaño para el texto actual y recentra abajo."""
        fm = QFontMetrics(self._font)
        text = self._text or self._placeholder()
        text_max_w = self.MAX_W - 2 * self.PAD_X - self.INDICATOR_W
        self._lines = self._wrap(text, text_max_w)

        line_h = fm.lineSpacing()
        text_w = max([w for _, w in self._lines] or [0.0])
        text_h = fm.height() + line_h * (len(self._lines) - 1)

        hint_h = 0.0
        if self.state == "correctable" and self._show_hint:
            hint_h = QFontMetrics(self._hint_font).height() + 6

        w = max(self.MIN_W, min(self.MAX_W, text_w + 2 * self.PAD_X + self.INDICATOR_W))
        h = text_h + hint_h + 2 * self.PAD_Y
        self._text_rect = QRectF(self.PAD_X + self.INDICATOR_W, self.PAD_Y,
                                 w - 2 * self.PAD_X - self.INDICATOR_W, text_h)

        # Rectángulo de cada palabra, en coordenadas del globo. Solo sirve en
        # estado corregible, pero calcularlo siempre cuesta nada y evita que
        # queden viejos si el texto cambió.
        #
        # `_flagged` son ÍNDICES dentro de _words, así que rehacer _words los
        # invalida a todos. Limpiarlo acá y no en cada sitio que cambia el
        # texto es lo que hace que la invariante no se pueda romper: si no, un
        # índice viejo apuntando a un _words más corto explota en paintEvent, y
        # una excepción ahí adentro NO se propaga: PyQt6 llama a qFatal() y
        # mata el proceso sin traceback. Pasó (toma con `threads` marcado,
        # toma siguiente con una sola palabra).
        self._flagged = set()
        self._words = []
        if self._text:
            for i, (tokens, _) in enumerate(self._lines):
                y = self._text_rect.y() + i * line_h
                for token, x, tw in tokens:
                    self._words.append(
                        (token, QRectF(self._text_rect.x() + x, y, tw, fm.height())))
        self._place(QSize(int(w), int(h)))

    # -------------------------------------------------------- corrección --

    def _flag_words(self):
        """Marcar las palabras que el diccionario del sistema no reconoce.

        Es una PISTA, no un veredicto: sirve para no tener que releer la frase
        entera buscando dónde metió la pata. Solo ve palabras inexistentes
        (`superiferia`), nunca errores entre palabras reales (`a cerca`).
        Cuesta ~0,2 ms por frase, así que se hace de una en el hilo de Qt.
        """
        self._flagged = set()
        if self._speller is None:
            return
        for i, (token, _) in enumerate(self._words):
            clean = token.strip(".,;:¿?¡!()[]\"'«»…")
            if len(clean) < 4 or not clean.isalpha():
                continue
            try:
                if self._speller.is_misspelled(clean):
                    self._flagged.add(i)
            except Exception:
                return      # corrector roto: mejor sin subrayados que a medias

    def _suggestions_for(self, text):
        if self._speller is None or " " in text:
            return []
        try:
            return self._speller.suggest(text.strip(".,;:¿?¡!()[]\"'«»…"), 3)
        except Exception:
            return []


    def _end_correctable(self):
        """Volver a ser un cartel: ni mouse, ni timer de espera, ni editor."""
        self._linger.stop()
        self._close_editor()
        self._hover = -1
        self._sel = None
        self._anchor = None
        self._show_hint = False
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.unsetCursor()

    def _linger_done(self):
        if self.state == "correctable" and self._editor is None:
            self._end_correctable()
            self._vanish()

    def _word_at(self, pos) -> int:
        if self.state != "correctable":
            return -1
        for i, (token, rect) in enumerate(self._words):
            # Un poco de aire arriba y abajo: apuntarle a una palabra de 13px
            # con el mouse en movimiento es incómodo si el blanco es exacto.
            if rect.adjusted(-2, -3, 2, 3).contains(QPointF(pos)):
                return -1 if token == "…" else i
        return -1

    def mouseMoveEvent(self, event):
        if self._anchor is not None:
            # Arrastrando: se va abarcando el tramo. Hace falta para los
            # errores que NO son de una palabra sola, como "a cerca" por
            # "acerca": corregir solo una de las dos deja una regla que
            # después rompe todas las veces que digas esa palabra suelta.
            idx = self._word_at(event.position())
            if idx >= 0:
                sel = (min(self._anchor, idx), max(self._anchor, idx))
                if sel != self._sel:
                    self._sel = sel
                    self.update()
            return      # el arrastre ya está consumido

        idx = self._word_at(event.position())
        if idx != self._hover:
            self._hover = idx
            if idx >= 0:
                self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            else:
                self.unsetCursor()
            self.update()
        # Mientras el mouse esté encima, el globo no se va.
        if self.state == "correctable" and self._editor is None:
            self._linger.start(self.LINGER_MS)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover != -1:
            self._hover = -1
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        idx = self._word_at(event.position())
        if idx < 0:
            super().mousePressEvent(event)
            return
        self._linger.stop()
        if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and self._sel:
            # Shift+clic estira lo ya marcado, para tramos largos donde
            # arrastrar con precisión es incómodo.
            self._sel = (min(self._sel[0], idx), max(self._sel[1], idx))
            self._anchor = None
            self.update()
            self._open_editor(*self._selection())
            return
        self._anchor = idx
        self._sel = (idx, idx)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._anchor is None:
            super().mouseReleaseEvent(event)
            return
        self._anchor = None
        if self._sel is None:
            return
        self._open_editor(*self._selection())

    def _selection(self):
        """(texto del tramo marcado, rectángulo para ubicar el editor)."""
        a, b = self._sel
        words = self._words[a:b + 1]
        text = " ".join(t for t, _ in words)
        rect = QRectF(words[0][1])
        for _, r in words[1:]:
            rect = rect.united(r)
        return text, rect

    def _open_editor(self, token, rect):
        self._close_editor()
        self._editor = _WordEditor(token, self._suggestions_for(token), self)
        self._editor.committed.connect(lambda right, w=token: self._commit(w, right))
        self._editor.closed.connect(self._editor_closed)
        # Centrado sobre la palabra y apoyado arriba del globo.
        gr = self.geometry()
        ew = self._editor.sizeHint().width()
        x = int(gr.x() + rect.center().x() - ew / 2)
        y = int(gr.y() - self._editor.sizeHint().height() - 8)
        screen = self._screen_rect()
        x = max(screen.left() + 8, min(x, screen.right() - ew - 8))
        self._editor.popup(QPoint(x, y))

    def _commit(self, wrong, right):
        right = (right or "").strip()
        if right and right != wrong:
            self.correction_made.emit(wrong, right)

    def _editor_closed(self):
        self._editor = None
        self._sel = None
        self._hover = -1
        if self.state == "correctable":
            # Dar un respiro para corregir otra palabra de la misma toma.
            self._linger.start(self.LINGER_MS)

    def _close_editor(self):
        if self._editor is not None:
            editor, self._editor = self._editor, None
            try:
                editor.closed.disconnect()
                editor.dismiss()
            except RuntimeError:
                pass        # ya se cerró y Qt lo borró (WA_DeleteOnClose)

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
        elif self.state == "correctable":
            # La toma terminó: un punto quieto y apagado. Sin animación, que
            # acá el globo ya no está trabajando, solo esperando.
            c = QColor(WF_ON_SURFACE_MUTED)
            c.setAlpha(110)
            painter.setBrush(c)
            painter.drawEllipse(QPointF(cx, cy), 3.5, 3.5)
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
            # Se dibuja línea por línea (y no con TextWordWrap) porque el corte
            # ya lo hizo _wrap para poder ubicar cada palabra. Cada línea se
            # dibuja entera, así el espaciado sigue siendo el de Qt.
            painter.setPen(WF_ON_SURFACE)
            line_h = fm.lineSpacing()
            for i, (tokens, _) in enumerate(self._lines):
                painter.drawText(
                    QRectF(self._text_rect.x(), self._text_rect.y() + i * line_h,
                           self._text_rect.width(), fm.height()),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                    " ".join(t for t, _, _ in tokens))

            # Subrayado coral bajo la palabra señalada: la misma señal de
            # "esto se puede tocar" que el acento del resto de la app.
            # Punteado tenue bajo lo que el diccionario no reconoce. Va ANTES
            # del subrayado coral para que, si señalás una de esas palabras,
            # gane el coral y no se superpongan.
            if self._flagged:
                pen = QPen(QColor(WF_ON_SURFACE_MUTED.red(), WF_ON_SURFACE_MUTED.green(),
                                  WF_ON_SURFACE_MUTED.blue(), 130))
                pen.setStyle(Qt.PenStyle.DotLine)
                pen.setWidthF(1.6)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                for i in sorted(i for i in self._flagged if i < len(self._words)):
                    r = self._words[i][1]
                    y = r.bottom()
                    painter.drawLine(QPointF(r.x(), y), QPointF(r.right(), y))

            marked = []
            if self._sel is not None:
                marked = self._words[self._sel[0]:self._sel[1] + 1]
            elif 0 <= self._hover < len(self._words):
                marked = [self._words[self._hover]]
            if marked:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(WF_ACCENT_RECORDING)
                # Un subrayado por renglón: si el tramo cruza de línea, se
                # dibujan dos, y cada uno cubre también los espacios de adentro.
                by_line = {}
                for _, r in marked:
                    by_line.setdefault(round(r.y(), 1), []).append(r)
                for rects in by_line.values():
                    x0 = min(r.x() for r in rects)
                    x1 = max(r.right() for r in rects)
                    y = rects[0].bottom() - 1.0
                    painter.drawRoundedRect(QRectF(x0, y, x1 - x0, 2.0), 1.0, 1.0)

            if self.state == "correctable" and self._show_hint:
                painter.setFont(self._hint_font)
                painter.setPen(QColor(WF_ON_SURFACE_MUTED.red(), WF_ON_SURFACE_MUTED.green(),
                                      WF_ON_SURFACE_MUTED.blue(), 150))
                painter.drawText(
                    QRectF(self._text_rect.x(), self._text_rect.bottom() + 4,
                           self._text_rect.width(),
                           QFontMetrics(self._hint_font).height()),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    self.HINT)
        else:
            painter.setPen(WF_ON_SURFACE_MUTED)
            painter.drawText(self._text_rect,
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             self._placeholder())
        painter.end()


class _WordEditor(QWidget):
    """Campo chico para escribir cómo se dice de verdad la palabra señalada.

    Es una ventana aparte del globo, y no un hijo suyo, por una razón: esta SÍ
    toma foco (el globo nunca lo hace). Al abrirse se anota qué ventana estaba
    activa y se lo devuelve al cerrarse, así el usuario vuelve a donde estaba
    escribiendo sin tener que hacer clic de nuevo.
    """

    committed = pyqtSignal(str)
    closed = pyqtSignal()

    PAD = 8
    MIN_W = 190

    def __init__(self, word, suggestions=None, parent=None):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._prev_hwnd = None
        self._done = False

        self.edit = QLineEdit(word, self)
        self.edit.setFont(QFont("Segoe UI", 11))
        self.edit.setStyleSheet(
            "QLineEdit {"
            f" background-color: rgb({WF_SURFACE_RAISED.red()}, {WF_SURFACE_RAISED.green()}, {WF_SURFACE_RAISED.blue()});"
            f" color: rgb({WF_ON_SURFACE.red()}, {WF_ON_SURFACE.green()}, {WF_ON_SURFACE.blue()});"
            " border: none; border-radius: 8px; padding: 7px 10px;"
            f" selection-background-color: rgb({WF_ACCENT_RECORDING.red()}, {WF_ACCENT_RECORDING.green()}, {WF_ACCENT_RECORDING.blue()});"
            " selection-color: rgb(18, 18, 22); }"
        )
        self.edit.returnPressed.connect(self._commit)
        self.edit.selectAll()

        w = max(self.MIN_W, self.edit.fontMetrics().horizontalAdvance(word) + 60)

        # Sugerencias del diccionario, en fila, para resolver de un clic. NO se
        # precargan en el campo: MEDIDO, la primera acierta a veces
        # (`superiferia`→periferia, `inkedin`→LinkedIn) pero otras no
        # (`farada`→fardada), y un prellenado malo se guarda sin que lo mires.
        self.chips = []
        x, row_y = self.PAD, self.PAD + 34 + 6
        for text in (suggestions or [])[:3]:
            chip = QPushButton(text, self)
            chip.setFont(QFont("Segoe UI", 9))
            chip.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            chip.setStyleSheet(
                "QPushButton {"
                f" background-color: rgb({WF_SURFACE_RAISED.red()}, {WF_SURFACE_RAISED.green()}, {WF_SURFACE_RAISED.blue()});"
                f" color: rgb({WF_ON_SURFACE.red()}, {WF_ON_SURFACE.green()}, {WF_ON_SURFACE.blue()});"
                " border: none; border-radius: 7px; padding: 4px 9px; }"
                "QPushButton:hover {"
                f" background-color: rgb({WF_ACCENT_RECORDING.red()}, {WF_ACCENT_RECORDING.green()}, {WF_ACCENT_RECORDING.blue()});"
                " color: rgb(18, 18, 22); }"
            )
            cw = chip.fontMetrics().horizontalAdvance(text) + 22
            chip.setGeometry(x, row_y, cw, 26)
            chip.clicked.connect(lambda _=False, t=text: self._pick(t))
            self.chips.append(chip)
            x += cw + 6

        if self.chips:
            w = max(w, x - 6 - self.PAD)
        self.edit.setGeometry(self.PAD, self.PAD, w, 34)
        h = 34 + (32 if self.chips else 0)
        self.resize(w + 2 * self.PAD, h + 2 * self.PAD)

    def sizeHint(self) -> QSize:
        return self.size()

    def popup(self, top_left: QPoint):
        self._prev_hwnd = _foreground_window()
        self.move(top_left)
        self.show()
        self.raise_()
        self.activateWindow()
        self.edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _pick(self, text):
        self.edit.setText(text)
        self._commit()

    def _commit(self):
        if self._done:
            return
        self._done = True
        self.committed.emit(self.edit.text())
        self._finish()

    def dismiss(self):
        """Cerrar sin guardar, desde afuera (empezó otra toma, por ejemplo)."""
        self._done = True
        self._finish()

    def _finish(self):
        _restore_foreground(self._prev_hwnd)
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._done = True
            self._finish()
            return
        super().keyPressEvent(event)

    def changeEvent(self, event):
        # Clic afuera o Alt+Tab: se cierra sin guardar, como cualquier popup.
        # Es WindowDeactivate y no focusOut porque el foco lo tiene el
        # QLineEdit de adentro, así que la ventana nunca lo pierde por sí sola.
        if event.type() == QEvent.Type.WindowDeactivate and not self._done:
            self._done = True
            self._finish()
        super().changeEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(WF_SURFACE)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(self.rect()), 12, 12)
        painter.end()
