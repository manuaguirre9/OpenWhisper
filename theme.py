"""
Shared visual theme for OpenWhisper's secondary windows (Settings, Batch).
Same palette as the floating overlay so the whole product feels coherent.
"""

import sys


# Single stylesheet used by every "normal" window in the app.
WINDOW_QSS = """
QMainWindow, QDialog, QWidget#root {
    background-color: rgb(14, 14, 18);
}
QLabel {
    color: rgb(232, 232, 240);
    font-family: "Segoe UI";
}
QLabel#title {
    font-size: 14pt;
    font-weight: 600;
}
QLabel#subtle {
    color: rgb(170, 170, 182);
    font-size: 9pt;
}
QLabel#section {
    color: rgb(180, 180, 190);
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1px;
}

QPushButton {
    background-color: rgb(32, 32, 40);
    color: rgb(232, 232, 240);
    border: none;
    border-radius: 10px;
    padding: 9px 16px;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QPushButton:hover {
    background-color: rgb(46, 46, 56);
}
QPushButton:disabled {
    color: rgb(110, 110, 118);
    background-color: rgb(26, 26, 32);
}
QPushButton#primary {
    background-color: rgb(255, 112, 122);
    color: rgb(18, 18, 22);
    font-weight: 600;
}
QPushButton#primary:hover {
    background-color: rgb(255, 132, 142);
}
QPushButton#primary:disabled {
    background-color: rgb(70, 40, 44);
    color: rgb(150, 110, 114);
}

QPlainTextEdit, QTextEdit {
    background-color: rgb(22, 22, 28);
    color: rgb(232, 232, 240);
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-family: "Segoe UI";
    font-size: 10pt;
    selection-background-color: rgb(255, 112, 122);
    selection-color: rgb(18, 18, 22);
}

QComboBox, QSpinBox, QLineEdit {
    background-color: rgb(28, 28, 36);
    color: rgb(232, 232, 240);
    border: none;
    border-radius: 8px;
    padding: 7px 10px;
    font-family: "Segoe UI";
    font-size: 10pt;
    min-height: 22px;
}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover {
    background-color: rgb(36, 36, 46);
}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {
    background-color: rgb(40, 40, 50);
}
QComboBox:disabled, QSpinBox:disabled, QLineEdit:disabled {
    background-color: rgb(22, 22, 28);
    color: rgb(95, 95, 105);
}

QCheckBox {
    color: rgb(232, 232, 240);
    font-family: "Segoe UI";
    font-size: 10pt;
    spacing: 10px;
    padding: 2px 0;
}
QCheckBox:disabled {
    color: rgb(110, 110, 120);
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    background-color: rgb(28, 28, 36);
    border: 1.5px solid rgb(90, 90, 102);
}
QCheckBox::indicator:hover {
    border-color: rgb(180, 160, 255);
}
QCheckBox::indicator:checked {
    background-color: rgb(255, 112, 122);
    border-color: rgb(255, 112, 122);
}
QCheckBox::indicator:checked:hover {
    background-color: rgb(255, 132, 142);
    border-color: rgb(255, 132, 142);
}
QCheckBox::indicator:disabled {
    background-color: rgb(22, 22, 28);
    border-color: rgb(50, 50, 58);
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid rgb(170, 170, 182);
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: rgb(28, 28, 36);
    color: rgb(232, 232, 240);
    border: none;
    border-radius: 8px;
    padding: 4px;
    outline: 0;
    selection-background-color: rgb(46, 46, 56);
}
QSpinBox::up-button, QSpinBox::down-button {
    background: transparent;
    width: 16px;
    border: none;
}
QSpinBox::up-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 4px solid rgb(170, 170, 182);
}
QSpinBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid rgb(170, 170, 182);
}

QProgressBar {
    background-color: rgb(28, 28, 36);
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: rgb(170, 170, 182);
    font-size: 8pt;
}
QProgressBar::chunk {
    background-color: rgb(180, 160, 255);
    border-radius: 4px;
}

QListWidget {
    background-color: rgb(22, 22, 28);
    color: rgb(232, 232, 240);
    border: none;
    border-radius: 12px;
    padding: 6px;
    font-family: "Segoe UI";
    font-size: 10pt;
    outline: 0;
}
QListWidget::item {
    background-color: transparent;
    padding: 8px 10px;
    border-radius: 8px;
    margin: 2px 0;
}
QListWidget::item:hover {
    background-color: rgb(32, 32, 40);
}
QListWidget::item:selected {
    background-color: rgb(46, 46, 56);
    color: rgb(232, 232, 240);
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: rgb(60, 60, 70);
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: rgb(80, 80, 90);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QMessageBox {
    background-color: rgb(14, 14, 18);
}
QMessageBox QLabel {
    color: rgb(232, 232, 240);
}
"""


def apply_windows_dark_titlebar(widget):
    """Render the native Windows 11 title bar in dark mode for `widget`."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        value = ctypes.c_int(1)
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 on Win11/recent Win10
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
        )
    except Exception:
        pass
