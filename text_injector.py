"""
Escribir texto en la app que tiene el foco. Sin Qt, sin modelos.

Dos caminos:
  type_text(text)   SendInput con KEYEVENTF_UNICODE. Sin portapapeles, ~3ms por
                    50 caracteres. Es el camino para las frases que llegan EN
                    VIVO mientras el usuario sigue dictando.
  paste_text(text)  Portapapeles + Ctrl+V, restaurando lo que había. Es el
                    camino viejo de la app para el texto final: funciona en
                    cualquier app, incluso las que no digieren bien VK_PACKET.

Y una regla que los dos respetan: NUNCA inyectar mientras el usuario tiene un
modificador apretado. MEDIDO en Chromium (WhatsApp, Claude, ChatGPT, VS Code):
con Ctrl físicamente apretado, toda letra inyectada se interpreta como atajo y
se descarta — solo entraron «¿éñú». Y con Win apretado, un Ctrl+V se convierte
en Win+Ctrl+V (el menú de salida de sonido de Windows 11). Por eso existe
`wait_modifiers_released`: el que inyecta espera a que el usuario suelte.
Consecuencia de diseño: escribir en vivo solo es posible si el hotkey NO se
mantiene apretado durante el dictado (modo "toggle" en app.py).
"""
import sys
import time

WIN32 = sys.platform == "win32"

if WIN32:
    import ctypes
    import ctypes.wintypes as _w

    _user32 = ctypes.windll.user32
    _INPUT_KEYBOARD = 1
    _KEYEVENTF_KEYUP = 0x0002
    _KEYEVENTF_UNICODE = 0x0004
    _MODIFIER_VKS = (0x11, 0x12, 0x10, 0x5B, 0x5C)   # Ctrl, Alt, Shift, LWin, RWin

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", _w.WORD), ("wScan", _w.WORD), ("dwFlags", _w.DWORD),
                    ("time", _w.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 32)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", _w.DWORD), ("u", _INPUTUNION)]

    def modifiers_down() -> bool:
        return any(_user32.GetAsyncKeyState(vk) & 0x8000 for vk in _MODIFIER_VKS)

    def _unicode_inputs(text: str):
        data = text.encode("utf-16-le")
        out = []
        for i in range(0, len(data), 2):
            code = int.from_bytes(data[i:i + 2], "little")
            for up in (False, True):
                inp = _INPUT(type=_INPUT_KEYBOARD)
                inp.u.ki = _KEYBDINPUT(0, code, _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if up else 0), 0, None)
                out.append(inp)
        return out

    def type_text(text: str, chunk_chars: int = 48) -> None:
        """Tipea `text` en la ventana con foco. En tandas chicas: algunas apps
        pierden caracteres si les llegan cientos de eventos de golpe."""
        if not text:
            return
        for start in range(0, len(text), chunk_chars):
            inputs = _unicode_inputs(text[start:start + chunk_chars])
            arr = (_INPUT * len(inputs))(*inputs)
            sent = _user32.SendInput(len(inputs), arr, ctypes.sizeof(_INPUT))
            if sent != len(inputs):
                raise OSError(f"SendInput entregó {sent}/{len(inputs)} eventos")
            time.sleep(0.005)

else:  # pragma: no cover - la app es Windows-only, esto es para importar en CI
    def modifiers_down() -> bool:
        return False

    def type_text(text: str, chunk_chars: int = 48) -> None:
        import pyautogui
        pyautogui.write(text)


def wait_modifiers_released(timeout: float = 3.0, poll: float = 0.02,
                            _down=None) -> bool:
    """Espera a que no haya Ctrl/Alt/Shift/Win apretados. False si se agotó."""
    is_down = _down or modifiers_down
    deadline = time.monotonic() + timeout
    while is_down():
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)
    return True


def paste_text(text: str) -> None:
    """El camino viejo: portapapeles + Ctrl+V, restaurando lo que había."""
    if not text:
        return
    import pyautogui
    import pyperclip

    # paste() puede fallar si el portapapeles tiene algo que no es texto (una
    # imagen copiada de un navegador). En ese caso no restauramos nada.
    original = None
    try:
        original = pyperclip.paste()
    except Exception as exc:  # noqa: BLE001
        print(f"[Injector] No pude leer el portapapeles original: {exc}")
    try:
        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.2)
    finally:
        if original is not None:
            try:
                pyperclip.copy(original)
            except Exception:
                pass
