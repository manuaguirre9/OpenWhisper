"""
Log a archivo para el exe (que no tiene consola): todo lo que la app imprime
va también a %APPDATA%/OpenWhisper/openwhisper.log con hora, para poder
diagnosticar una toma que anduvo mal sin tener que reproducirla.

El archivo se trunca al arrancar si pasa de 2 MB.
"""
import os
import sys
import time

from config_manager import CONFIG_DIR

LOG_PATH = os.path.join(CONFIG_DIR, "openwhisper.log")
MAX_BYTES = 2 * 1024 * 1024


class _Tee:
    """Escribe en el archivo (con hora al inicio de cada línea) y, si hay
    consola, también en ella."""

    def __init__(self, path, console):
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self._console = console
        self._at_line_start = True

    def write(self, text):
        if not text:
            return
        stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d} "
        out = []
        for piece in text.splitlines(keepends=True):
            if self._at_line_start:
                out.append(stamp)
            out.append(piece)
            self._at_line_start = piece.endswith("\n")
        try:
            self._file.write("".join(out))
        except Exception:
            pass
        if self._console is not None:
            try:
                self._console.write(text)
            except Exception:
                pass

    def flush(self):
        try:
            self._file.flush()
        except Exception:
            pass
        if self._console is not None:
            try:
                self._console.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        """El descriptor del archivo de log.

        Lo necesita faulthandler, que escribe la pila de una caída dura desde C
        y por lo tanto no puede pasar por write(): quiere un fd de verdad.
        Devolver el del log es justo lo que queremos, así el volcado termina
        en el mismo lugar que todo lo demás."""
        return self._file.fileno()


def install():
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > MAX_BYTES:
            os.remove(LOG_PATH)
    except Exception:
        pass
    try:
        console_out = sys.stdout if sys.stdout is not None else None
        console_err = sys.stderr if sys.stderr is not None else None
        sys.stdout = _Tee(LOG_PATH, console_out)
        sys.stderr = _Tee(LOG_PATH, console_err)
        print(f"===== OpenWhisper arranca (pid {os.getpid()}) =====")
    except Exception as exc:  # noqa: BLE001 - sin log se corre igual
        print(f"[log] No pude abrir {LOG_PATH}: {exc}")
