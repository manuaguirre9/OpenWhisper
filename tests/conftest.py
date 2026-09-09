"""
Aislamiento del directorio de configuración para toda la suite.

config_manager hace os.makedirs(CONFIG_DIR) a nivel de módulo, así que el
solo hecho de importarlo — cosa que hace streaming_prototype, y por lo tanto
cualquier test que lo toque — crea %APPDATA%/OpenWhisper (o ~/.openwhisper
fuera de Windows) en la máquina real. Redirigimos APPDATA a un temporal
ANTES de que se importe cualquier módulo del proyecto: conftest.py lo carga
pytest primero.
"""
import os
import tempfile

os.environ["APPDATA"] = tempfile.mkdtemp(prefix="openwhisper-tests-")

# test_session_threading instala un stub de `faster_whisper` con setdefault para
# poder correr sin la biblioteca. Si está instalada de verdad, la importamos ACÁ
# —conftest corre primero— así el setdefault no gana y los tests del recorte de
# ventana pueden verificar el parche contra el símbolo real. Sin esto el stub se
# queda pegado para toda la suite y esos tests fallan solo cuando corren juntos.
try:  # pragma: no cover - depende de la máquina
    import faster_whisper.transcribe  # noqa: F401
except Exception:
    pass
