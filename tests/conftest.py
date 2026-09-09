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
