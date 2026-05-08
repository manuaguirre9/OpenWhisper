# Script para compilar OpenWhisper como un ejecutable (.exe)
# --noconsole: Oculta la ventana de comandos negra de fondo
# --onedir: Crea una carpeta con los archivos en lugar de un unico .exe lento
# --collect-data y --collect-all: Fuerza a que se incluyan las librerias pesadas de IA

pyinstaller --noconsole --onedir `
    --name OpenWhisper `
    --collect-data faster_whisper `
    --collect-all ctranslate2 `
    app.py

Write-Host "¡Compilación completada! El ejecutable está en la carpeta 'dist\OpenWhisper'."
