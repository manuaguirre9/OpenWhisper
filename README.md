# OpenWhisper Dictation

OpenWhisper is a privacy-first, local-only push-to-talk dictation tool for Windows. It uses OpenAI's Whisper model (via `faster-whisper`) to provide blazing-fast, system-wide speech-to-text integration.

## Features
- **Local & Private:** Everything runs on your machine. No audio is ever sent to the cloud.
- **System-Wide Push-To-Talk:** Just hold `Ctrl + Windows`, speak, and release. The text is instantly injected into whatever application you are using.
- **Hardware Acceleration:** Automatically uses NVIDIA/AMD GPUs if available, falling back to highly optimized CPU execution.
- **Audio Ducking:** Automatically lowers background music/volume while you are recording so the AI can hear you clearly.
- **Unobtrusive UI:** A transparent, click-through, draggable widget shows you the current state (Ready, Recording, Processing).
- **Keep-Alive:** The AI model is kept "warm" in RAM so it responds instantly even after hours of inactivity.

## Installation (From Source)
1. Clone the repository:
   ```bash
   git clone https://github.com/manuaguirre9/OpenWhisper.git
   cd OpenWhisper
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```

## Configuration
When running, a red microphone icon will appear in your Windows System Tray (next to the clock). Right-click it and select "Configuración" to access:
- **Micrófono:** Select your preferred input device.
- **Idioma:** Force a specific language (e.g., Spanish) for better accuracy on short phrases, or use Auto-detect.
- **Modelo:** Choose the Whisper model size (`tiny`, `base`, `small`, `medium`). Larger models are more accurate but slower.
- **Ducking:** Select how much the system volume should drop while holding the push-to-talk hotkey.

## Building the Executable (.exe)
If you want to create a standalone executable that runs without installing Python:
1. Ensure `pyinstaller` is installed: `pip install pyinstaller`.
2. Run the build script (PowerShell):
   ```powershell
   .\build_exe.ps1
   ```
3. The standalone app will be generated in `dist/OpenWhisper/`. You can copy this folder to any Windows machine.

## How it Works
OpenWhisper listens for the global hotkey using `pynput`. When triggered, it captures audio using `sounddevice`, applies audio ducking via `pycaw`, and passes the audio to `faster-whisper`. The transcribed text is then instantly pasted into the active window using `pyperclip` and `pyautogui`.

## License
MIT
