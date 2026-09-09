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

## Benchmark: streaming vs. one-shot

The goal is Wispr-Flow-like responsiveness, and the number that decides it is
**ESPERA** — the seconds between releasing the hotkey and having the text. Today
the whole decode happens after you let go, so ESPERA grows with how long you
spoke. Streaming (`streaming_core.py`, LocalAgreement-2) decodes while you talk,
so it should stay flat.

`benchmark/bench_dictation.py` measures that over a fixed audio file, feeding it
in real time exactly like a microphone would — no mic, no human, same number
every run:

```bash
pip install -r requirements-dev.txt

python benchmark/bench_dictation.py                       # all clips
python benchmark/bench_dictation.py --models base,small   # sweep model sizes
python benchmark/bench_dictation.py --beams 1,5 --repeat 3
```

It reports ESPERA for both paths, the speedup between them, WER against a
hand-written ground truth (so a faster config that gets dumber is visible), and
whether a streaming pass keeps up with speech.

A 10s English clip ships so it runs out of the box. **Record your own Spanish
clips before deciding anything** — see `benchmark/resources/README.md`:

```bash
python benchmark/record_clip.py dictado-corto
python benchmark/record_clip.py dictado-largo --seconds 45
```

Unit tests for the streaming logic and the WER: `pytest`.

### Medido: la ventana del encoder (Raspberry Pi 5, `small` int8, 3 hilos)

Whisper padea **siempre** a 30 segundos y el encoder corre sobre esa ventana
entera, hables 2 segundos o 25. En un dictado de 4,8s eso son 4,4s de los 7,2s
de ESPERA gastados en encodear silencio. `streaming_core.short_window()` padea
solo lo que hace falta, y `FULL_BEAM` bajó de 5 a 1 (que es lo que el dictado ya
usaba de verdad — `config["beam_size"]`, default 1):

| clip | antes | ahora | gana | WER |
|---|---|---|---|---|
| es-AR 4,8s | 8,04s | **2,62s** | 3,1x | 0,0% en los dos |
| en 10,5s | 9,61s | **4,33s** | 2,2x | 0,0% en los dos |
| es-AR 47,1s | 34,09s | **23,71s** | 1,4x | 0,0% en los dos |

El texto sale idéntico. Reproducible con:

```bash
python benchmark/bench_dictation.py --models small --threads 3 --skip-streaming --repeat 2 --no-short-window --full-beam 5   # antes
python benchmark/bench_dictation.py --models small --threads 3 --skip-streaming --repeat 2                                   # ahora
```

**En el streaming el mismo recorte ROMPE** y por eso está apagado ahí
(`--stream-short-window` para re-verificarlo): con `tiny`, una pasada sobre un
buffer de 3s pasó de 1,5s a 41,7s — sin bastante silencio atrás el modelo no
emite `<|endoftext|>` y genera hasta `max_length`. Una pasada de streaming es el
caso peor: buffer corto, prompt largo y `word_timestamps=True`.

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
