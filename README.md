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

**Validado sobre audio real:** 12 notas de voz en castellano rioplatense (2,7s a
131s), cero degeneraciones, 207,3s -> 159,1s en total y 2,5-4x en las notas
cortas. Las mismas 12 notas con el recorte pero SIN `vad_filter` / SIN
`initial_prompt` / con `condition_on_previous_text` en su default dieron una
nota de 10,3s tardando **90,58s**: el recorte es seguro por la compañía que
tiene, no solo por el pad. Si lo llevás a otro proyecto, llevate las tres
opciones con él.

**En el streaming el mismo recorte ROMPE** y por eso está apagado ahí
(`--stream-short-window` para re-verificarlo): con `tiny`, una pasada sobre un
buffer de 3s pasó de 1,5s a 41,7s — sin bastante silencio atrás el modelo no
emite `<|endoftext|>` y genera hasta `max_length`. Una pasada de streaming es el
caso peor: buffer corto, prompt largo y `word_timestamps=True`.

## Los tres caminos del dictado

`bench_dictation.py` mide tres formas de llegar al texto. No compiten: cada una
gana en un caso distinto y el banco dice cuál.

| | cuándo decodifica | ESPERA (small, Pi 5) | WER | para qué sirve |
|---|---|---|---|---|
| **one-shot** (`transcribe_one_shot`) | todo al soltar | crece con lo que hablás: 2,6s (4,8s) / 23,7s (47s) | 0,0% | lo más simple; suficiente para dictados cortos |
| **segmentos** (`segment_asr.SegmentASR`) | cada frase apenas cierra por VAD, mientras seguís hablando | **plana: 2,7s (4,8s) / 2,8s (47s)** | 0,0% | el dictado de verdad |
| **streaming** (`streaming_core.OnlineASR`) | re-transcribe el buffer entero en cada pasada | 42s / 198s con `small`; solo `tiny` sigue el ritmo | 12-25% con `tiny` | mostrar texto ANTES de que termines la frase |

```bash
python benchmark/bench_dictation.py --live-mode segment --models small --threads 3
python benchmark/bench_dictation.py --live-mode segment --release-delay 0.8   # con el beat real de soltar la tecla
```

### Por qué segmentos y no streaming

Con la ventana del encoder ya recortada, el costo dejó de estar en el encoder y
pasó al **decoder**, que es autorregresivo y escala con las palabras (clip de
4,8s: encoder 0,53s, resto 2,08s). No queda nada para exprimir de una sola
pasada al final: la única forma de que la espera no crezca es decodificar
mientras el usuario habla. Y la CPU está libre igual — hoy el one-shot no hace
nada hasta que soltás.

El streaming LocalAgreement re-decodifica cada palabra muchas veces para poder
mostrarla antes de que la frase termine; en esta Pi `small` no le sigue el ritmo
ni de cerca. Por segmentos, cada muestra se decodifica **una sola vez**, cuando
la frase ya cerró, y se paga 0% de WER extra. Además un segmento de VAD termina
en silencio por construcción, que es justo la condición que le falta a una
pasada de streaming y que hacía degenerar el recorte de ventana.

Efecto colateral medido: la segmentación **arregla** el descarrilamiento de los
modelos chicos en audio largo, porque nunca ven más de una frase — `base` en el
clip de 47s pasó de 59,3% de WER (one-shot) a **6,2%** (segmentos), con la
espera en 0,82s. Es la opción si querés ~1s y podés pagar ese WER; `small`
queda en 0,0% con ~2,8s.

**El piso, sin vueltas:** la espera al soltar es la decodificación de la última
frase menos lo que se alcanzó a solapar. Con `small` en esta Pi eso son ~2,6s
para una frase de 5s, y ningún cambio de arquitectura lo baja — para eso hace
falta un modelo más rápido.

## Texto en vivo: Moonshine y los dos modos del hotkey

Whisper no puede mostrar texto mientras hablás en una CPU normal (arriba está
la medición). [Moonshine v2](https://github.com/moonshine-ai/moonshine) sí: es
un modelo pensado para streaming, procesa el audio una sola vez y emite
parciales cada ~0,5s. `moonshine_engine.py` lo envuelve con la misma interfaz
que `SegmentASR`; la app elige con `config["engine"]`.

MEDIDO en este Ryzen (`spike_moonshine.py`, clips del banco a tiempo real):

| motor | clip | espera al soltar | primer texto visible | WER |
|---|---|---|---|---|
| Moonshine small (es) | es-AR 47s | **0,05s** | 1,7s | 0,9% (una tilde) |
| Whisper small, segmentos | es-AR 47s | 0,56s | 6,4s (primera frase cerrada) | 0,0% |

Moonshine casi no pone puntuación en español; Whisper sí. Los modelos de
Moonshine que no son inglés están bajo la *Moonshine Community License* (no
comercial por encima de 1M USD anuales); el código es MIT.

**Escribir en la app destino mientras hablás tiene una condición física**,
medida en Chromium (WhatsApp, Claude, ChatGPT, VS Code): con Ctrl apretado,
toda letra inyectada se interpreta como atajo y se descarta. Con Win apretado,
un Ctrl+V es Win+Ctrl+V. Por eso hay dos modos (`config["hotkey_mode"]`,
también en Configuración):

| modo | cómo se usa | qué ves mientras hablás | cuándo se escribe en el destino |
|---|---|---|---|
| `hold` (default) | mantenés Ctrl+Win | el globo de dictado abajo al centro, con el texto en vivo | todo al soltar (~0,1s con Moonshine) |
| `toggle` | pulsás Ctrl+Win para empezar y otra vez para terminar | lo mismo | **cada frase apenas cierra**, mientras seguís hablando |

El globo (`dictation_bubble.py`) aparece cuando empieza la toma, crece con el
texto y se desvanece al terminar. No acepta foco ni clics: lo que se escribe
sigue yendo a la ventana que tenías activa.

En los dos modos los parciales (que Moonshine reescribe) solo se muestran; al
destino van únicamente frases cerradas, que son definitivas. `text_injector.py`
espera a que no haya modificadores apretados antes de escribir.

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
