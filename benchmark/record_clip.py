"""
Graba un clip de referencia para el banco de pruebas.

El banco necesita audio FIJO con transcripción correcta escrita a mano. El clip
de arranque (librispeech-en-3081) es en inglés; para medir dictado real en
castellano hacen falta clips propios, dichos por vos, con el micrófono y el
ruido de fondo que usás todos los días.

    python benchmark/record_clip.py mi-clip-corto
    python benchmark/record_clip.py mi-clip-largo --seconds 45

Graba hasta que apretás Enter (o hasta --seconds), guarda
benchmark/resources/<nombre>.wav y deja un <nombre>.json con el texto VACÍO
para que lo completes a mano. Ese texto es la vara: si está mal, el WER miente.

Elegí clips que se parezcan a cómo dictás de verdad — incluí jerga, nombres
propios y números, que es justo donde Whisper falla y donde el vocabulario
custom de la app tiene que notarse.
"""
import argparse
import json
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streaming_core import SAMPLE_RATE  # noqa: E402

RESOURCES = Path(__file__).resolve().parent / "resources"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="nombre del clip (sin extensión)")
    parser.add_argument("--seconds", type=float, default=0.0, help="corte automático (0 = manual)")
    parser.add_argument("--language", default="es")
    parser.add_argument("--device", default="", help="id de dispositivo de entrada")
    args = parser.parse_args()

    RESOURCES.mkdir(parents=True, exist_ok=True)
    wav_path = RESOURCES / f"{args.name}.wav"
    json_path = RESOURCES / f"{args.name}.json"
    if wav_path.exists():
        raise SystemExit(f"{wav_path.name} ya existe. Elegí otro nombre o borralo.")

    device = int(args.device) if args.device else None
    blocks = []
    stop = threading.Event()

    def callback(indata, frames, t, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        blocks.append(indata.copy().flatten())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        device=device, callback=callback):
        print(f"🔴 Grabando '{args.name}'… ", end="", flush=True)
        if args.seconds > 0:
            print(f"({args.seconds:.0f}s)")
            stop.wait(args.seconds)
        else:
            print("Enter para cortar.")
            input()

    if not blocks:
        raise SystemExit("No se capturó audio.")

    audio = np.concatenate(blocks)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    json_path.write_text(json.dumps(
        {"language": args.language, "text": "", "notes": "escribí acá lo que dijiste, palabra por palabra"},
        indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✓ {wav_path.name}  ({len(audio) / SAMPLE_RATE:.1f}s)")
    print(f"→ Ahora completá el campo \"text\" de {json_path.name} con lo que dijiste.")
    print(f"  Después: python benchmark/bench_dictation.py --clips {args.name}")


if __name__ == "__main__":
    main()
