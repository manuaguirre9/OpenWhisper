"""
Spike: Moonshine v2 streaming vs. the benchmark clips.

Pregunta que responde: ¿Moonshine puede mostrar texto MIENTRAS el usuario
habla, en esta CPU, con calidad comparable al `small` de Whisper?

Alimenta cada clip del banco en tiempo real (bloques de 100ms, como el mic),
registra cada parcial que emite el stream y mide:

  ESPERA      segundos entre el último bloque (soltar la tecla) y el texto final
  1er parcial segundos desde que empieza el audio hasta la primera palabra visible
  atraso      promedio de (instante en que apareció un parcial) - (posición del audio)
              → cuánto va "detrás" de lo que hablás el texto en pantalla
  WER         contra el mismo ground truth que usa bench_dictation.py

    python spike_moonshine.py                       # todos los clips, arch por defecto por idioma
    python spike_moonshine.py --arch tiny,small     # barrer tamaños
    python spike_moonshine.py --clips openslr61-es-ar-largo
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmark"))
from bench_dictation import discover_clips, load_clip, word_error_rate  # noqa: E402

from moonshine_voice import (  # noqa: E402
    ModelArch,
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
)

SAMPLE_RATE = 16000
FEED_BLOCK_S = 0.1

ARCHS = {
    "tiny": ModelArch.TINY_STREAMING,
    "small": ModelArch.SMALL_STREAMING,
    "medium": ModelArch.MEDIUM_STREAMING,
    "base": ModelArch.BASE,
}


class Recorder(TranscriptEventListener):
    """Guarda cada evento con el instante de pared en que llegó."""

    def __init__(self, t0):
        self.t0 = t0
        self.partials = []      # (t_wall, line_id, text)
        self.completed = {}     # line_id -> (t_wall, text, latency_ms)
        self.errors = []

    def on_line_text_changed(self, event):
        self.partials.append((time.perf_counter() - self.t0, event.line.line_id, event.line.text))

    def on_line_completed(self, event):
        self.completed[event.line.line_id] = (
            time.perf_counter() - self.t0, event.line.text,
            event.line.last_transcription_latency_ms,
        )

    def on_error(self, event):
        self.errors.append(repr(event))


def run_clip(transcriber, clip, update_interval):
    audio = clip["audio"]
    block = int(FEED_BLOCK_S * SAMPLE_RATE)
    blocks = [audio[i:i + block] for i in range(0, len(audio), block)]

    stream = transcriber.create_stream(update_interval=update_interval)
    t0 = time.perf_counter()
    rec = Recorder(t0)
    stream.add_listener(rec)
    stream.start()

    for i, chunk in enumerate(blocks):
        target = t0 + (i + 1) * FEED_BLOCK_S
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        stream.add_audio(chunk.astype(np.float32).tolist(), SAMPLE_RATE)
    t_release = time.perf_counter()

    stream.stop()
    t_ready = time.perf_counter()

    # Texto final: las líneas completas en orden de aparición. Si alguna no
    # cerró, tomamos su último parcial.
    final_by_line = {}
    for t_wall, line_id, text in rec.partials:
        final_by_line[line_id] = text
    for line_id, (_t, text, _ms) in rec.completed.items():
        final_by_line[line_id] = text
    text = " ".join(t.strip() for t in final_by_line.values() if t.strip())

    # atraso de cada parcial respecto del audio que YA había entrado en ese momento.
    # El audio avanza a tiempo real desde t0, así que posición ≈ t_wall (acotada al clip).
    # Lo que interesa es cuánto tarda en aparecer texto nuevo: medimos el delta entre
    # parciales consecutivos que cambian el texto visible.
    first_partial = next((t for t, _l, txt in rec.partials if txt.strip()), float("nan"))
    gaps = [b[0] - a[0] for a, b in zip(rec.partials, rec.partials[1:])]
    latencies = [ms for (_t, _txt, ms) in rec.completed.values() if ms]

    stream.close()
    return {
        "clip": clip["stem"],
        "duration": clip["duration"],
        "wait": t_ready - t_release,
        "first_partial": first_partial,
        "partials": len(rec.partials),
        "partial_gap_avg": statistics.fmean(gaps) if gaps else float("nan"),
        "lines": len(rec.completed),
        "line_latency_ms": statistics.fmean(latencies) if latencies else float("nan"),
        "wer": word_error_rate(clip["reference"], text),
        "text": text,
        "reference": clip["reference"],
        "errors": rec.errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", default="")
    parser.add_argument("--arch", default="", help="tiny,small,medium,base (default: el de la lib por idioma)")
    parser.add_argument("--update-interval", type=float, default=0.5)
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()

    stems = [s for s in args.clips.split(",") if s.strip()] or discover_clips()
    clips = [load_clip(s) for s in stems]
    archs = [a for a in args.arch.split(",") if a.strip()] or [None]

    rows = []
    for arch_name in archs:
        arch = ARCHS[arch_name] if arch_name else None
        by_lang = {}
        for clip in clips:
            lang = clip["language"] or "es"
            if lang not in by_lang:
                print(f"Cargando Moonshine {arch_name or 'default'} para '{lang}'…", flush=True)
                t_load = time.perf_counter()
                try:
                    path, resolved = get_model_for_language(lang, arch)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ✗ no hay {arch_name} para {lang}: {exc}")
                    by_lang[lang] = None
                    continue
                by_lang[lang] = (Transcriber(model_path=path, model_arch=resolved,
                                             update_interval=args.update_interval), resolved.name)
                print(f"  {resolved.name} listo en {time.perf_counter() - t_load:.1f}s", flush=True)
            if by_lang[lang] is None:
                continue
            transcriber, resolved_name = by_lang[lang]
            row = run_clip(transcriber, clip, args.update_interval)
            row["arch"] = resolved_name
            rows.append(row)
            print(f"  {row['clip']:<24} espera {row['wait']:5.2f}s  1er parcial {row['first_partial']:4.2f}s  "
                  f"WER {row['wer'] * 100:5.1f}%  parciales {row['partials']:3d}", flush=True)
            if args.show_text or row["wer"] > 0:
                print(f"     ref : {row['reference']}")
                print(f"     moon: {row['text']}")
            if row["errors"]:
                print(f"     errores: {row['errors'][:2]}")

    print("\n" + "=" * 100)
    print(f"{'arch':<17}{'clip':<24}{'dur':>6}{'ESPERA':>8}{'1er parc':>9}{'gap parc':>9}{'lat línea':>10}{'WER':>7}")
    print("-" * 100)
    for r in rows:
        print(f"{r['arch']:<17}{r['clip']:<24}{r['duration']:>5.1f}s{r['wait']:>7.2f}s"
              f"{r['first_partial']:>8.2f}s{r['partial_gap_avg']:>8.2f}s{r['line_latency_ms']:>8.0f}ms"
              f"{r['wer'] * 100:>6.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
