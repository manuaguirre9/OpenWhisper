"""
Reproducible dictation benchmark — streaming vs. one-shot, over a fixed file.

WHY THIS EXISTS
---------------
streaming_prototype.py can only be judged by holding F8 with a microphone and
eyeballing the result. That is not a measurement: the audio changes every run,
there is no ground truth, and "¿se siente más rápido?" is not a number.

This runs the SAME streaming code (streaming_core.py) over a fixed audio file,
feeding it in real time exactly like a microphone would, and reports the one
metric that decides whether OpenWhisper feels as fast as Wispr Flow:

    ESPERA = seconds between releasing the key and having the final text.

That is what the user actually experiences. Everything else (pass time, commit
lag, WER) explains that number or guards its quality.

  - one-shot (what app.py does today): the whole decode happens AFTER release,
    so ESPERA grows with how long you spoke.
  - streaming: almost everything is already decoded while you speak, so ESPERA
    is roughly one tail decode and should stay flat as the clip gets longer.

Nothing here touches app.py.

USAGE
-----
    python benchmark/bench_dictation.py                       # all clips, model from config
    python benchmark/bench_dictation.py --models base,small   # sweep model sizes
    python benchmark/bench_dictation.py --clips mi-clip       # one clip
    python benchmark/bench_dictation.py --beams 1,5 --repeat 3
    python benchmark/bench_dictation.py --no-realtime         # throughput only, ignores ESPERA
    python benchmark/bench_dictation.py --json out.json       # machine-readable

Add your own clips with benchmark/record_clip.py (see benchmark/resources/README.md).
"""
import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming_core import (  # noqa: E402
    SAMPLE_RATE,
    MIN_CHUNK_S,
    STREAM_BEAM,
    FULL_BEAM,
    OnlineASR,
    transcribe_one_shot,
)

RESOURCES = Path(__file__).resolve().parent / "resources"
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4")
FEED_BLOCK_S = 0.1        # mic-sized blocks; PortAudio delivers ~this often


# ---------------------------------------------------------------- fixtures --

def load_audio(path: Path) -> np.ndarray:
    """Return mono float32 @16kHz. Uses the decoder faster-whisper already ships."""
    try:
        from faster_whisper.audio import decode_audio
        return np.asarray(decode_audio(str(path), sampling_rate=SAMPLE_RATE),
                          dtype=np.float32)
    except Exception:
        pass
    # Fallback for plain 16-bit PCM WAV without pulling in av.
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path.name}: necesito WAV mono 16-bit @16kHz (o instalá faster-whisper "
                f"para decodificar cualquier formato). Convertilo con:\n"
                f"  ffmpeg -i {path.name} -ac 1 -ar 16000 -sample_fmt s16 salida.wav"
            )
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def load_clip(stem: str) -> dict:
    """Load one fixture: audio + ground truth."""
    audio_path = next(
        (RESOURCES / f"{stem}{ext}" for ext in AUDIO_EXTS if (RESOURCES / f"{stem}{ext}").exists()),
        None,
    )
    if audio_path is None:
        raise SystemExit(f"No encontré audio para el clip '{stem}' en {RESOURCES}")

    meta = {"language": None, "text": None}
    json_path = RESOURCES / f"{stem}.json"
    txt_path = RESOURCES / f"{stem}.txt"
    if json_path.exists():
        meta.update(json.loads(json_path.read_text(encoding="utf-8")))
    elif txt_path.exists():
        meta["text"] = txt_path.read_text(encoding="utf-8").strip()
    else:
        raise SystemExit(
            f"El clip '{stem}' no tiene ground truth. Creá {stem}.json "
            f'({{"language": "es", "text": "lo que se dice"}}) o {stem}.txt'
        )

    if not meta.get("text"):
        raise SystemExit(f"El ground truth de '{stem}' está vacío.")

    audio = load_audio(audio_path)
    return {
        "stem": stem,
        "audio": audio,
        "duration": len(audio) / SAMPLE_RATE,
        "language": meta.get("language"),
        "reference": meta["text"],
    }


def discover_clips() -> list:
    stems = set()
    for path in RESOURCES.glob("*"):
        if path.suffix.lower() in AUDIO_EXTS:
            stems.add(path.stem)
    return sorted(stems)


# --------------------------------------------------------------------- WER --

def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Accents/ñ survive."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over words / len(reference). Same definition jiwer uses."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0

    prev = list(range(len(hyp) + 1))
    for i, r_word in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h_word in enumerate(hyp, start=1):
            cur[j] = min(
                prev[j] + 1,                              # deletion
                cur[j - 1] + 1,                           # insertion
                prev[j - 1] + (r_word != h_word),         # substitution
            )
        prev = cur
    return prev[len(hyp)] / len(ref)


# ------------------------------------------------------------------ passes --

def run_streaming(model, clip: dict, realtime: bool, min_chunk_s: float,
                  beam_size: int) -> dict:
    """
    Feed the clip through OnlineASR the way a microphone would, then measure
    the wait between the last audio block (= key release) and the final text.
    """
    audio = clip["audio"]
    language = clip["language"]
    errors = []
    online = OnlineASR(model, language, beam_size=beam_size,
                       on_error=errors.append)

    audio_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
    block = int(FEED_BLOCK_S * SAMPLE_RATE)
    blocks = [audio[i:i + block] for i in range(0, len(audio), block)]

    state = {}

    def feeder():
        t0 = time.perf_counter()
        for i, chunk in enumerate(blocks):
            if realtime:
                # Pace to wall clock so decoding competes with arriving audio,
                # exactly as it does live. Without this the loop chews through
                # the file instantly and ESPERA is meaningless.
                target = t0 + (i + 1) * FEED_BLOCK_S
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            audio_q.put(chunk)
        state["t_release"] = time.perf_counter()   # the user lets go of the key
        audio_q.put(None)

    thread = threading.Thread(target=feeder, daemon=True)
    t_start = time.perf_counter()
    thread.start()

    pass_times, commit_lag = [], []
    since_last, done = 0, False
    while not done:
        try:
            chunk = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if chunk is None:
            done = True
        else:
            online.insert_audio(chunk)
            since_last += len(chunk)
        if since_last >= min_chunk_s * SAMPLE_RATE or (done and since_last):
            since_last = 0
            t_pass = time.perf_counter()
            committed = online.process_iter()
            pass_times.append(time.perf_counter() - t_pass)
            now = time.perf_counter()
            for (_s, end_abs, _w) in committed:
                commit_lag.append((now - t_start) - end_abs)

    t_finish = time.perf_counter()
    text = online.finish()
    t_ready = time.perf_counter()

    return {
        "text": text,
        "wait": t_ready - state["t_release"],
        "final_flush": t_ready - t_finish,
        "passes": len(pass_times),
        "pass_avg": statistics.fmean(pass_times) if pass_times else 0.0,
        "pass_max": max(pass_times) if pass_times else 0.0,
        "commit_lag_avg": statistics.fmean(commit_lag) if commit_lag else 0.0,
        "commit_lag_max": max(commit_lag) if commit_lag else 0.0,
        "failed_passes": online.failed_passes,
        "errors": [repr(e) for e in errors],
    }


def run_one_shot(model, clip: dict, beam_size: int) -> dict:
    """What app.py does today: the entire decode happens after release."""
    t0 = time.perf_counter()
    text = transcribe_one_shot(model, clip["audio"], clip["language"],
                               beam_size=beam_size)
    return {"text": text, "wait": time.perf_counter() - t0}


# -------------------------------------------------------------------- main --

def load_model(model_size: str, threads: int, compute_type: str):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="auto", compute_type=compute_type,
                         cpu_threads=threads, num_workers=1)
    # First inference is always slower (lazy kernels). Pay it before timing.
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    for _ in model.transcribe(silent, beam_size=1, vad_filter=False,
                              condition_on_previous_text=False)[0]:
        pass
    return model


def default_threads() -> int:
    try:
        from system_info import physical_core_count
        return physical_core_count()
    except Exception:
        return os.cpu_count() or 4


def default_model() -> str:
    try:
        from config_manager import load_config
        return load_config().get("model_size", "small")
    except Exception:
        return "small"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", default="", help="clips separados por coma (default: todos)")
    parser.add_argument("--models", default="", help="tamaños de modelo separados por coma")
    parser.add_argument("--beams", default=str(STREAM_BEAM),
                        help="beam sizes del streaming, separados por coma")
    parser.add_argument("--full-beam", type=int, default=FULL_BEAM,
                        help="beam del one-shot (baseline de app.py)")
    parser.add_argument("--threads", type=int, default=0, help="0 = núcleos físicos")
    parser.add_argument("--compute", default="int8", help="compute_type de CTranslate2")
    parser.add_argument("--min-chunk", type=float, default=MIN_CHUNK_S,
                        help="segundos de audio nuevo entre pasadas")
    parser.add_argument("--repeat", type=int, default=1, help="corridas por combinación")
    parser.add_argument("--no-realtime", action="store_true",
                        help="alimentar el audio lo más rápido posible (ESPERA deja de tener sentido)")
    parser.add_argument("--skip-oneshot", action="store_true", help="no medir el baseline")
    parser.add_argument("--target-wait", type=float, default=0.5,
                        help="objetivo de ESPERA en segundos (Wispr-Flow-like)")
    parser.add_argument("--json", default="", help="escribir resultados crudos a este archivo")
    parser.add_argument("--list", action="store_true", help="listar clips disponibles y salir")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        for stem in discover_clips():
            print(stem)
        return

    stems = [s.strip() for s in args.clips.split(",") if s.strip()] or discover_clips()
    if not stems:
        raise SystemExit(
            f"No hay clips en {RESOURCES}. Grabá uno con:\n"
            f"  python benchmark/record_clip.py mi-clip"
        )
    clips = [load_clip(s) for s in stems]

    models = [m.strip() for m in args.models.split(",") if m.strip()] or [default_model()]
    beams = [int(b) for b in args.beams.split(",") if b.strip()]
    threads = args.threads or default_threads()
    realtime = not args.no_realtime

    print(f"Clips: {', '.join(c['stem'] for c in clips)}")
    print(f"Modelos: {', '.join(models)} · beams {beams} · threads {threads} "
          f"· compute {args.compute} · min_chunk {args.min_chunk}s")
    if not realtime:
        print("⚠  --no-realtime: el audio entra de golpe. ESPERA y atraso NO representan\n   el uso real — este modo sirve solo para comparar throughput bruto de decodificación.")
    print()

    rows = []
    for model_size in models:
        print(f"Cargando '{model_size}'…", flush=True)
        model = load_model(model_size, threads, args.compute)
        for clip in clips:
            for beam in beams:
                for run in range(args.repeat):
                    stream = run_streaming(model, clip, realtime, args.min_chunk, beam)
                    one_shot = ({"text": "", "wait": float("nan")} if args.skip_oneshot
                                else run_one_shot(model, clip, args.full_beam))
                    row = {
                        "model": model_size,
                        "clip": clip["stem"],
                        "duration": clip["duration"],
                        "beam": beam,
                        "run": run + 1,
                        "threads": threads,
                        "realtime": realtime,
                        "wer_stream": word_error_rate(clip["reference"], stream["text"]),
                        "wer_full": (float("nan") if args.skip_oneshot
                                     else word_error_rate(clip["reference"], one_shot["text"])),
                        "wait_stream": stream["wait"],
                        "wait_full": one_shot["wait"],
                        "text_stream": stream["text"],
                        "text_full": one_shot["text"],
                        "reference": clip["reference"],
                        **{k: stream[k] for k in
                           ("passes", "pass_avg", "pass_max", "commit_lag_avg",
                            "commit_lag_max", "failed_passes", "errors")},
                    }
                    rows.append(row)
                    print(f"  {clip['stem']:<22} beam={beam} run={run + 1}  "
                          f"espera {row['wait_stream']:5.2f}s vs {row['wait_full']:5.2f}s  "
                          f"WER {row['wer_stream'] * 100:4.1f}% vs {row['wer_full'] * 100:4.1f}%",
                          flush=True)
        del model

    report(rows, args)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nResultados crudos → {args.json}")


def report(rows, args):
    print("\n" + "=" * 108)
    print("  ESPERA = segundos entre soltar la tecla y tener el texto. Es LA métrica.")
    print("=" * 108)
    header = (f"{'modelo':<9}{'clip':<22}{'dur':>6}{'beam':>5}"
              f"{'ESPERA':>9}{'hoy':>8}{'gana':>7}"
              f"{'WER':>7}{'WER hoy':>9}"
              f"{'pass avg':>10}{'pass max':>10}{'atraso':>8}")
    print(header)
    print("-" * 108)

    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["clip"], row["beam"]), []).append(row)

    for (model_size, clip, beam), group in groups.items():
        def avg(key):
            values = [g[key] for g in group if g[key] == g[key]]  # drop NaN
            return statistics.fmean(values) if values else float("nan")

        wait_stream, wait_full = avg("wait_stream"), avg("wait_full")
        speedup = (f"{wait_full / wait_stream:.1f}x"
                   if wait_stream > 0 and wait_full == wait_full else "—")
        flag = "" if wait_stream <= args.target_wait else "  ⚠"
        print(f"{model_size:<9}{clip:<22}{group[0]['duration']:>5.1f}s{beam:>5}"
              f"{wait_stream:>8.2f}s{wait_full:>7.2f}s{speedup:>7}"
              f"{avg('wer_stream') * 100:>6.1f}%{avg('wer_full') * 100:>8.1f}%"
              f"{avg('pass_avg'):>9.2f}s{avg('pass_max'):>9.2f}s"
              f"{avg('commit_lag_avg'):>7.1f}s{flag}")

    print("-" * 108)
    slow = [r for r in rows if r["pass_max"] > args.min_chunk]
    if slow:
        combos = sorted({(r["model"], r["beam"]) for r in slow})
        print(f"  ⚠  Pasada más lenta que min_chunk ({args.min_chunk}s) en: "
              + ", ".join(f"{m} beam={b}" for m, b in combos)
              + " → el streaming se atrasa respecto del habla.")
    failed = sum(r["failed_passes"] for r in rows)
    if failed:
        print(f"  ⚠  {failed} pasada(s) fallaron. Primer error: {next(r['errors'][0] for r in rows if r['errors'])}")
    over = [r for r in rows if r["wait_stream"] > args.target_wait]
    if over:
        print(f"  ⚠  {len(over)}/{len(rows)} corridas por encima del objetivo "
              f"de {args.target_wait}s de espera.")
    else:
        print(f"  ✓  Todas las corridas bajo el objetivo de {args.target_wait}s de espera.")
    print("=" * 108)


if __name__ == "__main__":
    main()
