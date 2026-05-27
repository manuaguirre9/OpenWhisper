"""
Speaker diarization using sherpa-onnx.

Stack:
  - Segmentation: pyannote-segmentation-3.0 exported to ONNX (~6MB)
  - Embedding:    CAM++ from 3D-Speaker (~28MB)
  - Runtime:      ONNX Runtime (no PyTorch, no HuggingFace token)

Models are hosted on the sherpa-onnx GitHub release pages and cached
locally in %APPDATA%/OpenWhisper/models/ on first use.
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Callable, List, Dict

import numpy as np
import sherpa_onnx

from config_manager import CONFIG_DIR


MODELS_DIR = Path(CONFIG_DIR) / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Segmentation model is distributed as a tarball; we unpack model.onnx out.
SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_PATH = MODELS_DIR / "pyannote-segmentation-3-0.onnx"

# CAM++ embedding model (multilingual despite the "zh-cn" tag — the embedding
# space generalizes well across languages since it uses acoustic features).
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
)
EMBEDDING_PATH = MODELS_DIR / "campplus_sv.onnx"


# ---------- model download ----------

def models_ready() -> bool:
    return SEGMENTATION_PATH.exists() and EMBEDDING_PATH.exists()


def _download(url: str, dest: Path, status_cb: Optional[Callable[[str, float], None]] = None,
              status_label: str = ""):
    """Download `url` to `dest`, reporting progress via status_cb(label, 0..1)."""
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(blocknum, blocksize, totalsize):
        if status_cb and totalsize > 0:
            status_cb(status_label, min(1.0, blocknum * blocksize / totalsize))

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    tmp.replace(dest)


def _extract_segmentation(tarball: Path, target: Path):
    """The segmentation release ships a tar.bz2 with model.onnx inside. Pull it out."""
    with tarfile.open(tarball, "r:bz2") as tar:
        for member in tar.getmembers():
            if member.name.endswith("model.onnx"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                with open(target, "wb") as out:
                    out.write(f.read())
                return
    raise RuntimeError("model.onnx not found inside segmentation tarball")


def ensure_models(status_cb: Optional[Callable[[str, float], None]] = None):
    """Download both models if not already cached locally. Idempotent."""
    if not SEGMENTATION_PATH.exists():
        tarball = MODELS_DIR / "segmentation.tar.bz2"
        _download(SEGMENTATION_URL, tarball, status_cb,
                  "Descargando modelo de segmentación")
        _extract_segmentation(tarball, SEGMENTATION_PATH)
        try:
            tarball.unlink()
        except OSError:
            pass

    if not EMBEDDING_PATH.exists():
        _download(EMBEDDING_URL, EMBEDDING_PATH, status_cb,
                  "Descargando modelo de hablantes")


# ---------- diarization engine ----------

class Diarizer:
    """
    Wraps sherpa-onnx's OfflineSpeakerDiarization. Construct once and reuse
    across files; `num_clusters` can be changed per call via set_config.
    """

    def __init__(self):
        if not models_ready():
            raise RuntimeError(
                "Modelos de diarización no descargados. "
                "Llamá a ensure_models() primero."
            )
        self._build_pipeline(num_clusters=-1)

    def _build_config(self, num_clusters: int) -> sherpa_onnx.OfflineSpeakerDiarizationConfig:
        return sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(SEGMENTATION_PATH)
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(EMBEDDING_PATH)
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters,
                threshold=0.5,
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )

    def _build_pipeline(self, num_clusters: int):
        self._pipeline = sherpa_onnx.OfflineSpeakerDiarization(
            self._build_config(num_clusters)
        )

    @property
    def sample_rate(self) -> int:
        return int(self._pipeline.sample_rate)

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        progress_cb: Optional[Callable[[float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> List[Dict]:
        """
        Run diarization on `audio_path`. Audio is auto-decoded to 16kHz mono
        via faster-whisper's bundled ffmpeg/PyAV pipeline (any common format
        works: mp3/wav/m4a/mp4/ogg/flac/...).

        num_speakers: known speaker count, or None to auto-detect.
        progress_cb: optional callable(fraction 0..1) called during processing.
        Returns sorted list of {"start", "end", "speaker"} dicts.
        """
        # Reconfigure clustering for this call if num_speakers changed.
        target = num_speakers if (num_speakers and num_speakers > 0) else -1
        self._pipeline.set_config(self._build_config(target))

        # Decode any container/codec to 16kHz mono float32.
        from faster_whisper.audio import decode_audio
        samples = decode_audio(audio_path, sampling_rate=self.sample_rate)

        # sherpa-onnx callback gets (processed_chunks, total_chunks) and
        # returns an int (0 to continue). Bridge to our fractional progress.
        cancelled = {"flag": False}

        def _sherpa_cb(done: int, total: int) -> int:
            if cancel_check is not None and cancel_check():
                cancelled["flag"] = True
                return 1  # any non-zero signals cancel to sherpa-onnx
            if progress_cb and total > 0:
                progress_cb(min(1.0, done / total))
            return 0

        result = self._pipeline.process(samples, callback=_sherpa_cb).sort_by_start_time()

        if cancelled["flag"]:
            return []

        return [
            {"start": float(r.start), "end": float(r.end), "speaker": int(r.speaker)}
            for r in result
        ]


# ---------- merge with whisper segments ----------

def assign_speakers_to_segments(whisper_segments: List[Dict],
                                 diar_segments: List[Dict]) -> List[Dict]:
    """
    For each Whisper segment, assign the speaker who has the most temporal
    overlap with it in the diarization output.

    Returns the Whisper segments with an extra "speaker" key (int).
    """
    enriched = []
    for ws in whisper_segments:
        ws_start = ws["start"]
        ws_end = ws["end"]
        overlaps: Dict[int, float] = {}
        for ds in diar_segments:
            ov = max(0.0, min(ws_end, ds["end"]) - max(ws_start, ds["start"]))
            if ov > 0:
                overlaps[ds["speaker"]] = overlaps.get(ds["speaker"], 0.0) + ov
        speaker = max(overlaps.items(), key=lambda x: x[1])[0] if overlaps else 0
        enriched.append({**ws, "speaker": speaker})
    return enriched
