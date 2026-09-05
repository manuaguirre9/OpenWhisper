"""
Smoke tests for the benchmark plumbing (no model, no microphone).

The point of the harness is that ESPERA — the wait between releasing the key
and having the text — is measured the same way every run. These tests pin the
parts that would silently corrupt that number: fixture loading, real-time
pacing, and where the stopwatch starts and stops.
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

import bench_dictation as bench  # noqa: E402
from streaming_core import SAMPLE_RATE  # noqa: E402


class FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class FakeSegment:
    def __init__(self, words, text=""):
        self.words = [FakeWord(*w) for w in words]
        self.text = text


class SteadyModel:
    """Always returns the same two words, so LocalAgreement commits immediately."""

    WORDS = [(0.0, 0.5, " hola"), (0.5, 1.0, " mundo")]

    def transcribe(self, audio, **kwargs):
        if kwargs.get("word_timestamps"):
            return iter([FakeSegment(self.WORDS)]), None
        return iter([FakeSegment([], text="hola mundo")]), None


@pytest.fixture()
def clip():
    return {
        "stem": "fixture",
        "audio": np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32),
        "duration": 1.0,
        "language": "es",
        "reference": "hola mundo",
    }


def test_streaming_run_reports_a_wait_and_a_transcript(clip):
    result = bench.run_streaming(SteadyModel(), clip, realtime=False,
                                 min_chunk_s=0.5, beam_size=1)
    assert result["text"] == "hola mundo"
    assert result["wait"] >= 0.0
    assert result["passes"] >= 1
    assert result["failed_passes"] == 0


def test_one_shot_run_uses_the_no_word_timestamps_path(clip):
    result = bench.run_one_shot(SteadyModel(), clip, beam_size=5)
    assert result["text"] == "hola mundo"
    assert result["wait"] >= 0.0


def test_realtime_feeding_takes_about_as_long_as_the_audio(clip):
    """
    The whole point of pacing: decoding must compete with arriving audio.
    Fed instantly, a 1s clip would finish in milliseconds and ESPERA would be
    a fantasy number.
    """
    import time
    start = time.perf_counter()
    bench.run_streaming(SteadyModel(), clip, realtime=True,
                        min_chunk_s=0.5, beam_size=1)
    elapsed = time.perf_counter() - start
    assert elapsed >= 0.9        # ~1s of audio, paced


def test_wait_excludes_the_time_spent_speaking(clip):
    """ESPERA starts at the last audio block, not at the start of the clip."""
    result = bench.run_streaming(SteadyModel(), clip, realtime=True,
                                 min_chunk_s=0.5, beam_size=1)
    assert result["wait"] < clip["duration"]


def test_load_clip_reads_audio_and_ground_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RESOURCES", tmp_path)
    pcm = (np.zeros(SAMPLE_RATE, dtype=np.float32) * 32767).astype(np.int16)
    with wave.open(str(tmp_path / "demo.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    (tmp_path / "demo.json").write_text(
        json.dumps({"language": "es", "text": "hola"}), encoding="utf-8")

    loaded = bench.load_clip("demo")
    assert loaded["language"] == "es"
    assert loaded["reference"] == "hola"
    assert loaded["duration"] == pytest.approx(1.0, abs=0.05)


def test_a_clip_without_ground_truth_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "RESOURCES", tmp_path)
    with wave.open(str(tmp_path / "mute.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"\x00\x00" * SAMPLE_RATE)
    with pytest.raises(SystemExit, match="ground truth"):
        bench.load_clip("mute")


def test_shipped_fixture_is_discoverable_and_loadable():
    assert "librispeech-en-3081" in bench.discover_clips()
    loaded = bench.load_clip("librispeech-en-3081")
    assert loaded["language"] == "en"
    assert loaded["duration"] == pytest.approx(10.5, abs=0.2)
    assert "breakfast table" in loaded["reference"]
