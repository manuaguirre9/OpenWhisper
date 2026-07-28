"""
Lightweight CPU topology helpers shared by the settings UI and the
transcription engine. Kept dependency-light (os + psutil) so importing it
from the settings window doesn't pull in faster-whisper.
"""
import os

try:
    import psutil
except Exception:  # psutil is optional; fall back to logical count.
    psutil = None


def logical_core_count() -> int:
    """Logical processors (includes SMT / hyperthreading)."""
    return os.cpu_count() or 4


def physical_core_count() -> int:
    """
    Physical CPU cores, excluding SMT/hyperthreading. CTranslate2 (the
    backend under faster-whisper) generally peaks at the physical core
    count; using all logical threads oversubscribes and can slow things
    down. Falls back to the logical count, then 4, if detection fails.
    """
    if psutil is not None:
        try:
            n = psutil.cpu_count(logical=False)
            if n:
                return int(n)
        except Exception:
            pass
    return logical_core_count()


def resolve_cpu_threads(value) -> int:
    """
    Turn the configured `cpu_threads` value into the concrete thread count
    that will actually be handed to WhisperModel.

    - 0 (or invalid / non-positive) = auto = physical core count.
    - Any positive value is clamped to the logical core count, since asking
      for more threads than the CPU has only adds contention.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0

    if value <= 0:
        return physical_core_count()

    return min(value, logical_core_count())
