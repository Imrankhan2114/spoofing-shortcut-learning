"""I/O helpers for metadata builder pipelines."""

from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import yaml

from src.utils.schema import UNIFIED_METADATA_COLUMNS

try:
    import soundfile as sf
except Exception:  # pragma: no cover - fallback path tested without mocking imports
    sf = None


def load_yaml(path: str | Path) -> dict:
    """Load YAML file into a dictionary."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_metadata_csv(rows: Iterable[Mapping[str, object]], out_path: str | Path) -> None:
    """Write metadata rows to CSV using the unified column order."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def audio_info(path: str | Path) -> Tuple[int, float]:
    """Return (sample_rate, duration_sec) for an audio file.

    Uses soundfile when available and falls back to Python's wave module for WAV
    files.
    """

    path = Path(path)

    if sf is not None:
        info = sf.info(str(path))
        return int(info.samplerate), float(info.duration)

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_f:
            sr = wav_f.getframerate()
            n_frames = wav_f.getnframes()
        return int(sr), float(n_frames / sr)

    raise RuntimeError(
        "soundfile is unavailable and wave fallback only supports '.wav' files"
    )


def check_sample_paths_exist(rows: Iterable[Mapping[str, object]], sample_size: int = 20) -> None:
    """Check that the first N metadata paths exist on disk."""

    missing = []
    for idx, row in enumerate(rows):
        if idx >= sample_size:
            break
        p = Path(str(row["path"]))
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Sample path existence check failed for first "
            f"{sample_size} rows. Missing: {missing}"
        )
