"""Tests for dataset metadata builders and schema validation."""

from __future__ import annotations

import csv
import wave
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.datasets.asvspoof2019_la import build_asvspoof2019_la_rows
from src.experiments.build_metadata import build_metadata_from_config
from src.utils.schema import (
    SchemaValidationError,
    UNIFIED_METADATA_COLUMNS,
    validate_metadata_rows,
)


def _write_tiny_wav(path: Path, sr: int = 16000, frames: int = 1600) -> None:
    """Write a short, valid mono WAV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(b"\x00\x00" * frames)


def test_schema_validation_happy_path() -> None:
    row = {
        "utt_id": "utt1",
        "path": "/tmp/utt1.wav",
        "dataset": "toy",
        "split": "train",
        "label": "bonafide",
        "speaker_id": "spk1",
        "attack_id": "bonafide",
        "sr": 16000,
        "duration_sec": 0.1,
    }
    validate_metadata_rows([row])


def test_schema_validation_rejects_wrong_column_order() -> None:
    bad_row = {
        "path": "/tmp/utt1.wav",
        "utt_id": "utt1",
        "dataset": "toy",
        "split": "train",
        "label": "bonafide",
        "speaker_id": "spk1",
        "attack_id": "bonafide",
        "sr": 16000,
        "duration_sec": 0.1,
    }
    with pytest.raises(SchemaValidationError):
        validate_metadata_rows([bad_row])


def test_for_builder_and_csv_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "for"
    _write_tiny_wav(root / "training" / "real" / "a.wav")
    _write_tiny_wav(root / "training" / "fake" / "b.wav")
    _write_tiny_wav(root / "validation" / "real" / "c.wav")

    config = {
        "dataset": "for_original",
        "root": str(root),
    }
    rows = build_metadata_from_config(config)

    assert len(rows) == 3
    assert list(rows[0].keys()) == UNIFIED_METADATA_COLUMNS
    labels = {r["utt_id"]: r["label"] for r in rows}
    assert labels["a"] == "bonafide"
    assert labels["b"] == "spoof"
    assert {r["speaker_id"] for r in rows} == {"unknown"}
    assert all(r["sr"] == 16000 for r in rows)


def test_in_the_wild_speaker_disjoint_split(tmp_path: Path) -> None:
    root = tmp_path / "itw"
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True)

    for name in ["u1.wav", "u2.wav", "u3.wav", "u4.wav", "u5.wav", "u6.wav"]:
        _write_tiny_wav(audio_dir / name)

    meta_csv = root / "meta.csv"
    with meta_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"file": "u1.wav", "speaker": "s1", "label": "bona-fide"},
                {"file": "u2.wav", "speaker": "s1", "label": "bona-fide"},
                {"file": "u3.wav", "speaker": "s2", "label": "spoof"},
                {"file": "u4.wav", "speaker": "s3", "label": "spoof"},
                {"file": "u5.wav", "speaker": "s4", "label": "bona-fide"},
                {"file": "u6.wav", "speaker": "s5", "label": "spoof"},
            ]
        )

    config = {
        "dataset": "in_the_wild",
        "root": str(root),
        "meta_csv": "meta.csv",
        "audio_dir": "audio",
        "label_map": {"bona-fide": "bonafide", "spoof": "spoof"},
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "split_seed": 123,
    }
    rows = build_metadata_from_config(config)

    speaker_to_split = {}
    for row in rows:
        prev = speaker_to_split.get(row["speaker_id"])
        if prev is not None:
            assert prev == row["split"]
        speaker_to_split[row["speaker_id"]] = row["split"]

    assert {r["label"] for r in rows} == {"bonafide", "spoof"}


def test_asvspoof2019_protocol_parsing_and_paths(tmp_path: Path) -> None:
    root = tmp_path / "asvspoof"
    protocol = root / "protocols" / "train.txt"
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(
        "LA_0001 LA_T_1000001 - A01 spoof\n"
        "LA_0002 LA_T_1000002 - - bonafide\n",
        encoding="utf-8",
    )

    _write_tiny_wav(root / "train_wav" / "LA_T_1000001.wav")
    _write_tiny_wav(root / "train_wav" / "LA_T_1000002.wav")

    config = {
        "dataset": "asvspoof2019_la",
        "root": str(root),
        "audio_ext": ".wav",
        "audio_dirs": {"train": "train_wav"},
        "protocol_files": {"train": "protocols/train.txt"},
    }

    base_rows = build_asvspoof2019_la_rows(config)
    assert base_rows[0]["attack_id"] == "A01"
    assert base_rows[1]["attack_id"] == "bonafide"
    assert base_rows[0]["path"].endswith("LA_T_1000001.wav")

    full_rows = build_metadata_from_config(config)
    assert len(full_rows) == 2
    assert all(r["sr"] == 16000 for r in full_rows)
