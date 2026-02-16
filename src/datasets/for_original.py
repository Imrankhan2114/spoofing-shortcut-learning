"""Fake-or-Real (FoR) dataset metadata adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def build_for_rows(config: Dict) -> List[Dict[str, object]]:
    """Build unified metadata rows for FoR from directory structure.

    Expected folder structure under root:
      training/{real,fake}, validation/{real,fake}, testing/{real,fake}
    """

    root = Path(config["root"])
    dataset = config.get("dataset", "for_original")
    audio_exts = tuple(config.get("audio_exts", [".wav", ".flac", ".mp3"]))

    split_map = {"training": "train", "validation": "val", "testing": "test"}
    label_map = {"real": "bonafide", "fake": "spoof"}

    rows: List[Dict[str, object]] = []
    for split_dir_name, split in split_map.items():
        for class_dir_name, label in label_map.items():
            audio_dir = root / split_dir_name / class_dir_name
            if not audio_dir.exists():
                continue
            for path in sorted(audio_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in audio_exts:
                    continue
                rows.append(
                    {
                        "utt_id": path.stem,
                        "path": str(path),
                        "dataset": dataset,
                        "split": split,
                        "label": label,
                        "speaker_id": "unknown",
                        "attack_id": "unknown",
                    }
                )

    return rows
