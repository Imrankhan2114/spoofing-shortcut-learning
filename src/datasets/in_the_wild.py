"""In-the-wild dataset metadata adapter."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List


def _speaker_disjoint_split(
    speakers: List[str], split_ratio: Dict[str, float], split_seed: int
) -> Dict[str, str]:
    """Create deterministic speaker-disjoint split assignment."""

    speakers = sorted(set(speakers))
    rng = random.Random(split_seed)
    rng.shuffle(speakers)

    n_total = len(speakers)
    n_train = int(n_total * split_ratio["train"])
    n_val = int(n_total * split_ratio["val"])
    # remainder goes to test
    split_by_speaker: Dict[str, str] = {}
    for idx, speaker in enumerate(speakers):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_by_speaker[speaker] = split

    return split_by_speaker


def build_in_the_wild_rows(config: Dict) -> List[Dict[str, object]]:
    """Build unified metadata rows for in-the-wild dataset.

    Expected config keys:
      - root
      - meta_csv
      - audio_dir (optional, default: '.')
      - label_map (e.g. {'bona-fide': 'bonafide', 'spoof': 'spoof'})
      - split_ratio: {'train': float, 'val': float, 'test': float}
      - split_seed: int
    """

    root = Path(config["root"])
    dataset = config.get("dataset", "in_the_wild")
    meta_csv = root / config["meta_csv"]
    audio_dir = root / config.get("audio_dir", ".")
    label_map = config["label_map"]
    split_ratio = config["split_ratio"]
    split_seed = int(config.get("split_seed", 0))

    if round(sum(split_ratio.values()), 6) != 1.0:
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")

    entries = []
    speakers: List[str] = []
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row["file"]
            speaker_id = row["speaker"]
            label = label_map[row["label"]]
            speakers.append(speaker_id)
            entries.append((file_name, speaker_id, label))

    split_by_speaker = _speaker_disjoint_split(speakers, split_ratio, split_seed)

    rows: List[Dict[str, object]] = []
    for file_name, speaker_id, label in entries:
        path = audio_dir / file_name
        rows.append(
            {
                "utt_id": Path(file_name).stem,
                "path": str(path),
                "dataset": dataset,
                "split": split_by_speaker[speaker_id],
                "label": label,
                "speaker_id": speaker_id,
                "attack_id": "unknown",
            }
        )

    return rows
