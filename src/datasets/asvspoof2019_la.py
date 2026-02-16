"""ASVspoof 2019 LA dataset metadata adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def build_asvspoof2019_la_rows(config: Dict) -> List[Dict[str, object]]:
    """Build unified metadata rows for ASVspoof2019 LA.

    Expected config keys:
      - root
      - dataset (optional)
      - audio_ext (default: .flac)
      - audio_dirs: mapping from split names to relative directories
      - protocol_files: mapping from split names to relative protocol file paths
    """

    root = Path(config["root"])
    dataset = config.get("dataset", "asvspoof2019_la")
    audio_ext = config.get("audio_ext", ".flac")
    audio_dirs = config["audio_dirs"]
    protocol_files = config["protocol_files"]

    rows: List[Dict[str, object]] = []
    for split, protocol_rel in protocol_files.items():
        protocol_path = root / protocol_rel
        audio_dir = root / audio_dirs[split]

        with protocol_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid protocol line in {protocol_path}: {line!r}"
                    )
                speaker_id, audio_name, _, system_id, key = parts
                label = key
                attack_id = system_id if label == "spoof" else "bonafide"

                rows.append(
                    {
                        "utt_id": audio_name,
                        "path": str(audio_dir / f"{audio_name}{audio_ext}"),
                        "dataset": dataset,
                        "split": split,
                        "label": label,
                        "speaker_id": speaker_id,
                        "attack_id": attack_id,
                    }
                )

    return rows
