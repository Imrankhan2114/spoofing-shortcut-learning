"""CLI for building unified dataset metadata CSV files."""

from __future__ import annotations

import argparse
from typing import Dict, List

from src.datasets.asvspoof2019_la import build_asvspoof2019_la_rows
from src.datasets.for_original import build_for_rows
from src.datasets.in_the_wild import build_in_the_wild_rows
from src.utils.io import audio_info, check_sample_paths_exist, load_yaml, write_metadata_csv
from src.utils.schema import validate_metadata_rows


def _dispatch_builder(config: Dict) -> List[Dict[str, object]]:
    dataset = str(config.get("dataset", "")).lower()
    if dataset in {"asvspoof2019_la", "asvspoof2019-la", "asvspoof2019"}:
        return build_asvspoof2019_la_rows(config)
    if dataset in {"for", "for_original", "fake_or_real"}:
        return build_for_rows(config)
    if dataset in {"in_the_wild", "inthewild", "in-the-wild"}:
        return build_in_the_wild_rows(config)
    raise ValueError(
        "Unknown dataset type in config['dataset']. "
        "Expected one of: asvspoof2019_la, for_original, in_the_wild"
    )


def _attach_audio_metadata(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    with_audio: List[Dict[str, object]] = []
    for row in rows:
        sr, duration_sec = audio_info(row["path"])
        with_audio.append(
            {
                **row,
                "sr": sr,
                "duration_sec": duration_sec,
            }
        )
    return with_audio


def build_metadata_from_config(config: Dict) -> List[Dict[str, object]]:
    """Build and validate unified metadata rows from a dataset config."""

    rows = _dispatch_builder(config)
    rows = _attach_audio_metadata(rows)
    validate_metadata_rows(rows)
    check_sample_paths_exist(rows, sample_size=20)
    return rows


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    config = load_yaml(args.config)
    rows = build_metadata_from_config(config)
    write_metadata_csv(rows, args.out)


if __name__ == "__main__":
    main()
