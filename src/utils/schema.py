"""Schema definitions and validation for unified metadata tables."""

from __future__ import annotations

from typing import Iterable, Mapping

UNIFIED_METADATA_COLUMNS = [
    "utt_id",
    "path",
    "dataset",
    "split",
    "label",
    "speaker_id",
    "attack_id",
    "sr",
    "duration_sec",
]

_ALLOWED_SPLITS = {"train", "val", "validation", "test"}
_ALLOWED_LABELS = {"bonafide", "spoof"}


class SchemaValidationError(ValueError):
    """Raised when metadata rows do not match the unified metadata schema."""


def _validate_row_types(row: Mapping[str, object], index: int) -> None:
    if not isinstance(row["utt_id"], str) or not row["utt_id"]:
        raise SchemaValidationError(f"row {index}: utt_id must be a non-empty string")
    if not isinstance(row["path"], str) or not row["path"]:
        raise SchemaValidationError(f"row {index}: path must be a non-empty string")
    if not isinstance(row["dataset"], str) or not row["dataset"]:
        raise SchemaValidationError(f"row {index}: dataset must be a non-empty string")
    if row["split"] not in _ALLOWED_SPLITS:
        raise SchemaValidationError(
            f"row {index}: split must be one of {_ALLOWED_SPLITS}, got {row['split']!r}"
        )
    if row["label"] not in _ALLOWED_LABELS:
        raise SchemaValidationError(
            f"row {index}: label must be one of {_ALLOWED_LABELS}, got {row['label']!r}"
        )
    if not isinstance(row["speaker_id"], str) or not row["speaker_id"]:
        raise SchemaValidationError(f"row {index}: speaker_id must be a non-empty string")
    if not isinstance(row["attack_id"], str) or not row["attack_id"]:
        raise SchemaValidationError(f"row {index}: attack_id must be a non-empty string")

    sr = row["sr"]
    if not isinstance(sr, int) or sr <= 0:
        raise SchemaValidationError(f"row {index}: sr must be a positive integer")

    duration_sec = row["duration_sec"]
    if not isinstance(duration_sec, (int, float)) or duration_sec <= 0:
        raise SchemaValidationError(f"row {index}: duration_sec must be a positive number")


def validate_metadata_rows(rows: Iterable[Mapping[str, object]]) -> None:
    """Validate iterable metadata rows against the unified schema.

    Args:
        rows: Iterable of row dictionaries.

    Raises:
        SchemaValidationError: If any row is missing columns, has extra columns,
            or contains invalid values.
    """

    for idx, row in enumerate(rows):
        row_keys = list(row.keys())
        if row_keys != UNIFIED_METADATA_COLUMNS:
            raise SchemaValidationError(
                "row "
                f"{idx}: expected keys {UNIFIED_METADATA_COLUMNS}, got {row_keys}"
            )
        _validate_row_types(row, idx)
