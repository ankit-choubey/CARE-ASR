"""CARE-ASR AfriSpeech-200 dataset download utility (Task S1a).

Downloads the AfriSpeech-200 clinical ASR corpus from Hugging Face
(``intronhealth/afrispeech-200``, configuration ``all``, split ``test``),
validates that the download succeeded, that the dataset is non-empty, and
that the columns required by the evaluation pipeline (``audio`` and
``transcript``) are present. With ``--save-to-disk`` the dataset is persisted
under ``data/raw/afrispeech`` (or a custom ``--output-dir``) via
``datasets.Dataset.save_to_disk()`` so downstream tooling such as
``scripts/run_eval.py`` can reload it locally with ``load_from_disk()``.

This script is ONLY responsible for downloading and optionally saving the raw
AfriSpeech dataset. Preprocessing, FAISS index construction, and NER
extraction are intentionally out of scope.

Usage:
    python scripts/download_afrispeech.py
    python scripts/download_afrispeech.py --save-to-disk
    python scripts/download_afrispeech.py --save-to-disk --overwrite

Run from the repository root.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

DATASET_ID = "intronhealth/afrispeech-200"
DATASET_CONFIG = "all"
DATASET_SPLIT = "test"
REQUIRED_COLUMNS: tuple[str, ...] = ("audio", "transcript")
DEFAULT_OUTPUT_DIR = "data/raw/afrispeech"


def load_afrispeech_dataset() -> Dataset:
    """Download the AfriSpeech-200 test split from Hugging Face.

    Loads ``intronhealth/afrispeech-200`` (configuration ``all``, split
    ``test``) using the Hugging Face ``datasets`` library. The import is
    performed lazily so the module stays importable even when the ``datasets``
    package is unavailable or misconfigured.

    Returns:
        Dataset: The downloaded AfriSpeech-200 test split.

    Raises:
        RuntimeError: If the dataset cannot be downloaded from Hugging Face,
            or if the download returned no dataset object.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download AfriSpeech-200 dataset from Hugging Face "
            f"'{DATASET_ID}' (config='{DATASET_CONFIG}', split='{DATASET_SPLIT}'): {e}"
        ) from e

    if dataset is None:
        raise RuntimeError(f"AfriSpeech-200 dataset '{DATASET_ID}' download returned no dataset object.")

    return dataset


def validate_dataset(dataset: Dataset) -> None:
    """Validate the downloaded dataset is usable by the evaluation pipeline.

    Checks that the dataset contains at least one sample and that every column
    required by the pipeline (``audio`` and ``transcript``) is present.

    Args:
        dataset: The downloaded AfriSpeech-200 test split.

    Raises:
        RuntimeError: If the dataset is empty or a required column is missing.
    """
    if len(dataset) == 0:
        raise RuntimeError(f"AfriSpeech-200 dataset '{DATASET_ID}' loaded with no samples.")

    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing:
        raise RuntimeError(
            f"AfriSpeech-200 dataset '{DATASET_ID}' is missing required columns: " f"{', '.join(missing)}."
        )


def save_dataset_to_disk(dataset: Dataset, output_dir: Path) -> None:
    """Persist the dataset to disk with ``datasets.Dataset.save_to_disk()``.

    Any pre-existing directory at ``output_dir`` is removed first because the
    ``datasets`` library raises ``FileExistsError`` when ``save_to_disk()``
    targets an existing directory (behavior since ``datasets`` >= 3.x). Parent
    directories are created automatically, the dataset is saved, and the
    saved-dataset marker file is verified to exist afterwards.

    Args:
        dataset: The validated AfriSpeech-200 test split.
        output_dir: Directory under which the dataset is persisted.

    Raises:
        RuntimeError: If the dataset cannot be saved, or if the saved dataset
            was not created at the target location.
    """
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))
    except Exception as e:
        raise RuntimeError(f"Failed to save AfriSpeech-200 dataset to '{output_dir}': {e}") from e

    if not (output_dir / "dataset_info.json").exists():
        raise RuntimeError(f"Saved dataset was not created at '{output_dir}'.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the download utility."""
    parser = argparse.ArgumentParser(
        description=(
            "Download the AfriSpeech-200 dataset (intronhealth/afrispeech-200, "
            "config 'all', split 'test') from Hugging Face and optionally persist "
            "it to disk with save_to_disk()."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save the dataset into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--save-to-disk",
        action="store_true",
        help="Persist the downloaded dataset to disk with Dataset.save_to_disk().",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing saved dataset at the output directory if present.",
    )
    return parser


def print_summary(dataset: Dataset, output_dir: Path, save_to_disk: bool, elapsed_sec: float) -> None:
    """Print a concise summary of the download run.

    Args:
        dataset: The validated AfriSpeech-200 test split.
        output_dir: Configured output directory.
        save_to_disk: Whether the dataset was persisted to disk.
        elapsed_sec: Wall-clock duration of the run in seconds.
    """
    print("=" * 60)
    print("AfriSpeech-200 Download Complete")
    print("=" * 60)
    print(f"Dataset:            {DATASET_ID}")
    print(f"Split:              {DATASET_SPLIT}")
    print(f"Samples:            {len(dataset):,}")
    saved_note = " (not saved)" if not save_to_disk else ""
    print(f"Output directory:   {output_dir}{saved_note}")
    print(f"Duration:           {elapsed_sec:.2f}s")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """Orchestrate the AfriSpeech-200 download and optional persistence.

    Execution order:
        1. Parse CLI arguments.
        2. Skip the download when a saved dataset already exists and
           ``--overwrite`` was not requested.
        3. Download the dataset (load_afrispeech_dataset).
        4. Validate the dataset (validate_dataset).
        5. Persist the dataset with ``--save-to-disk`` (save_dataset_to_disk).
        6. Print a concise summary.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        int: 0 on success, 1 on any failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    start_time = time.perf_counter()

    if args.save_to_disk and not args.overwrite and (args.output_dir / "dataset_info.json").exists():
        logger.info(
            "Dataset already exists at '%s'; use --overwrite to replace it.",
            args.output_dir,
        )
        return 0

    try:
        logger.info(
            "Downloading %s (config='%s', split='%s')...",
            DATASET_ID,
            DATASET_CONFIG,
            DATASET_SPLIT,
        )
        dataset = load_afrispeech_dataset()
        validate_dataset(dataset)
        logger.info("Downloaded %d samples.", len(dataset))

        if args.save_to_disk:
            save_dataset_to_disk(dataset, args.output_dir)
            logger.info("Saved dataset to '%s'.", args.output_dir)
    except RuntimeError as e:
        logger.error("%s", e)
        return 1

    elapsed_sec = time.perf_counter() - start_time
    print_summary(dataset, args.output_dir, args.save_to_disk, elapsed_sec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
