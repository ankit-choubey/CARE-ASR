"""Run the T11 threshold-tuning and error-analysis evaluation pipeline.

This script orchestrates the full offline evaluation flow built in T11:

    1. Load ground-truth and predicted evaluation datasets.
    2. Generate a baseline audit report (ErrorAnalysisEngine).
    3. Build per-category candidate metrics from the predicted outputs.
    4. Tune category thresholds (ThresholdTuner.run_grid).
    5. Apply the winning thresholds (done internally by the tuner).
    6. Re-run error analysis.
    7. Emit the final ErrorAnalysisAuditOutput.
    8. Export it as JSON (save_audit_report).

All component logic is reused from the existing care_asr modules; this script
only wires them together, so no evaluation, tuning, or configuration logic is
duplicated. Threshold configuration is loaded by CategoryThresholdEngine via
get_settings().

Usage:
    python scripts/run_tuning_eval.py \
        --ground-truth data/processed/ground_truth.json \
        --predictions data/processed/predictions.json \
        --output outputs/audit_reports/run_001_audit_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from care_asr.contracts.error_analysis_output import ErrorAnalysisAuditOutput, save_audit_report
from care_asr.evaluation.metrics_calculator import ErrorAnalysisEngine
from care_asr.thresholds.threshold_engine import CategoryThresholdEngine
from care_asr.thresholds.threshold_tuner import ThresholdTuner, ThresholdTuningResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_METRIC_KEYS: tuple[str, ...] = (
    "semantic_similarity",
    "phonetic_distance",
    "asr_confidence",
    "entropy",
)

DEFAULT_GRID: dict[str, list[float]] = {
    "min_semantic_similarity": [0.80, 0.85, 0.90],
    "max_phonetic_distance": [1.0, 2.0, 3.0],
    "min_asr_confidence": [0.60, 0.75, 0.85],
    "max_entropy": [0.30, 0.45, 0.60],
}


def load_dataset(path: Path | str) -> list[dict[str, Any]]:
    """Loads a JSON evaluation dataset (a list of transcript dicts) from disk.

    Args:
        path (Path | str): Path to the JSON dataset file.

    Returns:
        list[dict[str, Any]]: The loaded transcript list.

    Raises:
        RuntimeError: If the file is missing, unreadable, not valid JSON, or
            does not contain a JSON list.
    """
    data_path = Path(path)
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Dataset file not found: {data_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from '{data_path}': {exc}") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Dataset at '{data_path}' must be a JSON list of transcripts, got {type(data).__name__}.")
    return data


def build_candidate_metrics(
    predicted_outputs: list[dict[str, Any]],
    known_categories: set[str] | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Extracts per-entity threshold metrics grouped by category for the tuner.

    Predicted entities are expected to carry a ``metrics`` dict with the four
    keys consumed by CategoryThresholdEngine: ``semantic_similarity``,
    ``phonetic_distance``, ``asr_confidence``, and ``entropy``. Entities with a
    missing or non-numeric metrics block, and entities whose category is not in
    ``known_categories`` when supplied, are logged and skipped so tuning never
    fabricates data. The audit engine ignores the extra ``metrics`` field.

    Args:
        predicted_outputs (list[dict[str, Any]]): Predicted validated entity outputs.
        known_categories (set[str] | None): Categories the engine can tune; entities
            outside it are skipped, mirroring the audit engine's tolerant handling.

    Returns:
        dict[str, list[dict[str, float]]]: Candidate metrics keyed by category.
    """
    by_category: dict[str, list[dict[str, float]]] = {}
    for item in predicted_outputs:
        transcript_id = str(item.get("transcript_id", ""))
        for entity in item.get("entities", []):
            metrics = entity.get("metrics")
            if not isinstance(metrics, dict):
                logger.warning("Entity in transcript '%s' has no metrics dict; skipping for tuning.", transcript_id)
                continue
            missing = [key for key in _METRIC_KEYS if key not in metrics]
            if missing:
                logger.warning(
                    "Entity in transcript '%s' is missing metrics keys %s; skipping for tuning.",
                    transcript_id,
                    missing,
                )
                continue
            non_numeric = [
                key
                for key in _METRIC_KEYS
                if not isinstance(metrics[key], (int, float)) or isinstance(metrics[key], bool)
            ]
            if non_numeric:
                logger.warning(
                    "Entity in transcript '%s' has non-numeric metrics keys %s; skipping for tuning.",
                    transcript_id,
                    non_numeric,
                )
                continue
            category = str(entity.get("category", ""))
            if known_categories is not None and category not in known_categories:
                logger.warning(
                    "Entity in transcript '%s' has unknown category '%s'; skipping for tuning.",
                    transcript_id,
                    category,
                )
                continue
            by_category.setdefault(category, []).append({key: float(metrics[key]) for key in _METRIC_KEYS})
    return by_category


def run_tuning_evaluation(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    batch_id: str,
    grid: dict[str, list[float]],
    output_path: Path,
) -> tuple[list[ThresholdTuningResult], ErrorAnalysisAuditOutput]:
    """Executes the full tuning-evaluation pipeline and returns its results.

    The pipeline runs a baseline audit, tunes category thresholds via
    ThresholdTuner.run_grid (which also applies the winning thresholds to the
    shared CategoryThresholdEngine), re-runs the audit, and exports the final
    report to ``output_path``. In this offline evaluation the audit metrics are
    computed from the fixed prediction files, so the baseline and final reports
    coincide; the re-run keeps the pipeline shape ready for online flows where
    thresholds affect downstream acceptance.

    Args:
        ground_truth (list[dict[str, Any]]): Ground-truth transcript annotations.
        predictions (list[dict[str, Any]]): Predicted validated entity outputs.
        batch_id (str): Audit report batch identifier.
        grid (dict[str, list[float]]): Threshold tuning grid.
        output_path (Path): Destination path for the exported audit report.

    Returns:
        tuple[list[ThresholdTuningResult], ErrorAnalysisAuditOutput]: The tuning
            summaries (empty when no entity metrics were available) and the final
            audit report.
    """
    engine = ErrorAnalysisEngine()
    threshold_engine = CategoryThresholdEngine()
    tuner = ThresholdTuner(threshold_engine)

    logger.info("Generating baseline audit report '%s'...", batch_id)
    baseline_report = engine.generate_audit_report(batch_id, ground_truth, predictions)

    candidate_metrics = build_candidate_metrics(predictions, set(threshold_engine.thresholds))
    if not candidate_metrics:
        logger.warning(
            "No entity metrics found in predictions; skipping threshold tuning and keeping current thresholds."
        )
        tuning_results: list[ThresholdTuningResult] = []
    else:
        logger.info("Tuning thresholds over %d category/categories...", len(candidate_metrics))
        tuning_results = tuner.run_grid(candidate_metrics, grid)

    logger.info("Re-running error analysis with tuned thresholds...")
    final_report = engine.generate_audit_report(batch_id, ground_truth, predictions)

    logger.info(
        "Baseline rectified F1: %.4f | Final rectified F1: %.4f",
        baseline_report.overall_metrics.rectified_f1,
        final_report.overall_metrics.rectified_f1,
    )

    save_audit_report(final_report, output_path)
    logger.info("Audit report written to %s", output_path)
    return tuning_results, final_report


def parse_grid(raw_grid: str | None) -> dict[str, list[float]]:
    """Parses the optional --grid JSON argument, defaulting to DEFAULT_GRID.

    Args:
        raw_grid (str | None): JSON object mapping threshold names to value lists.

    Returns:
        dict[str, list[float]]: The validated tuning grid.

    Raises:
        RuntimeError: If the JSON is invalid or the shape is wrong.
    """
    if raw_grid is None:
        return {key: list(values) for key, values in DEFAULT_GRID.items()}
    try:
        parsed = json.loads(raw_grid)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid --grid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("--grid must be a JSON object mapping threshold names to value lists.")

    grid: dict[str, list[float]] = {}
    for key, values in parsed.items():
        if not isinstance(values, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            raise RuntimeError(f"--grid key '{key}' must map to a JSON list of numbers.")
        grid[str(key)] = [float(value) for value in values]
    return grid


def print_summary(tuning_results: list[ThresholdTuningResult], output_path: Path) -> None:
    """Prints the concise terminal summary of the tuning run."""
    print("\n===== Threshold Tuning & Error Analysis Summary =====")
    if not tuning_results:
        print("No categories tuned (no entity metrics available in predictions).")
    else:
        print(f"Categories tuned: {', '.join(result.category for result in tuning_results)}")
        for result in tuning_results:
            print(f"  {result.category}:")
            print(f"    original thresholds:        {result.original_thresholds}")
            print(f"    tuned thresholds:           {result.tuned_thresholds}")
            print(f"    combinations evaluated:     {result.combinations_evaluated}")
            print(f"    best score:                 {result.best_score:.4f}")
    print(f"Audit report output path: {output_path}")
    print("=====================================================")


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the T11 threshold-tuning and error-analysis pipeline: "
        "baseline audit, category threshold tuning, final audit, and JSON export."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/processed/ground_truth.json"),
        help="Path to the ground-truth dataset JSON (list of transcripts).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/predictions.json"),
        help="Path to the predicted outputs JSON (list of transcripts).",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Audit report batch identifier (defaults to the predictions file stem).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Audit report output path (defaults to outputs/audit_reports/<batch_id>_audit_report.json).",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Optional JSON object of threshold-name to candidate value lists, "
        "e.g. '{\"min_semantic_similarity\": [0.80, 0.85]}' (defaults to a built-in grid).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the pipeline from command-line arguments and prints the summary.

    Returns:
        int: 0 on success, 1 on failure.
    """
    args = build_parser().parse_args(argv)
    try:
        grid = parse_grid(args.grid)
        ground_truth = load_dataset(args.ground_truth)
        predictions = load_dataset(args.predictions)

        batch_id = args.batch_id if args.batch_id else Path(args.predictions).stem
        output_path = (
            args.output if args.output is not None else Path("outputs/audit_reports") / f"{batch_id}_audit_report.json"
        )

        logger.info(
            "Loaded %d ground-truth transcripts and %d predicted transcripts.",
            len(ground_truth),
            len(predictions),
        )
        tuning_results, _ = run_tuning_evaluation(ground_truth, predictions, batch_id, grid, output_path)
    except Exception as exc:
        logger.error("Tuning-evaluation pipeline failed: %s", exc)
        return 1

    print_summary(tuning_results, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
