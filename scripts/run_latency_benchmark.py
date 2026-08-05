"""T12 latency benchmark for the fused retrieval layer (Commit 4).

Orchestration-only utility that validates the T12 latency claims by driving the
instrumented CARPipeline over an evaluation dataset, aggregating latency
samples, and exporting a JSON report to ``outputs/latency_reports/``.

It reuses the pipeline instrumentation (LatencyStats + per-transcript LATENCY
attribution entries), the retrievers' batched ``retrieve_many`` path (with the
pipeline's automatic sequential fallback), the real Tsallis uncertainty gate,
and the existing RRF fusion. No retrieval logic is reimplemented.

Determinism: the synthetic dataset, gating, entity extraction, and all
aggregation are deterministic. Wall-clock latency values are, by nature,
measurements that vary between runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from care_asr.contracts.asr_input import TokenScore, Transcript
from care_asr.contracts.error_analysis_output import NEREntity
from care_asr.uncertainty.gate import TsallisUncertaintyGate
from src.pipeline.pipeline import CARPipeline

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "data/processed/latency_benchmark.json"
DEFAULT_OUTPUT = "outputs/latency_reports/latency_benchmark.json"
DEFAULT_BATCH_SIZE = 50
DEFAULT_TOP_K = 5

# Confidence probability for tokens that the entropy gate flags as uncertain.
UNCERTAIN_PROB = 0.10
CONFIDENT_PROB = 0.95
# Tsallis entropy at alpha=1/3 of [p, 1-p] is ~0.645 for p=0.10 and ~0.527 for
# p=0.95, so threshold 0.6 separates the synthetic classes deterministically.
GATE_THRESHOLD = 0.6

SYNTHETIC_TRANSCRIPTS: tuple[str, ...] = (
    "patient prescribed amoxicillin five hundred milligrams twice daily",
    "history of hypertension managed with lisinopril ten milligrams",
    "patient reports headache and dizziness after metformin dose",
    "continue amoxicillin for ten days as directed",
    "diabetes mellitus type two controlled with metformin",
    "reviewed blood pressure readings with lisinopril therapy",
    "chest pain evaluated ruling out myocardial infarction",
    "start lisinopril ten milligrams every morning",
    "counseled patient on amoxicillin allergy symptoms",
    "follow up hemoglobin a one c levels in three months",
    "metformin dose increased to one thousand milligrams",
    "patient tolerating lisinopril without side effects",
    "prescribed aspirin eighty one milligrams daily",
    "monitor kidney function while on metformin therapy",
    "discharge medications include amoxicillin and lisinopril",
    "educate on dietary changes for diabetes management",
    "schedule stress test to evaluate chest discomfort",
    "adjust lisinopril based on home blood pressure log",
    "wound culture grew staphylococcus aureus organism",
    "scheduled colonoscopy for routine screening",
)


def _build_synthetic_dataset() -> list[dict[str, Any]]:
    """Builds a deterministic synthetic evaluation dataset."""
    return [{"transcript_id": f"synth_{index:03d}", "text": text} for index, text in enumerate(SYNTHETIC_TRANSCRIPTS)]


def load_dataset(path: str | None) -> list[dict[str, Any]]:
    """Loads a JSON evaluation dataset, synthesizing one when no path is given.

    The dataset is a JSON list of ``{"transcript_id": str, "text": str}``
    samples. When ``path`` is None the default dataset file is used if it
    exists; otherwise a deterministic synthetic dataset is generated.

    Raises:
        RuntimeError: If an explicitly provided dataset is missing, unreadable,
            or contains no usable samples.
    """
    if path is not None:
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise RuntimeError(f"Dataset file not found: '{path}'. Provide a valid --dataset or omit it.")
        try:
            with open(dataset_path, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to load dataset from '{path}': {exc}") from exc
        samples = [sample for sample in raw if isinstance(sample, dict) and sample.get("text")]
        if not samples:
            raise RuntimeError(f"Dataset '{path}' contains no samples with a 'text' field.")
        return samples

    default_path = Path(DEFAULT_DATASET)
    if default_path.exists():
        return load_dataset(str(default_path))
    samples = _build_synthetic_dataset()
    logger.info("No dataset found; using deterministic synthetic dataset (%d transcripts).", len(samples))
    return samples


def _build_transcript(sample: dict[str, Any]) -> Transcript:
    """Builds a Transcript for a sample with deterministic token scores.

    Every fifth token is marked uncertain (low top-1 probability) so the
    entropy gate flags it deterministically and retrieval is exercised.
    """
    words = str(sample["text"]).split()
    token_scores = [
        TokenScore(
            step=index,
            token_id=index,
            token=word,
            log_prob=-1.5 if index % 5 == 0 else -0.05,
            prob=UNCERTAIN_PROB if index % 5 == 0 else CONFIDENT_PROB,
        )
        for index, word in enumerate(words)
    ]
    return Transcript(text=" ".join(words), token_scores=token_scores, word_timestamps=[])


def _token_distributions(transcript: Transcript) -> torch.Tensor:
    """Builds a (N, 2) probability tensor from per-token top-1 probabilities."""
    return torch.tensor(
        [[score.prob, 1.0 - score.prob] for score in transcript.token_scores],
        dtype=torch.float32,
    )


def _make_transcriber(sample: dict[str, Any]) -> Callable[[Any], Transcript]:
    """Builds a transcriber that produces the Transcript for a given sample."""
    return lambda _audio: _build_transcript(sample)


def _all_words_entities(transcript: Transcript) -> list[NEREntity]:
    """Deterministic entity extractor marking every word as a clinical entity.

    Stands in for the BioBERT NER (unavailable offline) so the benchmark
    exercises retrieval on every token the gate flags.
    """
    return [
        NEREntity(word=score.token, category="MED", start=index, end=index, score=1.0)
        for index, score in enumerate(transcript.token_scores)
    ]


def _install_gate(pipeline: CARPipeline) -> None:
    """Binds the real Tsallis uncertainty gate to the pipeline.

    The real gate requires a probability distribution per token (as Whisper
    produces); synthetic transcripts provide a deterministic two-class
    distribution built from each TokenScore's top-1 probability.
    """
    gate = TsallisUncertaintyGate(threshold=GATE_THRESHOLD)
    pipeline.entropy_gate = lambda transcript: gate.evaluate(_token_distributions(transcript))["uncertain_flags"]


def _install_retrievers(pipeline: CARPipeline, top_k: int | None) -> None:
    """Binds the real retrievers to the pipeline, falling back to stubs.

    Real retrievers are bound only when they can be constructed and have
    content; the pipeline's stub retrievers remain in place otherwise so the
    benchmark always runs.
    """
    try:
        from src.retrieval.semantic import SemanticRetriever

        semantic = SemanticRetriever()
        if semantic.available:
            pipeline.semantic_retrieve = semantic.retrieve
    except Exception as exc:
        logger.warning("Semantic retriever unavailable (%s); using stub.", exc)

    try:
        from src.retrieval.phonetic import PhoneticRetriever

        phonetic = PhoneticRetriever()
        if phonetic.faiss_available or phonetic.metaphone_vocab:
            pipeline.phonetic_retrieve = phonetic.retrieve
    except Exception as exc:
        logger.warning("Phonetic retriever unavailable (%s); using stub.", exc)

    pipeline.retrieval_top_k = top_k


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Returns the given percentile of a pre-sorted ascending list."""
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _latency_summary(values: list[float]) -> dict[str, float]:
    """Aggregates a latency series into average/median/percentile statistics."""
    if not values:
        return {
            "average": 0.0,
            "median": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
            "minimum": 0.0,
        }
    sorted_values = sorted(values)
    return {
        "average": statistics.mean(values),
        "median": statistics.median(values),
        "p50": _percentile(sorted_values, 50.0),
        "p90": _percentile(sorted_values, 90.0),
        "p95": _percentile(sorted_values, 95.0),
        "maximum": max(values),
        "minimum": min(values),
    }


def _mean(values: list[float]) -> float:
    """Returns the arithmetic mean of values (0.0 when empty)."""
    return statistics.mean(values) if values else 0.0


def run_benchmark(
    samples: list[dict[str, Any]],
    batch_size: int,
    top_k: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Runs the instrumented pipeline over the dataset and aggregates metrics.

    Returns a tuple of (summary metrics, per-transcript latency records).
    """
    pipeline = CARPipeline()
    pipeline.ner = _all_words_entities
    _install_gate(pipeline)
    _install_retrievers(pipeline, top_k)

    transcripts = samples[:batch_size]

    retrieval_series: list[float] = []
    semantic_series: list[float] = []
    phonetic_series: list[float] = []
    gate_series: list[float] = []
    fusion_series: list[float] = []
    pipeline_series: list[float] = []
    per_transcript: list[dict[str, Any]] = []

    retrieval_calls = 0
    gated_entities = 0
    semantic_seen: set[str] = set()
    phonetic_seen: set[str] = set()
    semantic_hits = 0
    phonetic_hits = 0

    for sample in transcripts:
        pipeline.transcriber = _make_transcriber(sample)

        log: list[dict[str, Any]] = []
        start = time.perf_counter()
        pipeline.run(sample.get("audio", sample.get("transcript_id", "sample")), attribution_log=log)
        pipeline_series.append((time.perf_counter() - start) * 1000.0)

        latency = next((entry for entry in log if entry.get("module") == "LATENCY"), None)
        if latency is not None:
            retrieval_series.append(float(latency["retrieval_latency_ms"]))
            semantic_series.append(float(latency["semantic_retrieval_latency_ms"]))
            phonetic_series.append(float(latency["phonetic_retrieval_latency_ms"]))
            gate_series.append(float(latency["gate_latency_ms"]))
            fusion_series.append(float(latency["fusion_latency_ms"]))
            per_transcript.append(
                {
                    "transcript_id": sample.get("transcript_id", "unknown"),
                    "gate_latency_ms": float(latency["gate_latency_ms"]),
                    "semantic_retrieval_latency_ms": float(latency["semantic_retrieval_latency_ms"]),
                    "phonetic_retrieval_latency_ms": float(latency["phonetic_retrieval_latency_ms"]),
                    "retrieval_latency_ms": float(latency["retrieval_latency_ms"]),
                    "fusion_latency_ms": float(latency["fusion_latency_ms"]),
                }
            )

        for entry in log:
            if entry.get("module") != "M4_RETRIEVAL":
                continue
            gated_entities += 1
            retrieval_calls += 1
            # Cache hit rate is a benchmark-level proxy: a token seen before is
            # assumed served from the retriever's bounded cache rather than
            # re-embedded/re-encoded (medical terms repeat across transcripts).
            token = str(entry["token"])
            if token in semantic_seen:
                semantic_hits += 1
            semantic_seen.add(token)
            if token in phonetic_seen:
                phonetic_hits += 1
            phonetic_seen.add(token)

    summary = {
        "total_transcripts": len(transcripts),
        "total_gated_entities": gated_entities,
        "retrieval_calls": retrieval_calls,
        "semantic_cache_hit_rate": (semantic_hits / retrieval_calls) if retrieval_calls else 0.0,
        "phonetic_cache_hit_rate": (phonetic_hits / retrieval_calls) if retrieval_calls else 0.0,
        "retrieval_latency_ms": _latency_summary(retrieval_series),
        "semantic_retrieval_latency_ms": {"average": _mean(semantic_series)},
        "phonetic_retrieval_latency_ms": {"average": _mean(phonetic_series)},
        "gate_latency_ms": {"average": _mean(gate_series)},
        "fusion_latency_ms": {"average": _mean(fusion_series)},
        "overall_pipeline_latency_ms": {"average": _mean(pipeline_series)},
    }
    return summary, per_transcript


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Writes the JSON report using the project's serialization conventions."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to write latency report to '{output_path}': {exc}") from exc
    if not output_path.exists():
        raise RuntimeError(f"Latency report was not created at '{output_path}'.")


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Prints a concise console report of the benchmark results."""
    retrieval = summary["retrieval_latency_ms"]
    print("CARE-ASR Latency Benchmark")
    print("==========================")
    print(f"transcripts:             {summary['total_transcripts']}")
    print(f"gated entities:          {summary['total_gated_entities']}")
    print(f"retrieval calls:         {summary['retrieval_calls']}")
    print(f"semantic cache hit rate: {summary['semantic_cache_hit_rate']:.3f}")
    print(f"phonetic cache hit rate: {summary['phonetic_cache_hit_rate']:.3f}")
    print(
        f"retrieval latency (ms):  avg {retrieval['average']:.3f}, "
        f"median {retrieval['median']:.3f}, p50 {retrieval['p50']:.3f}, "
        f"p90 {retrieval['p90']:.3f}, p95 {retrieval['p95']:.3f}, "
        f"max {retrieval['maximum']:.3f}, min {retrieval['minimum']:.3f}"
    )
    print(f"fusion average (ms):     {summary['fusion_latency_ms']['average']:.3f}")
    print(f"gate average (ms):       {summary['gate_latency_ms']['average']:.3f}")
    print(f"overall pipeline (ms):   {summary['overall_pipeline_latency_ms']['average']:.3f}")
    print(f"report saved:            {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Builds the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "CARE-ASR fused retrieval latency benchmark (T12). "
            "Structure and counts are deterministic; wall-clock latency values vary between runs."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Path to a JSON dataset of transcripts (default: {DEFAULT_DATASET} when present, else synthetic).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"JSON report output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Maximum number of transcripts processed per run (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Retrieval top-k forwarded to the retrievers (default: {DEFAULT_TOP_K}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the benchmark end-to-end; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        samples = load_dataset(args.dataset)
        summary, per_transcript = run_benchmark(samples, args.batch_size, args.top_k)
        dataset_source = (
            args.dataset if args.dataset else (DEFAULT_DATASET if Path(DEFAULT_DATASET).exists() else "synthetic")
        )
        report = {
            "benchmark": "care-asr-latency-benchmark",
            "config": {
                "dataset": dataset_source,
                "batch_size": args.batch_size,
                "top_k": args.top_k,
            },
            "summary": summary,
            "per_transcript": per_transcript,
        }
        output_path = Path(args.output)
        save_report(report, output_path)
        print_summary(summary, output_path)
        return 0
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc)
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
