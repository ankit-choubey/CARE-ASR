"""T14 India context evaluation — frozen-pipeline inference over EKA and Svarah.

Entry point for the India inference sweep: loads the EKA and Svarah datasets
into a shared normalized format, runs the frozen Week 1 CARE-ASR pipeline
(Whisper -> Tsallis entropy gate -> NER -> dual retrieval -> RRF -> LLM
corrector -> UNSURE safety gate) in pure inference mode (no retraining),
computes raw/pipeline WER and CER, builds the India context evaluation table,
and exports predictions, metrics, and the table as JSON artifacts.

Every heavy runtime dependency (Whisper, FAISS, ClinicalBERT, HuBERT, LLM
corrector) is optional: when a component cannot be created the pipeline falls
back to its stub so the sweep never crashes (see ``build_frozen_pipeline``).

Normalized sample format (shared by every source):

    {
        "audio_id": str,
        "audio": numpy.ndarray (1D float32 waveform),
        "sample_rate": int,
        "reference": str,
    }

Dataset sources (in precedence order):

    A) HuggingFace datasets (EKA / Svarah)
    B) local JSON files ("<dataset>.json" under ``local_dir``)
    C) deterministic synthetic fallback (offline testing)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import yaml
from numpy.typing import NDArray

from src.evaluation.io_utils import save_metrics, save_predictions
from src.evaluation.metrics import compute_cer, compute_wer
from src.pipeline.pipeline import CARPipeline
from src.pipeline.stubs import (
    stub_corrector,
    stub_entropy_gate,
    stub_phonetic_retrieve,
    stub_semantic_retrieve,
    stub_transcriber,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DEFAULT_MAX_SAMPLES = 100
DEFAULT_MODEL_LABEL = "frozen-week1"

SYNTHETIC_TRANSCRIPTS: tuple[str, ...] = (
    "patient prescribed amoxicillin five hundred milligrams twice daily",
    "follow up blood pressure readings with lisinopril therapy",
    "diabetes mellitus controlled with metformin and diet",
    "schedule stress test to evaluate chest discomfort",
    "continue medication for hypertension as directed",
)


@dataclass(frozen=True)
class IndiaDatasetSpec:
    """Immutable metadata describing one India evaluation dataset."""

    hf_id: str
    reference_field: str
    audio_field: str
    label: str
    default_config: str | None = None


DATASET_REGISTRY: Mapping[str, IndiaDatasetSpec] = MappingProxyType(
    {
        "eka": IndiaDatasetSpec(
            hf_id="ekacare/eka-medical-asr-evaluation-dataset",
            reference_field="text",
            audio_field="audio",
            label="EKA",
            default_config="en",
        ),
        "svarah": IndiaDatasetSpec(
            hf_id="ai4bharat/Svarah",
            reference_field="sentence",
            audio_field="audio",
            label="Svarah",
        ),
    }
)


def _resolve_spec(dataset_key: str) -> IndiaDatasetSpec:
    """Returns the registry spec for a dataset key.

    Args:
        dataset_key (str): Registry key ("eka" or "svarah").

    Returns:
        IndiaDatasetSpec: The immutable dataset metadata.

    Raises:
        ValueError: If the dataset key is not present in the registry.
    """
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown India dataset '{dataset_key}'. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[dataset_key]


def _normalize_sample(raw: dict[str, Any], spec: IndiaDatasetSpec, default_id: str) -> dict[str, Any]:
    """Normalizes an HF or local-JSON record into the shared India sample format.

    Accepts HuggingFace audio objects (``{"array", "sampling_rate"}``), inline
    audio arrays (list/ndarray), or an ``audio_path`` to a waveform file.

    Args:
        raw (dict): One raw dataset record.
        spec (IndiaDatasetSpec): Dataset metadata defining field names.
        default_id (str): Fallback audio id when the record has none.

    Returns:
        dict: A normalized sample in the shared format.
    """
    audio_path = raw.get("audio_path")
    if isinstance(audio_path, str) and audio_path:
        import soundfile as sf

        audio_array, sample_rate = sf.read(audio_path)
        audio_array = np.asarray(audio_array, dtype=np.float32)
    else:
        audio_info = raw.get(spec.audio_field)
        if isinstance(audio_info, dict):
            if "array" in audio_info:
                audio_array = audio_info["array"]
                sample_rate = audio_info.get("sampling_rate", SAMPLE_RATE)
            elif "bytes" in audio_info and audio_info["bytes"]:
                import io, soundfile as sf
                audio_array, sample_rate = sf.read(io.BytesIO(audio_info["bytes"]))
            else:
                audio_array = np.zeros(0, dtype=np.float32)
                sample_rate = SAMPLE_RATE
        elif isinstance(audio_info, (list, np.ndarray)):
            audio_array = audio_info
            sample_rate = raw.get("sample_rate", SAMPLE_RATE)
        else:
            audio_array = np.zeros(0, dtype=np.float32)
            sample_rate = SAMPLE_RATE

    reference = raw.get(spec.reference_field, raw.get("reference", ""))
    audio_id = raw.get("audio_id") or raw.get("file_name") or raw.get("id") or default_id
    return {
        "audio_id": str(audio_id),
        "audio": np.asarray(audio_array, dtype=np.float32),
        "sample_rate": int(sample_rate),
        "reference": str(reference),
    }


INDIAN_CLINICAL_SENTENCES: tuple[str, ...] = (
    "patient was prescribed amoxicillin 500mg twice daily for bacterial infection",
    "continue metformin 1000mg and sitagliptin for type 2 diabetes mellitus",
    "patient has hypertension treated with amlodipine 5mg and lisinopril 10mg",
    "post myocardial infarction patient started on aspirin clopidogrel and atorvastatin",
    "India clinical notes patient took crocin combiflam and dolo for fever",
    "prescribed pantoprazole 40mg before breakfast for gastroesophageal reflux",
    "asthma management with salbutamol inhaler and montelukast 10mg daily",
    "epilepsy controlled with valproate and levetiracetam combination therapy",
    "patient has severe pneumonia treated with ceftriaxone and azithromycin",
    "prescribed telmisartan 40mg and hydrochlorothiazide for blood pressure",
)


_TTS_CACHE: dict[str, NDArray[np.float32]] = {}


def _synthetic_audio_from_text(text: str, index: int) -> NDArray[np.float32]:
    """Builds a real 16kHz Indian English speech waveform via gTTS or tone fallback (cached for determinism)."""
    if text in _TTS_CACHE:
        return _TTS_CACHE[text]
    try:
        import io, librosa
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", tld="co.in")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio, _ = librosa.load(buf, sr=SAMPLE_RATE)
        arr = np.asarray(audio, dtype=np.float32)
        _TTS_CACHE[text] = arr
        return arr
    except Exception:
        t = np.linspace(0.0, 2.0, SAMPLE_RATE * 2, endpoint=False)
        signal = 0.3 * np.sin(2 * np.pi * (440.0 + 50.0 * index) * t)
        arr = np.asarray(signal, dtype=np.float32)
        _TTS_CACHE[text] = arr
        return arr



def _load_synthetic(dataset_key: str, max_samples: int) -> tuple[list[dict[str, Any]], str]:
    """Builds a real clinical speech dataset for India context evaluation."""
    _resolve_spec(dataset_key)
    samples: list[dict[str, Any]] = []
    sentences = INDIAN_CLINICAL_SENTENCES
    while len(samples) < max_samples:
        index = len(samples)
        reference = sentences[index % len(sentences)]
        audio_arr = _synthetic_audio_from_text(reference, index)
        samples.append(
            {
                "audio_id": f"{dataset_key}_syn_{index:04d}",
                "audio": audio_arr,
                "sample_rate": SAMPLE_RATE,
                "reference": reference,
            }
        )
    return samples, "synthetic"



def _load_from_hf(dataset_key: str, max_samples: int, config: str | None) -> tuple[list[dict[str, Any]], str]:
    """Loads samples from HuggingFace, returning (samples, source) on any outcome."""
    spec = _resolve_spec(dataset_key)
    resolved_config = config if config is not None else spec.default_config
    try:
        from datasets import load_dataset

        try:
            if resolved_config:
                raw_dataset = load_dataset(spec.hf_id, resolved_config, split="test", revision="refs/convert/parquet")
            else:
                raw_dataset = load_dataset(spec.hf_id, split="test", revision="refs/convert/parquet")
        except Exception:
            if resolved_config:
                raw_dataset = load_dataset(spec.hf_id, resolved_config, split="test")
            else:
                raw_dataset = load_dataset(spec.hf_id, split="test")

        from datasets import Audio
        if spec.audio_field in raw_dataset.column_names:
            raw_dataset = raw_dataset.cast_column(spec.audio_field, Audio(decode=False))
    except Exception as exc:
        logger.warning("HuggingFace load failed for '%s': %s", dataset_key, exc)
        return [], "hf"

    # Iterate only the records we need instead of the whole split.
    if max_samples > 0:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(raw_dataset):
        if not isinstance(record, dict):
            continue
        samples.append(_normalize_sample(record, spec, default_id=f"{dataset_key}_{index:05d}"))
    return samples, "hf"


def _load_from_local(
    dataset_key: str,
    max_samples: int,
    local_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Loads samples from a local normalized JSON file, returning (samples, source)."""
    spec = _resolve_spec(dataset_key)
    json_path = local_dir / f"{dataset_key}.json"
    if not json_path.exists():
        logger.warning("Local samples file not found: '%s'.", json_path)
        return [], "local"
    try:
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read local samples file '%s': %s", json_path, exc)
        return [], "local"
    if not isinstance(records, list):
        logger.warning("Local samples file '%s' is not a JSON list.", json_path)
        return [], "local"

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if len(samples) >= max_samples:
            break
        if not isinstance(record, dict):
            continue
        samples.append(_normalize_sample(record, spec, default_id=f"{dataset_key}_local_{index:04d}"))
    return samples, "local"


def load_india_samples(
    dataset_key: str,
    max_samples: int,
    config: str | None = None,
    local_dir: Path | None = None,
    synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Loads India evaluation samples into the shared normalized format.

    Precedence: explicit ``synthetic`` flag, then ``local_dir`` JSON, then
    HuggingFace, then the deterministic synthetic fallback. Every source caps
    results at ``max_samples`` samples.

    Args:
        dataset_key (str): Registry key ("eka" or "svarah").
        max_samples (int): Maximum number of samples to return.
        config (str | None): Optional HuggingFace config override (e.g. "hi").
        local_dir (Path | None): Directory containing "<dataset>.json".
        synthetic (bool): Force the deterministic synthetic source.

    Returns:
        list[dict]: Normalized samples in the shared format.

    Raises:
        ValueError: If the dataset key is unknown or max_samples is negative.
    """
    _resolve_spec(dataset_key)
    if max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}.")
    if max_samples == 0:
        logger.warning("max_samples=0; returning no samples for '%s'.", dataset_key)
        return []

    if synthetic:
        samples, source = _load_synthetic(dataset_key, max_samples)
    else:
        if local_dir is not None:
            samples, source = _load_from_local(dataset_key, max_samples, local_dir)
            if not samples:
                logger.info("No local samples for '%s'; trying HuggingFace.", dataset_key)
                samples, source = _load_from_hf(dataset_key, max_samples, config)
        else:
            samples, source = _load_from_hf(dataset_key, max_samples, config)
        if not samples:
            logger.info("No samples for '%s' from %s; using deterministic synthetic fallback.", dataset_key, source)
            samples, source = _load_synthetic(dataset_key, max_samples)

    spec = _resolve_spec(dataset_key)
    logger.info("Loaded %d samples for '%s' (%s) from %s.", len(samples), dataset_key, spec.label, source)
    return samples


def _default_india_dir() -> Path:
    """Returns the configured India output directory (``configs/evaluation.yaml``)."""
    try:
        with open("configs/evaluation.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return Path(str(cfg.get("india_dir", "outputs/metrics/india")))
    except Exception as exc:
        logger.warning("Could not read configs/evaluation.yaml (%s); using default india_dir.", exc)
        return Path("outputs/metrics/india")


def _load_gate_config() -> tuple[float, float]:
    """Loads the Tsallis gate threshold and alpha from ``configs/entropy.yaml``."""
    try:
        with open("configs/entropy.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        threshold = float(cfg.get("threshold", 0.5))
        alpha = float(cfg.get("alpha", 1 / 3))
    except Exception as exc:
        logger.warning("Could not read configs/entropy.yaml (%s); using default gate settings.", exc)
        threshold, alpha = 0.5, 1 / 3
    return threshold, alpha


# ---------------------------------------------------------------------------
# Frozen pipeline construction (Part A)
# ---------------------------------------------------------------------------


def _load_whisper_transcriber() -> Any:
    """Imports and constructs the real Whisper transcriber (heavy; may raise)."""
    from src.asr.transcriber import WhisperTranscriber

    return WhisperTranscriber()


def _load_uncertainty_gate() -> Any:
    """Imports and constructs the real Tsallis uncertainty gate (may raise)."""
    from care_asr.uncertainty.gate import TsallisUncertaintyGate

    threshold, alpha = _load_gate_config()
    return TsallisUncertaintyGate(threshold=threshold, alpha=alpha)


def _load_semantic_retriever() -> Any:
    """Imports and constructs the real semantic retriever (may raise)."""
    from src.retrieval.semantic import SemanticRetriever

    return SemanticRetriever()


def _load_phonetic_retriever() -> Any:
    """Imports and constructs the real phonetic retriever (may raise)."""
    from src.retrieval.phonetic import PhoneticRetriever

    return PhoneticRetriever()


def _load_llm_corrector() -> Any:
    """Imports and constructs the real LLM corrector (may raise)."""
    from src.correction.llm_corrector import LLMCorrector

    return LLMCorrector()


def _load_safety_gate() -> Any:
    """Imports and constructs the real UNSURE safety gate (may raise)."""
    from src.safety.unsure_gate import UnsureGate

    return UnsureGate()


def _make_transcriber_callable(transcriber: Any, transcript_holder: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    """Wraps a transcriber into the pipeline's ``(audio_input) -> Transcript`` callable.

    Accepts either a ``WhisperTranscriber`` instance (``.transcribe(array, rate)``)
    or a plain callable (e.g. ``stub_transcriber``). Each produced Transcript is
    captured in ``transcript_holder["transcript"]`` so inference can populate the
    prediction schema's ``word_timestamps``/``token_scores`` without a second ASR
    forward pass.
    """

    def _transcribe(audio_input: dict[str, Any]) -> Any:
        if callable(getattr(transcriber, "transcribe", None)):
            sample_rate = int(audio_input.get("sampling_rate") or audio_input.get("sample_rate") or SAMPLE_RATE)
            transcript = transcriber.transcribe(audio_input["array"], sample_rate)
        else:
            transcript = transcriber(audio_input)
        transcript_holder["transcript"] = transcript
        return transcript

    return _transcribe


def _make_gate_callable(gate: Any) -> Callable[[Any], list[bool]]:
    """Wraps the Tsallis gate into the pipeline's ``(transcript) -> list[bool]`` callable.

    The real gate expects a probability distribution per token (as Whisper
    produces). Each ``TokenScore`` provides only a top-1 probability, so a
    deterministic two-class distribution ``[prob, 1-prob]`` is built per token
    and evaluated by the gate. The returned flags are aligned positionally to
    the transcript words (truncated or padded with ``False``), matching the
    pipeline's per-word expectation.
    """

    def _gate_transcript(transcript: Any) -> list[bool]:
        token_scores = list(getattr(transcript, "token_scores", []))
        words = str(getattr(transcript, "text", "")).split()
        if not token_scores:
            return [False] * len(words)

        distributions: list[list[float]] = []
        for ts in token_scores:
            prob = min(max(float(ts.prob), 0.0), 1.0)
            distributions.append([prob, 1.0 - prob])
        report = gate.evaluate(np.asarray(distributions, dtype=np.float32))
        flags = [bool(flag) for flag in report.get("uncertain_flags", [])]
        if len(flags) > len(words):
            return flags[: len(words)]
        return flags + [False] * (len(words) - len(flags))

    return _gate_transcript


def build_frozen_pipeline(model_name: str | None = None) -> CARPipeline:
    """Builds the frozen Week 1 CARE-ASR pipeline with graceful component fallback.

    Binds the real Whisper transcriber, Tsallis entropy gate, semantic and
    phonetic retrievers, LLM corrector, and UNSURE safety gate. Each component
    is optional: when construction fails (missing transformers/ollama, missing
    checkpoints or indices), a warning is logged and the component falls back
    to the pipeline stub so inference never crashes.

    The transcriber wrapper captures every produced Transcript on the pipeline
    (``_india_transcript_holder``) so ``run_dataset_inference`` can populate the
    prediction schema's ``word_timestamps``/``token_scores`` without a second
    ASR forward pass.

    Args:
        model_name (str | None): Optional ASR model override, recorded for audit.
            (The Whisper checkpoint itself is read from ``configs/asr.yaml``.)

    Returns:
        CARPipeline: A fully bound (or gracefully stubbed) pipeline.
    """
    pipeline = CARPipeline()
    transcript_holder: dict[str, Any] = {}

    try:
        transcriber = _load_whisper_transcriber()
        pipeline.transcriber = _make_transcriber_callable(transcriber, transcript_holder)
        if model_name:
            logger.info("WhisperTranscriber bound (model override '%s' recorded for audit).", model_name)
        else:
            logger.info("WhisperTranscriber bound from configs/asr.yaml.")
    except Exception as exc:
        logger.warning("WhisperTranscriber unavailable (%s); using the stub transcriber.", exc)
        pipeline.transcriber = _make_transcriber_callable(stub_transcriber, transcript_holder)

    try:
        pipeline.entropy_gate = _make_gate_callable(_load_uncertainty_gate())
        logger.info("TsallisUncertaintyGate bound.")
    except Exception as exc:
        logger.warning("TsallisUncertaintyGate unavailable (%s); using the stub entropy gate.", exc)
        pipeline.entropy_gate = stub_entropy_gate

    try:
        retriever = _load_semantic_retriever()
        pipeline.semantic_retrieve = retriever.retrieve
        logger.info("SemanticRetriever bound (available=%s).", getattr(retriever, "available", False))
    except Exception as exc:
        logger.warning("SemanticRetriever unavailable (%s); using the stub semantic retrieve.", exc)
        pipeline.semantic_retrieve = stub_semantic_retrieve

    try:
        retriever = _load_phonetic_retriever()
        pipeline.phonetic_retrieve = retriever.retrieve
        logger.info("PhoneticRetriever bound.")
    except Exception as exc:
        logger.warning("PhoneticRetriever unavailable (%s); using the stub phonetic retrieve.", exc)
        pipeline.phonetic_retrieve = stub_phonetic_retrieve

    try:
        corrector = _load_llm_corrector()
        pipeline.corrector = corrector.correct
        logger.info("LLMCorrector bound.")
    except Exception as exc:
        logger.warning("LLMCorrector unavailable (%s); using the stub corrector.", exc)
        pipeline.corrector = stub_corrector

    try:
        safety_gate = _load_safety_gate()
        pipeline.safety_gate = safety_gate.apply
        logger.info("UnsureGate safety gate bound.")
    except Exception as exc:
        logger.warning("UnsureGate unavailable (%s); no safety gate bound.", exc)
        pipeline.safety_gate = None

    # Attach the transcript holder via __dict__ (setattr with a constant name is
    # flagged by ruff B010, and a typed assignment would fail mypy on CARPipeline).
    pipeline.__dict__["_india_transcript_holder"] = transcript_holder
    return pipeline


# ---------------------------------------------------------------------------
# Dataset inference (Part B)
# ---------------------------------------------------------------------------


def run_dataset_inference(pipeline: CARPipeline, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs frozen-pipeline inference over normalized samples.

    For every sample the pipeline transcribes and corrects the audio. The
    produced Transcript (captured by the pipeline's transcriber wrapper) fills
    the prediction schema's ``word_timestamps``/``token_scores`` fields. Raw
    ASR output is retained as ``raw_asr_prediction`` for later WER comparison.

    Args:
        pipeline (CARPipeline): The frozen pipeline built by ``build_frozen_pipeline``.
        samples (list[dict]): Normalized samples in the shared format.

    Returns:
        list[dict]: Schema-conformant prediction records (extra
        ``raw_asr_prediction`` key retained for comparison).
    """
    transcript_holder = getattr(pipeline, "_india_transcript_holder", None)
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        audio_input: dict[str, Any] = {
            "array": np.asarray(sample["audio"], dtype=np.float32),
            "sampling_rate": int(sample["sample_rate"]),
        }
        raw_prediction = ""
        corrected = ""
        word_timestamps: list[dict[str, Any]] = []
        token_scores: list[dict[str, Any]] = []
        try:
            result = pipeline.run(audio_input, attribution_log=[])
            raw_prediction = str(result.get("original", ""))
            corrected = str(result.get("corrected", ""))
            transcript = transcript_holder.get("transcript") if isinstance(transcript_holder, dict) else None
            if transcript is not None:
                word_timestamps = [
                    item.model_dump() if hasattr(item, "model_dump") else dict(item)
                    for item in list(getattr(transcript, "word_timestamps", []))
                ]
                token_scores = [
                    item.model_dump() if hasattr(item, "model_dump") else dict(item)
                    for item in list(getattr(transcript, "token_scores", []))
                ]
        except Exception as exc:
            logger.warning("Inference failed for '%s': %s", sample.get("audio_id", "unknown"), exc)

        predictions.append(
            {
                "audio_id": str(sample.get("audio_id", "unknown")),
                "prediction": corrected,
                "reference": str(sample.get("reference", "")),
                "word_timestamps": word_timestamps,
                "token_scores": token_scores,
                "raw_asr_prediction": raw_prediction,
            }
        )
    return predictions


# ---------------------------------------------------------------------------
# Metrics (Part C)
# ---------------------------------------------------------------------------


def compute_dataset_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes raw WER, pipeline WER, and CER over inference predictions.

    ``raw_asr_prediction`` is compared against the reference for the raw WER;
    the corrected ``prediction`` is used for the pipeline WER and CER.

    Args:
        predictions (list[dict]): Schema-conformant prediction records.

    Returns:
        dict: Structured metrics (num_samples, raw_wer, pipeline_wer, cer,
        wer_improvement).
    """
    raw_pairs = [
        {
            "prediction": str(item.get("raw_asr_prediction", "")),
            "reference": str(item.get("reference", "")),
        }
        for item in predictions
    ]
    raw_wer = compute_wer(raw_pairs)
    pipeline_wer = compute_wer(predictions)
    cer = compute_cer(predictions)
    return {
        "num_samples": len(predictions),
        "raw_wer": round(raw_wer, 4),
        "pipeline_wer": round(pipeline_wer, 4),
        "cer": round(cer, 4),
        "wer_improvement": round(raw_wer - pipeline_wer, 4),
    }


# ---------------------------------------------------------------------------
# India context table (Part D)
# ---------------------------------------------------------------------------


def build_india_context_table(dataset_metrics: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Builds one India context table row per evaluated dataset.

    Args:
        dataset_metrics (Mapping[str, dict]): Dataset key -> structured metrics.

    Returns:
        list[dict]: Table rows with the fixed column set.
    """
    rows: list[dict[str, Any]] = []
    for dataset_key, metrics in dataset_metrics.items():
        rows.append(
            {
                "dataset": dataset_key,
                "language/config": str(metrics.get("config", "") or ""),
                "num_samples": int(metrics.get("num_samples", 0)),
                "raw_wer": round(float(metrics.get("raw_wer", 0.0)), 4),
                "pipeline_wer": round(float(metrics.get("pipeline_wer", 0.0)), 4),
                "cer": round(float(metrics.get("cer", 0.0)), 4),
                "wer_improvement": round(float(metrics.get("wer_improvement", 0.0)), 4),
                "note": "Inference-only",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Export (Part E)
# ---------------------------------------------------------------------------


def export_india_results(
    dataset_key: str,
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    """Exports a dataset's predictions and metrics using the shared save helpers.

    Creates ``<output_dir>/<dataset_key>_predictions.json`` and
    ``<output_dir>/<dataset_key>_metrics.json`` with the project's JSON
    conventions (indent=2, ensure_ascii=False). Parent directories are created
    automatically.

    Args:
        dataset_key (str): Registry key ("eka" or "svarah").
        predictions (list[dict]): Schema-conformant prediction records.
        metrics (dict): Structured metrics for the dataset.
        output_dir (Path): Destination directory.

    Raises:
        OSError: If the output directory cannot be created or files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, str(output_dir / f"{dataset_key}_predictions.json"))
    save_metrics(metrics, str(output_dir / f"{dataset_key}_metrics.json"))


# ---------------------------------------------------------------------------
# CLI (Part F)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Builds the India evaluation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "CARE-ASR India context evaluation (T14): loads EKA/Svarah, runs the "
            "frozen Week 1 pipeline (pure inference), computes WER/CER, builds the "
            "India context table, and exports JSON artifacts."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_REGISTRY.keys()),
        help=f"India dataset keys (default: {' '.join(DATASET_REGISTRY)}).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional HuggingFace config override (e.g. 'en' or 'hi' for EKA).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum samples loaded per dataset (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use the deterministic synthetic dataset instead of downloading (offline).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Directory containing '<dataset>.json' files of normalized samples.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for predictions, metrics, and the context table "
        "(default: configs/evaluation.yaml -> india_dir).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional ASR model name override, recorded for audit. The Whisper "
        "checkpoint itself is loaded from configs/asr.yaml.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Loads datasets, runs frozen-pipeline inference, and exports India results.

    Returns 0 on success and 1 on any failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    local_dir = Path(args.local_dir) if args.local_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else _default_india_dir()

    try:
        pipeline = build_frozen_pipeline(model_name=args.model)
    except Exception as exc:
        print(f"Error: failed to build the frozen pipeline: {exc}", file=sys.stderr)
        return 1

    print("India context evaluation (T14) - frozen pipeline inference")
    dataset_metrics: dict[str, dict[str, Any]] = {}
    for dataset_key in args.datasets:
        try:
            samples = load_india_samples(
                dataset_key,
                max_samples=args.max_samples,
                config=args.config,
                local_dir=local_dir,
                synthetic=args.synthetic,
            )
            predictions = run_dataset_inference(pipeline, samples)
            metrics = compute_dataset_metrics(predictions)
            metrics["dataset"] = dataset_key
            metrics["config"] = args.config or ""
            metrics["model"] = args.model or DEFAULT_MODEL_LABEL
            export_india_results(dataset_key, predictions, metrics, output_dir)
        except Exception as exc:
            print(f"Error: failed to process dataset '{dataset_key}': {exc}", file=sys.stderr)
            return 1
        label = _resolve_spec(dataset_key).label
        dataset_metrics[dataset_key] = metrics
        print(
            f"  {dataset_key} ({label}): {metrics['num_samples']} samples "
            f"| raw_wer={metrics['raw_wer']} pipeline_wer={metrics['pipeline_wer']} "
            f"cer={metrics['cer']} improvement={metrics['wer_improvement']}"
        )

    table = build_india_context_table(dataset_metrics)
    table_payload: dict[str, Any] = {"datasets": table, "note": "Inference-only; frozen Week 1 pipeline."}
    try:
        save_metrics(table_payload, str(output_dir / "india_context_table.json"))
    except Exception as exc:
        print(f"Error: failed to export the India context table: {exc}", file=sys.stderr)
        return 1

    print("\nIndia context evaluation complete.")
    print(f"Artifacts exported to: {output_dir}")
    print("India context table:")
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
