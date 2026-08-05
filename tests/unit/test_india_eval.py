"""Unit tests for the T14 India context evaluation script (Commits 1-4)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not a package; add it to the import path so the script's module
# functions can be unit-tested without a conftest.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_india_eval as india_eval  # noqa: E402

from care_asr.contracts.asr_input import TokenScore, Transcript, WordTimestamp  # noqa: E402
from care_asr.contracts.validated_output import CorrectionOutput  # noqa: E402
from src.evaluation.io_utils import load_predictions, validate_prediction_schema  # noqa: E402
from src.pipeline.pipeline import CARPipeline  # noqa: E402

NORMALIZED_KEYS = {"audio_id", "audio", "sample_rate", "reference"}
TABLE_COLUMNS = {
    "dataset",
    "language/config",
    "num_samples",
    "raw_wer",
    "pipeline_wer",
    "cer",
    "wer_improvement",
    "note",
}


@pytest.fixture(autouse=True)
def _hermetic_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stubs every heavy component loader so all tests run offline and deterministically.

    Without this, CLI end-to-end tests would construct the real WhisperTranscriber,
    retrievers, and LLM corrector, requiring Whisper/Ollama/FAISS/internet.
    Individual tests may override individual seams via their own monkeypatch.
    """
    monkeypatch.setattr(india_eval, "_load_whisper_transcriber", lambda: _FakeTranscriber())
    monkeypatch.setattr(india_eval, "_load_uncertainty_gate", lambda: _FakeGate())
    monkeypatch.setattr(india_eval, "_load_semantic_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(india_eval, "_load_phonetic_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(india_eval, "_load_llm_corrector", lambda: _FakeCorrector())
    monkeypatch.setattr(india_eval, "_load_safety_gate", lambda: _FakeSafetyGate())


class _FakeTranscriber:
    """Deterministic WhisperTranscriber stand-in with .transcribe()."""

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Transcript:
        return Transcript(
            text="alpha beta",
            token_scores=[
                TokenScore(step=0, token_id=1, token="alpha", log_prob=-0.1, prob=0.9),
                TokenScore(step=1, token_id=2, token="beta", log_prob=-0.1, prob=0.9),
            ],
            word_timestamps=[WordTimestamp(word="alpha", start=0.0, end=0.4)],
        )


class _FakeGate:
    """Deterministic TsallisUncertaintyGate stand-in."""

    def evaluate(self, probs: np.ndarray) -> dict[str, list[bool]]:
        return {"uncertain_flags": [False, True]}


class _FakeRetriever:
    """Retriever stand-in that never produces candidates."""

    available = True

    def retrieve(self, token: str, top_k: int = 5) -> list[object]:
        return []


class _FakeCorrector:
    """LLMCorrector stand-in that keeps the token unchanged."""

    def correct(self, token: str, candidates: list[object], context: str = "") -> CorrectionOutput:
        return CorrectionOutput(original_token=token, corrected_token=token, label="CORRECT", confidence=0.9)


class _FakeSafetyGate:
    """UnsureGate stand-in that passes corrections through."""

    def apply(self, correction: CorrectionOutput) -> CorrectionOutput:
        return correction


# ---------------------------------------------------------------------------
# Commit 1: dataset loading layer
# ---------------------------------------------------------------------------


def test_registry_contains_expected_datasets() -> None:
    """The registry defines EKA and Svarah with their dataset metadata."""
    assert set(india_eval.DATASET_REGISTRY) == {"eka", "svarah"}
    eka = india_eval.DATASET_REGISTRY["eka"]
    assert eka.hf_id == "ekacare/eka-medical-asr-evaluation-dataset"
    assert eka.default_config == "en"
    assert eka.reference_field == "text"
    assert eka.audio_field == "audio"
    assert eka.label == "EKA"
    svarah = india_eval.DATASET_REGISTRY["svarah"]
    assert svarah.hf_id == "ai4bharat/Svarah"
    assert svarah.reference_field == "sentence"
    assert svarah.label == "Svarah"


def test_invalid_dataset_raises_value_error() -> None:
    """An unknown dataset key raises ValueError."""
    with pytest.raises(ValueError):
        india_eval.load_india_samples("bogus", max_samples=5, synthetic=True)


def test_normalized_schema() -> None:
    """Loaded samples follow the shared normalized schema."""
    samples = india_eval.load_india_samples("eka", max_samples=3, synthetic=True)
    assert len(samples) == 3
    for sample in samples:
        assert set(sample) == NORMALIZED_KEYS
        assert isinstance(sample["audio_id"], str)
        assert isinstance(sample["audio"], np.ndarray)
        assert sample["audio"].dtype == np.float32
        assert isinstance(sample["sample_rate"], int)
        assert isinstance(sample["reference"], str)


def test_respects_max_samples() -> None:
    """max_samples caps the number of loaded samples."""
    assert len(india_eval.load_india_samples("svarah", max_samples=2, synthetic=True)) == 2
    assert india_eval.load_india_samples("eka", max_samples=0, synthetic=True) == []


def test_synthetic_fallback_deterministic() -> None:
    """Repeated synthetic loads produce identical samples."""
    first = india_eval.load_india_samples("eka", max_samples=5, synthetic=True)
    second = india_eval.load_india_samples("eka", max_samples=5, synthetic=True)
    assert [sample["audio_id"] for sample in first] == [sample["audio_id"] for sample in second]
    assert [sample["reference"] for sample in first] == [sample["reference"] for sample in second]
    assert all(
        np.array_equal(audio_a, audio_b)
        for audio_a, audio_b in zip(
            (sample["audio"] for sample in first),
            (sample["audio"] for sample in second),
            strict=True,
        )
    )


def test_local_json_loading(tmp_path: Path) -> None:
    """Samples are loaded from a local normalized JSON file."""
    records = [
        {
            "audio_id": "l_001",
            "audio": [0.1, 0.2, 0.3],
            "sample_rate": 16000,
            "reference": "patient stable",
        },
        {
            "audio_id": "l_002",
            "audio": [0.4, 0.5],
            "sample_rate": 16000,
            "reference": "continue metformin",
        },
    ]
    (tmp_path / "eka.json").write_text(json.dumps(records), encoding="utf-8")

    samples = india_eval.load_india_samples("eka", max_samples=10, local_dir=tmp_path)

    assert len(samples) == 2
    assert samples[0]["audio_id"] == "l_001"
    assert samples[0]["reference"] == "patient stable"
    assert samples[0]["audio"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert samples[1]["audio_id"] == "l_002"
    assert samples[1]["audio"].tolist() == pytest.approx([0.4, 0.5])


def test_cli_summary_executes(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """The CLI loads datasets, runs inference, and prints a summary."""
    exit_code = india_eval.main(
        ["--synthetic", "--datasets", "eka", "--max-samples", "2", "--output-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "eka" in captured.out
    assert "2 samples" in captured.out


def test_empty_dataset_handled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An empty local/HF source falls back to synthetic samples without crashing."""
    monkeypatch.setattr(india_eval, "_load_from_hf", lambda *args, **kwargs: ([], "hf"))
    samples = india_eval.load_india_samples("eka", max_samples=5, local_dir=tmp_path / "missing")
    assert isinstance(samples, list)
    assert len(samples) == 5  # deterministic synthetic fallback
    assert all(set(sample) == NORMALIZED_KEYS for sample in samples)


# ---------------------------------------------------------------------------
# Commit 2: frozen pipeline construction + inference
# ---------------------------------------------------------------------------


def test_build_frozen_pipeline_graceful_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable heavy components fall back to stubs without crashing."""

    def _raise(message: str):
        def _loader() -> object:
            raise RuntimeError(message)

        return _loader

    monkeypatch.setattr(india_eval, "_load_whisper_transcriber", _raise("transformers unavailable"))
    monkeypatch.setattr(india_eval, "_load_uncertainty_gate", _raise("torch unavailable"))
    monkeypatch.setattr(india_eval, "_load_semantic_retriever", _raise("faiss unavailable"))
    monkeypatch.setattr(india_eval, "_load_phonetic_retriever", _raise("faiss unavailable"))
    monkeypatch.setattr(india_eval, "_load_llm_corrector", _raise("ollama unavailable"))
    monkeypatch.setattr(india_eval, "_load_safety_gate", _raise("safety unavailable"))

    pipeline = india_eval.build_frozen_pipeline()

    # Transcriber falls back to a wrapper over the stub; still produces a Transcript.
    transcript = pipeline.transcriber({"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000})
    assert transcript.text == "patient prescribed amoxicillin five hundred milligrams"
    # Gate falls back to the stub; still returns boolean flags.
    flags = pipeline.entropy_gate(transcript)
    assert isinstance(flags, list) and all(isinstance(flag, bool) for flag in flags)
    # Correctors still produce the canonical CorrectionOutput.
    correction = pipeline.corrector("amoxicilin", [])
    assert isinstance(correction, CorrectionOutput)
    # No safety gate was bound.
    assert pipeline.safety_gate is None


def test_build_frozen_pipeline_binds_real_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successfully constructed components are bound to the pipeline."""
    monkeypatch.setattr(india_eval, "_load_whisper_transcriber", lambda: _FakeTranscriber())
    monkeypatch.setattr(india_eval, "_load_uncertainty_gate", lambda: _FakeGate())
    monkeypatch.setattr(india_eval, "_load_semantic_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(india_eval, "_load_phonetic_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(india_eval, "_load_llm_corrector", lambda: _FakeCorrector())
    monkeypatch.setattr(india_eval, "_load_safety_gate", lambda: _FakeSafetyGate())

    pipeline = india_eval.build_frozen_pipeline(model_name="openai/whisper-small")

    audio = {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}
    transcript = pipeline.transcriber(audio)
    assert transcript.text == "alpha beta"
    assert getattr(pipeline, "_india_transcript_holder", {}).get("transcript") is not None
    flags = pipeline.entropy_gate(transcript)
    assert flags == [False, True]
    assert pipeline.semantic_retrieve("beta") == []
    correction = pipeline.corrector("beta", [])
    assert isinstance(correction, CorrectionOutput)
    assert pipeline.safety_gate(correction) is correction


def test_gate_flags_aligned_to_words() -> None:
    """Gate flags are truncated/padded to match the transcript word count."""

    class _OneFlagGate:
        def evaluate(self, probs: np.ndarray) -> dict[str, list[bool]]:
            return {"uncertain_flags": [True]}

    gate = india_eval._make_gate_callable(_OneFlagGate())
    one_token = Transcript(
        text="alpha beta",
        token_scores=[TokenScore(step=0, token_id=1, token="alpha", log_prob=0.0, prob=0.1)],
        word_timestamps=[],
    )
    assert gate(one_token) == [True, False]  # padded to two words

    empty = Transcript(text="alpha beta", token_scores=[], word_timestamps=[])
    assert india_eval._make_gate_callable(_OneFlagGate())(empty) == [False, False]


def test_run_dataset_inference_schema() -> None:
    """Inference produces schema-conformant predictions with raw ASR output."""
    pipeline = CARPipeline()

    def _fixed_transcriber(audio_input: object) -> Transcript:
        return Transcript(
            text="patient stable",
            token_scores=[TokenScore(step=0, token_id=1, token="patient", log_prob=-0.1, prob=0.9)],
            word_timestamps=[WordTimestamp(word="patient", start=0.0, end=0.5)],
        )

    pipeline.transcriber = _fixed_transcriber
    pipeline.entropy_gate = lambda transcript: [False, False]
    pipeline.__dict__["_india_transcript_holder"] = {"transcript": _fixed_transcriber(None)}

    samples = [
        {
            "audio_id": "s_001",
            "audio": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
            "reference": "patient stable",
        },
        {
            "audio_id": "s_002",
            "audio": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
            "reference": "patient stable",
        },
    ]
    predictions = india_eval.run_dataset_inference(pipeline, samples)
    assert len(predictions) == 2
    for prediction in predictions:
        validate_prediction_schema(prediction)
        assert prediction["audio_id"] in {"s_001", "s_002"}
        assert prediction["raw_asr_prediction"] == "patient stable"
        assert prediction["prediction"] == "patient stable"
        assert prediction["word_timestamps"] == [{"word": "patient", "start": 0.0, "end": 0.5}]
        assert prediction["token_scores"][0]["token"] == "patient"


# ---------------------------------------------------------------------------
# Commit 3: metrics + India context table
# ---------------------------------------------------------------------------


def test_export_uses_save_predictions(tmp_path: Path) -> None:
    """Export reuses the shared save_predictions/save_metrics helpers."""
    predictions = [
        {
            "audio_id": "a_1",
            "prediction": "amoxicillin five hundred",
            "reference": "amoxicillin five hundred",
            "word_timestamps": [],
            "token_scores": [],
            "raw_asr_prediction": "amoxicilin five hundred",
        }
    ]
    india_eval.export_india_results("eka", predictions, {"num_samples": 1}, tmp_path)

    pred_file = tmp_path / "eka_predictions.json"
    metrics_file = tmp_path / "eka_metrics.json"
    assert pred_file.exists()
    assert metrics_file.exists()
    loaded = load_predictions(str(pred_file))
    assert loaded[0]["audio_id"] == "a_1"
    assert loaded[0]["raw_asr_prediction"] == "amoxicilin five hundred"


def test_compute_dataset_metrics() -> None:
    """Raw WER, pipeline WER, CER, and improvement are computed from predictions."""
    predictions = [
        {
            "audio_id": "a",
            "prediction": "patient stable",
            "reference": "patient stable",
            "word_timestamps": [],
            "token_scores": [],
            "raw_asr_prediction": "patient stabel",
        },
        {
            "audio_id": "b",
            "prediction": "continue meds",
            "reference": "continue meds",
            "word_timestamps": [],
            "token_scores": [],
            "raw_asr_prediction": "continue med",
        },
        {
            "audio_id": "c",
            "prediction": "stop",
            "reference": "stop",
            "word_timestamps": [],
            "token_scores": [],
            "raw_asr_prediction": "stop now",
        },
    ]
    metrics = india_eval.compute_dataset_metrics(predictions)
    assert metrics["num_samples"] == 3
    # 3 errors over 5 reference words (1 substitution + 1 deletion + 1 insertion).
    assert metrics["raw_wer"] == pytest.approx(0.6)
    assert metrics["pipeline_wer"] == 0.0
    assert metrics["cer"] == 0.0
    assert metrics["wer_improvement"] == pytest.approx(0.6)


def test_build_india_context_table() -> None:
    """The context table has the fixed column set and one row per dataset."""
    dataset_metrics = {
        "eka": {
            "config": "en",
            "num_samples": 10,
            "raw_wer": 0.5,
            "pipeline_wer": 0.4,
            "cer": 0.2,
            "wer_improvement": 0.1,
        },
        "svarah": {
            "config": "",
            "num_samples": 5,
            "raw_wer": 0.6,
            "pipeline_wer": 0.5,
            "cer": 0.3,
            "wer_improvement": 0.1,
        },
    }
    table = india_eval.build_india_context_table(dataset_metrics)
    assert len(table) == 2
    for row in table:
        assert set(row) == TABLE_COLUMNS
        assert row["note"] == "Inference-only"
    assert table[0]["dataset"] == "eka"
    assert table[0]["language/config"] == "en"
    assert table[0]["num_samples"] == 10
    assert table[1]["dataset"] == "svarah"
    assert table[1]["language/config"] == ""


# ---------------------------------------------------------------------------
# Commit 4: CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_full_run_exports(tmp_path: Path) -> None:
    """A full synthetic CLI run exports predictions, metrics, and the table."""
    exit_code = india_eval.main(
        ["--synthetic", "--datasets", "eka", "--max-samples", "2", "--output-dir", str(tmp_path)]
    )
    assert exit_code == 0
    assert (tmp_path / "eka_predictions.json").exists()
    assert (tmp_path / "eka_metrics.json").exists()
    assert (tmp_path / "india_context_table.json").exists()
    metrics = json.loads((tmp_path / "eka_metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_samples"] == 2
    assert metrics["model"] == "frozen-week1"
    table = json.loads((tmp_path / "india_context_table.json").read_text(encoding="utf-8"))
    assert table["datasets"][0]["dataset"] == "eka"
    assert table["datasets"][0]["note"] == "Inference-only"


def test_cli_deterministic(tmp_path: Path) -> None:
    """Repeated synthetic runs produce identical metrics JSON."""
    first = tmp_path / "run1"
    second = tmp_path / "run2"
    assert india_eval.main(["--synthetic", "--datasets", "eka", "--max-samples", "3", "--output-dir", str(first)]) == 0
    assert india_eval.main(["--synthetic", "--datasets", "eka", "--max-samples", "3", "--output-dir", str(second)]) == 0
    first_metrics = json.loads((first / "eka_metrics.json").read_text(encoding="utf-8"))
    second_metrics = json.loads((second / "eka_metrics.json").read_text(encoding="utf-8"))
    assert first_metrics == second_metrics


def test_cli_invalid_dataset_exit_1() -> None:
    """An unknown dataset makes the CLI exit with code 1."""
    assert india_eval.main(["--synthetic", "--datasets", "bogus", "--max-samples", "2"]) == 1


def test_cli_invalid_output_path_exit_1(tmp_path: Path) -> None:
    """An output path that is an existing file makes the CLI exit with code 1."""
    blocker = tmp_path / "blocker.json"
    blocker.write_text("{}", encoding="utf-8")
    assert (
        india_eval.main(["--synthetic", "--datasets", "eka", "--max-samples", "1", "--output-dir", str(blocker)]) == 1
    )
