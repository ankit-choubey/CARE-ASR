"""Unit tests for T12 latency instrumentation (Commit 3)."""

from typing import Any

import pytest

from care_asr.contracts.retrieval_input import RetrievalCandidate
from src.pipeline.pipeline import CARPipeline
from src.retrieval.latency import LatencyStats

LATENCY_FIELDS = (
    "gate_latency_ms",
    "semantic_retrieval_latency_ms",
    "phonetic_retrieval_latency_ms",
    "retrieval_latency_ms",
    "fusion_latency_ms",
)


class FakeBatchRetriever:
    """Retriever exposing both retrieve() and retrieve_many() for batching tests."""

    def __init__(self) -> None:
        self.retrieve_calls = 0
        self.retrieve_many_calls: list[list[str]] = []
        self.last_top_k: int | None = None

    def retrieve(self, token: str, top_k: int = 5) -> list[RetrievalCandidate]:
        self.retrieve_calls += 1
        return [RetrievalCandidate(candidate=token, score=1.0, source="semantic")]

    def retrieve_many(self, tokens: list[str], top_k: int = 5) -> list[list[RetrievalCandidate]]:
        self.retrieve_many_calls.append(list(tokens))
        self.last_top_k = top_k
        return [[RetrievalCandidate(candidate=token, score=1.0, source="semantic")] for token in tokens]


def latency_entry(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Extracts the LATENCY attribution entry from a pipeline log."""
    return next(entry for entry in log if entry.get("module") == "LATENCY")


# --- LatencyStats unit tests -------------------------------------------------


def test_stats_records_values() -> None:
    """record() appends samples that values() returns as independent copies."""
    stats = LatencyStats()
    stats.record("op", 1.5)
    stats.record("op", 2.5)
    assert stats.values("op") == [1.5, 2.5]
    assert stats.values("missing") == []
    first = stats.values("op")
    first.append(99.0)
    assert stats.values("op") == [1.5, 2.5]


def test_stats_start_stop_records_elapsed() -> None:
    """start()/stop() records a non-negative elapsed duration in ms."""
    stats = LatencyStats()
    stats.start("op")
    elapsed = stats.stop("op")
    assert elapsed >= 0.0
    assert stats.values("op") == [elapsed]


def test_stats_stop_without_start_raises() -> None:
    """stop() without a matching start() raises ValueError."""
    stats = LatencyStats()
    with pytest.raises(ValueError):
        stats.stop("op")


def test_stats_summary_returns_means() -> None:
    """summary() returns the mean latency per recorded label."""
    stats = LatencyStats()
    stats.record("a", 1.0)
    stats.record("a", 3.0)
    stats.record("b", 4.0)
    assert stats.summary() == {"a": 2.0, "b": 4.0}


def test_stats_timed_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timed() context manager records the wrapped block's elapsed ms."""
    stats = LatencyStats()
    ticks = iter([100.0, 100.5])
    monkeypatch.setattr("src.retrieval.latency.time.perf_counter", lambda: next(ticks))
    with stats.timed("op"):
        pass
    assert stats.values("op") == [500.0]


# --- Pipeline instrumentation tests ------------------------------------------


def test_attribution_log_contains_latency_fields() -> None:
    """Each transcript's attribution log gains the five latency fields."""
    pipeline = CARPipeline()
    log: list[dict[str, Any]] = []
    pipeline.run("fake_audio.wav", attribution_log=log)
    entry = latency_entry(log)
    for field in LATENCY_FIELDS:
        assert field in entry
        assert isinstance(entry[field], float)
        assert entry[field] >= 0.0


def test_timing_values_are_non_negative() -> None:
    """All recorded latency samples are non-negative milliseconds."""
    pipeline = CARPipeline()
    pipeline.run("fake_audio.wav")
    assert pipeline.stats.names()
    for name in pipeline.stats.names():
        assert all(value >= 0.0 for value in pipeline.stats.values(name))


def test_retrieve_many_path_used_when_available() -> None:
    """Batched retrieval is used when the bound retriever exposes retrieve_many."""
    semantic_fake = FakeBatchRetriever()
    phonetic_fake = FakeBatchRetriever()
    pipeline = CARPipeline()
    pipeline.semantic_retrieve = semantic_fake.retrieve
    pipeline.phonetic_retrieve = phonetic_fake.retrieve
    pipeline.run("fake_audio.wav")
    assert semantic_fake.retrieve_many_calls == [["amoxicillin"]]
    assert phonetic_fake.retrieve_many_calls == [["amoxicillin"]]
    assert semantic_fake.retrieve_calls == 0
    assert phonetic_fake.retrieve_calls == 0


def test_sequential_fallback_still_works() -> None:
    """Per-token retrieve() is used when no batched retrieve_many exists."""
    seen: list[str] = []

    def single(token: str) -> list[RetrievalCandidate]:
        seen.append(token)
        return [RetrievalCandidate(candidate=token, score=1.0, source="semantic")]

    pipeline = CARPipeline()
    pipeline.semantic_retrieve = single
    pipeline.phonetic_retrieve = single
    result = pipeline.run("fake_audio.wav")
    assert seen == ["amoxicillin", "amoxicillin"]  # once per retrieval channel
    assert result["corrected"] == result["original"]


def test_retrieval_top_k_forwarded_to_batch_path() -> None:
    """retrieval_top_k is forwarded to retrieve_many when set."""
    fake = FakeBatchRetriever()
    pipeline = CARPipeline()
    pipeline.semantic_retrieve = fake.retrieve
    pipeline.phonetic_retrieve = fake.retrieve
    pipeline.retrieval_top_k = 3
    pipeline.run("fake_audio.wav")
    assert fake.last_top_k == 3


def test_retrieval_top_k_forwarded_to_fallback_path() -> None:
    """retrieval_top_k is forwarded to per-token retrieve() in the fallback."""
    seen: list[tuple[str, int]] = []

    def single(token: str, top_k: int = 5) -> list[RetrievalCandidate]:
        seen.append((token, top_k))
        return [RetrievalCandidate(candidate=token, score=1.0, source="semantic")]

    pipeline = CARPipeline()
    pipeline.semantic_retrieve = single
    pipeline.phonetic_retrieve = single
    pipeline.retrieval_top_k = 2
    pipeline.run("fake_audio.wav")
    assert seen == [("amoxicillin", 2), ("amoxicillin", 2)]  # once per retrieval channel


def test_pipeline_output_unchanged() -> None:
    """Timing only appends a LATENCY entry; the pipeline result is unchanged."""
    first = CARPipeline().run("fake_audio.wav")
    second = CARPipeline().run("fake_audio.wav")
    assert first["original"] == second["original"]
    assert first["corrected"] == second["corrected"]
    non_timing_first = [e for e in first["attribution"] if e.get("module") != "LATENCY"]
    non_timing_second = [e for e in second["attribution"] if e.get("module") != "LATENCY"]
    assert non_timing_first == non_timing_second


def test_repeated_runs_deterministic() -> None:
    """Repeated runs produce identical output and module sequences."""
    pipeline = CARPipeline()
    first = pipeline.run("fake_audio.wav")
    second = pipeline.run("fake_audio.wav")
    assert first["original"] == second["original"]
    assert first["corrected"] == second["corrected"]
    assert [e["module"] for e in first["attribution"]] == [e["module"] for e in second["attribution"]]
    assert first["attribution"][-1]["module"] == "LATENCY"


def test_latency_entry_present_with_no_gated_tokens() -> None:
    """The LATENCY entry is still appended with zero timings when nothing is gated."""
    pipeline = CARPipeline()
    pipeline.entropy_gate = lambda _transcript: [False] * 6  # stub transcript has six tokens
    log: list[dict[str, Any]] = []
    pipeline.run("fake_audio.wav", attribution_log=log)
    entry = latency_entry(log)
    assert entry["gate_latency_ms"] >= 0.0
    assert entry["semantic_retrieval_latency_ms"] == 0.0
    assert entry["phonetic_retrieval_latency_ms"] == 0.0
    assert entry["retrieval_latency_ms"] == 0.0
    assert entry["fusion_latency_ms"] == 0.0
