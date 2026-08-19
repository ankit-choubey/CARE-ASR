"""
End-to-end CARE-ASR pipeline orchestrator.
Connects M1 ASR -> M2 Entropy Gate -> M3 NER -> M4 Dual Retrieval -> M5 Fusion -> M6 LLM Corrector -> M7 Safety Gate.
"""

from __future__ import annotations

import inspect
from typing import Any

from src.fusion.rrf import reciprocal_rank_fusion
from src.pipeline.stubs import (
    stub_corrector,
    stub_ner,
    stub_phonetic_retrieve,
    stub_semantic_retrieve,
    stub_transcriber,
)
from src.retrieval.latency import LatencyStats


class CARPipeline:
    """Master CARE-ASR pipeline orchestrating all 8 modular processing layers.

    Component attributes (``transcriber``, ``entropy_gate``, ``ner``,
    ``semantic_retrieve``, ``phonetic_retrieve``, ``corrector``,
    ``safety_gate``) are swappable callables, defaulting to the stubs in
    ``src.pipeline.stubs`` and typically overridden by callers such as
    ``scripts/run_eval.py`` or the latency benchmark.
    """

    transcriber: Any
    entropy_gate: Any
    ner: Any
    semantic_retrieve: Any
    phonetic_retrieve: Any
    corrector: Any
    safety_gate: Any
    stats: LatencyStats
    retrieval_top_k: int | None

    def __init__(self) -> None:
        self.transcriber = stub_transcriber
        from src.entropy.gate import TsallisEntropyGate

        self.entropy_gate = TsallisEntropyGate()
        self.ner = stub_ner
        self.semantic_retrieve = stub_semantic_retrieve
        self.phonetic_retrieve = stub_phonetic_retrieve
        self.corrector = stub_corrector
        self.safety_gate = None
        self.stats = LatencyStats()
        self.retrieval_top_k: int | None = None

    def run(self, audio_input: Any, attribution_log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Runs end-to-end CARE-ASR correction pipeline across input audio.

        T12 latency instrumentation: M2 entropy gate, M4 semantic/phonetic
        retrieval, and M5 fusion are timed via ``self.stats`` and appended to
        the attribution log as a single per-transcript ``LATENCY`` entry.
        Retrieval batches all gated tokens through ``retrieve_many`` whenever
        the bound retrievers expose it and falls back to per-token
        ``retrieve()`` calls otherwise. Because retrieval is batched, all
        retrieval calls happen before the per-token fusion loop; candidate
        results, ordering, and attribution entries are unchanged. Existing
        attribution entries are untouched; only the ``LATENCY`` entry is
        appended.
        """
        if attribution_log is None:
            attribution_log = []

        # Latency samples always describe the most recent run() call.
        self.stats = LatencyStats()

        transcript = self.transcriber(audio_input)
        attribution_log.append({"module": "M1_ASR", "text": transcript.text})

        with self.stats.timed("gate"):
            uncertain_flags = self.entropy_gate(transcript)
        attribution_log.append({"module": "M2_ENTROPY", "uncertain_count": sum(uncertain_flags)})

        entities = self.ner(transcript)
        entity_words = {e.word.lower() for e in entities}
        attribution_log.append({"module": "M3_NER", "entity_count": len(entities)})

        words = transcript.text.split()
        corrected_words = list(words)

        retrieval_tokens = [
            word
            for i, (word, is_uncertain) in enumerate(zip(words, uncertain_flags, strict=False))
            if is_uncertain and word.lower() in entity_words
        ]

        semantic_results: dict[str, list[Any]] = {}
        phonetic_results: dict[str, list[Any]] = {}
        if retrieval_tokens:
            with self.stats.timed("retrieval"):
                semantic_results = self._retrieve_all(self.semantic_retrieve, "semantic_retrieval", retrieval_tokens)
                phonetic_results = self._retrieve_all(self.phonetic_retrieve, "phonetic_retrieval", retrieval_tokens)

        for i, (word, is_uncertain) in enumerate(zip(words, uncertain_flags, strict=False)):
            if not (is_uncertain and word.lower() in entity_words):
                continue

            semantic_candidates = semantic_results.get(word, [])
            phonetic_candidates = phonetic_results.get(word, [])
            attribution_log.append(
                {
                    "module": "M4_RETRIEVAL",
                    "token": word,
                    "semantic_top1": (semantic_candidates[0].candidate if semantic_candidates else None),
                    "phonetic_top1": (phonetic_candidates[0].candidate if phonetic_candidates else None),
                }
            )

            with self.stats.timed("fusion"):
                fused = reciprocal_rank_fusion([semantic_candidates, phonetic_candidates])
            attribution_log.append(
                {
                    "module": "M5_FUSION",
                    "fused_top1": fused[0].candidate if fused else None,
                }
            )

            correction = self.corrector(word, fused)
            if self.safety_gate is not None:
                correction = self.safety_gate(correction)

            attribution_log.append(
                {
                    "module": "M6M7_CORRECT_GATE",
                    "label": correction.label,
                    "corrected": correction.corrected_token,
                }
            )

            if correction.label == "CORRECT":
                corrected_words[i] = correction.corrected_token
            # WRONG and UNSURE both preserve original token — zero false drug replacements

        attribution_log.append(
            {
                "module": "LATENCY",
                "gate_latency_ms": self._run_total("gate"),
                "semantic_retrieval_latency_ms": self._run_total("semantic_retrieval"),
                "phonetic_retrieval_latency_ms": self._run_total("phonetic_retrieval"),
                "retrieval_latency_ms": self._run_total("retrieval"),
                "fusion_latency_ms": self._run_total("fusion"),
            }
        )

        return {
            "original": transcript.text,
            "corrected": " ".join(corrected_words),
            "attribution": attribution_log,
        }

    @staticmethod
    def _resolve_retrieve_many(retrieve_callable: Any) -> Any | None:
        """Resolves a batched ``retrieve_many`` callable when one is available.

        Accepts either a callable exposing ``retrieve_many`` directly (e.g. a
        retriever namespace) or a bound ``retrieve()`` method whose instance
        provides ``retrieve_many`` (e.g. ``SemanticRetriever().retrieve``).
        Returns None when only sequential retrieval is supported.
        """
        if hasattr(retrieve_callable, "retrieve_many"):
            return retrieve_callable.retrieve_many
        instance = getattr(retrieve_callable, "__self__", None)
        if instance is not None:
            return getattr(instance, "retrieve_many", None)
        return None

    def _retrieve_all(
        self,
        retrieve_callable: Any,
        stats_name: str,
        tokens: list[str],
    ) -> dict[str, list[Any]]:
        """Retrieves candidates for every token, batching via ``retrieve_many`` when available.

        Falls back to sequential per-token ``retrieve()`` calls otherwise.
        Results are keyed by token; the optional ``retrieval_top_k`` attribute
        is forwarded to the retrievers when set and accepted by the callable.
        """
        many = self._resolve_retrieve_many(retrieve_callable)
        top_k = self.retrieval_top_k
        results: dict[str, list[Any]] = {}
        with self.stats.timed(stats_name):
            if many is not None:
                forward_top_k = top_k is not None and self._accepts_top_k(many)
                batched = many(tokens) if not forward_top_k else many(tokens, top_k)
                results = dict(zip(tokens, batched, strict=False))
            else:
                forward_top_k = top_k is not None and self._accepts_top_k(retrieve_callable)
                for token in tokens:
                    results[token] = retrieve_callable(token) if not forward_top_k else retrieve_callable(token, top_k)
        return results

    @staticmethod
    def _accepts_top_k(retrieve_callable: Any) -> bool:
        """Returns True when the callable accepts a second positional (top_k) argument.

        Guards top_k forwarding so single-argument callables (e.g. the pipeline
        stub retrievers) are never called with an unexpected argument.
        """
        try:
            parameters = list(inspect.signature(retrieve_callable).parameters.values())
        except (TypeError, ValueError):
            return False
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
        ]
        return len(positional) >= 2

    def _run_total(self, name: str) -> float:
        """Returns the total elapsed milliseconds recorded under ``name`` for the current run."""
        return float(sum(self.stats.values(name)))
