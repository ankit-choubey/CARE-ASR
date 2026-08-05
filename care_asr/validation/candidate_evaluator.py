"""Candidate Evaluator Module.

Why It Exists:
    Evaluates FAISS retrieval candidates for each aligned entity against the threshold
    engine and computes a ranked composite score to produce a validated candidate list.

Teammate Dependencies:
    - Divya (FAISS & Retrieval Lead): Provides `RetrievalCandidatesInput`.
    - Ankit (Integration Lead): Consumes `ValidatedCandidatesOutput`.

Design Rationale:
    - Merges semantic and phonetic candidates, resolving their origin (HYBRID, SEMANTIC, PHONETIC).
    - Normalized weights: Allows direct combination of disparate metrics.
    - O(N log N) ranking: Minimal allocations, highly performant sorting.
"""

import logging
import time
import uuid
from typing import Any

from care_asr.config.settings import get_settings
from care_asr.contracts.retrieval_input import EntityQuery, RetrievalCandidatesInput
from care_asr.contracts.validated_output import (
    AppliedThresholds,
    DetectedEntity,
    ProcessingMetadata,
    ValidatedCandidate,
    ValidatedCandidatesOutput,
)
from care_asr.thresholds.threshold_engine import CategoryThresholdEngine
from care_asr.utils.exceptions import ThresholdConfigurationError
from care_asr.validation.decision_router import DecisionRouter

logger = logging.getLogger(__name__)


class CandidateEvaluator:
    """Evaluates and ranks retrieval candidates using category thresholds and composite scoring.

    Recovery decisions (requires_recovery) are delegated to ``DecisionRouter``, which
    evaluates the primary ASR confidence and entropy against the category thresholds.
    """

    def __init__(self, threshold_engine: CategoryThresholdEngine) -> None:
        self.threshold_engine = threshold_engine
        self.decision_router = DecisionRouter(threshold_engine)
        self.settings = get_settings()
        self.weights = self._load_weights()

    def _load_weights(self) -> dict[str, float]:
        """Loads normalization weights for composite scoring from config.yaml."""
        yaml_config = self.settings.load_yaml_config()
        weights: dict[str, float] = yaml_config.get(
            "weights",
            {
                "semantic_similarity": 0.50,
                "phonetic_similarity": 0.25,
                "asr_confidence": 0.15,
                "entropy": 0.10,
            },
        )
        return weights

    def evaluate_candidates(
        self,
        aligned_entities: list[dict[str, Any]],
        retrieval_input: RetrievalCandidatesInput,
        ner_exec_time_ms: float = 0.0,
    ) -> ValidatedCandidatesOutput:
        """Evaluates FAISS candidates, computes composite utility scores, and builds final output."""
        start_time = time.perf_counter()
        logger.info(f"Evaluating candidates for transcript '{retrieval_input.transcript_id}'...")

        detected_entities: list[DetectedEntity] = []
        high_entropy_count = 0

        for aligned_ent in aligned_entities:
            entity_data = aligned_ent["entity"]
            category = entity_data["category"]

            ent_id = entity_data.get("entity_id", str(uuid.uuid4()))
            asr_conf = aligned_ent.get("average_asr_confidence", 0.0)
            entropy = aligned_ent.get("maximum_entropy", 0.0)

            query = self._find_matching_query(entity_data, retrieval_input.entity_queries)

            validated_cands = []
            if query:
                merged_candidates = self._merge_candidates(query)
                validated_cands = self._validate_and_rank(merged_candidates, category, asr_conf, entropy)

            if category not in self.threshold_engine.thresholds:
                raise ThresholdConfigurationError(f"Unknown category: {category}")

            threshold_rules = self.threshold_engine.thresholds[category]

requires_recovery = self.decision_router.should_trigger_recovery(
    category=category,
    asr_confidence=asr_conf,
    asr_entropy=entropy,
)

if requires_recovery:
    high_entropy_count += 1

            detected_entities.append(
                DetectedEntity(
                    entity_id=ent_id,
                    category=category,
                    original_text=entity_data.get("entity_text", ""),
                    start_char=entity_data.get("start_char", 0),
                    end_char=entity_data.get("end_char", 0),
                    primary_asr_confidence=asr_conf,
                    asr_entropy=entropy,
                    requires_recovery=requires_recovery,
                    validated_candidates=validated_cands,
                    applied_thresholds=AppliedThresholds(
                        min_asr_conf_threshold=threshold_rules.get("min_asr_confidence", 0.0),
                        min_sim_threshold=threshold_rules.get("min_semantic_similarity", 0.0),
                        max_phonetic_dist_threshold=threshold_rules.get("max_phonetic_distance", 0.0),
                    ),
                )
            )

            if validated_cands:
                top_cand = validated_cands[0]
                logger.debug(
                    f"Entity '{entity_data.get('entity_text')}' top candidate: '{top_cand.canonical_name}' (Score: {top_cand.composite_utility_score:.2f})"
                )

        eval_exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        return self._build_output(
            retrieval_input.transcript_id,
            detected_entities,
            ner_exec_time_ms,
            eval_exec_time_ms,
            high_entropy_count,
        )

    def _find_matching_query(self, entity_data: dict[str, Any], queries: list[EntityQuery]) -> EntityQuery | None:
        """Matches an entity to a FAISS EntityQuery using character offsets."""
        start_char = entity_data.get("start_char")
        end_char = entity_data.get("end_char")
        for q in queries:
            if q.start_char == start_char and q.end_char == end_char:
                return q
        return None

    def _merge_candidates(self, query: EntityQuery) -> dict[str, dict[str, Any]]:
        """Merges and deduplicates semantic and phonetic FAISS candidates."""
        merged: dict[str, dict[str, Any]] = {}
        for sc in query.semantic_candidates:
            merged[sc.concept_id] = {
                "concept_id": sc.concept_id,
                "canonical_name": sc.canonical_name,
                "cui": sc.cui,
                "semantic_similarity": sc.similarity_score,
                "phonetic_distance": 10.0,  # Default penalty
                "source": "SEMANTIC_FAISS",
            }

        for pc in query.phonetic_candidates:
            if pc.concept_id in merged:
                merged[pc.concept_id]["phonetic_distance"] = pc.phonetic_distance
                merged[pc.concept_id]["source"] = "HYBRID"
            else:
                merged[pc.concept_id] = {
                    "concept_id": pc.concept_id,
                    "canonical_name": pc.canonical_name,
                    "cui": pc.cui,
                    "semantic_similarity": -1.0,  # Default penalty
                    "phonetic_distance": pc.phonetic_distance,
                    "source": "PHONETIC_FAISS",
                }
        return merged

    def _validate_candidate(
        self, category: str, sem_sim: float, phon_dist: float, asr_conf: float, entropy: float
    ) -> bool:
        """Evaluates thresholds without throwing exceptions internally."""
        res = self.threshold_engine.evaluate_candidate_acceptance(
            category=category,
            semantic_similarity=sem_sim,
            phonetic_distance=phon_dist,
            asr_confidence=asr_conf,
            entropy=entropy,
        )
        return res.accepted

    def _compute_score(self, sem_sim: float, phon_dist: float, asr_conf: float, entropy: float) -> float:
        """Computes weighted score with non-linear normalization for distance metrics."""
        phon_sim = 1.0 / (1.0 + phon_dist)
        ent_sim = 1.0 / (1.0 + entropy)
        norm_sem = max(0.0, (sem_sim + 1.0) / 2.0)

        score = (
            self.weights.get("semantic_similarity", 0.50) * norm_sem
            + self.weights.get("phonetic_similarity", 0.25) * phon_sim
            + self.weights.get("asr_confidence", 0.15) * asr_conf
            + self.weights.get("entropy", 0.10) * ent_sim
        )
        return round(score, 4)

    def _rank_candidates(self, evaluated_cands: list[dict[str, Any]]) -> list[ValidatedCandidate]:
        """Ranks candidates dynamically and builds Pydantic records."""
        evaluated_cands.sort(key=lambda x: x["composite_score"], reverse=True)

        ranked_list = []
        for rank, cand in enumerate(evaluated_cands, start=1):
            ranked_list.append(
                ValidatedCandidate(
                    candidate_rank=rank,
                    canonical_name=cand["canonical_name"],
                    cui=cand["cui"],
                    candidate_source=cand["source"],
                    composite_utility_score=cand["composite_score"],
                    semantic_similarity=cand["semantic_similarity"],
                    phonetic_distance=cand["phonetic_distance"],
                    passes_category_threshold=cand["passes_threshold"],
                )
            )

        return ranked_list

    def _validate_and_rank(
        self,
        merged_candidates: dict[str, dict[str, Any]],
        category: str,
        asr_conf: float,
        entropy: float,
    ) -> list[ValidatedCandidate]:
        evaluated = []
        accepted_count = 0

        for cand in merged_candidates.values():
            sem_sim = cand["semantic_similarity"]
            phon_dist = cand["phonetic_distance"]

            passes = self._validate_candidate(category, sem_sim, phon_dist, asr_conf, entropy)
            score = self._compute_score(sem_sim, phon_dist, asr_conf, entropy)

            if passes:
                accepted_count += 1

            cand["passes_threshold"] = passes
            cand["composite_score"] = score
            evaluated.append(cand)

        logger.info(f"Validated {len(evaluated)} candidates for category {category}. Accepted: {accepted_count}.")
        return self._rank_candidates(evaluated)

    def _build_output(
        self,
        transcript_id: uuid.UUID,
        detected_entities: list[DetectedEntity],
        ner_ms: float,
        eval_ms: float,
        high_entropy_count: int,
    ) -> ValidatedCandidatesOutput:
        metadata = ProcessingMetadata(
            ner_execution_time_ms=ner_ms,
            validation_execution_time_ms=eval_ms,
            entities_count=len(detected_entities),
            high_entropy_entities_count=high_entropy_count,
        )

        return ValidatedCandidatesOutput(
            transcript_id=transcript_id,
            detected_entities=detected_entities,
            processing_metadata=metadata,
        )
