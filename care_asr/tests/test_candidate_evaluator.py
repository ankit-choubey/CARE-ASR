import uuid

import pytest

from care_asr.contracts.retrieval_input import (
    EntityQuery,
    PhoneticCandidate,
    RetrievalCandidatesInput,
    SemanticCandidate,
)
from care_asr.thresholds.threshold_engine import CategoryThresholdEngine
from care_asr.utils.exceptions import ThresholdConfigurationError
from care_asr.validation.candidate_evaluator import CandidateEvaluator


@pytest.fixture
def evaluator():
    engine = CategoryThresholdEngine()
    return CandidateEvaluator(engine)


@pytest.fixture
def base_retrieval_input():
    return RetrievalCandidatesInput(transcript_id=uuid.uuid4(), entity_queries=[])


def test_single_candidate(evaluator, base_retrieval_input):
    query = EntityQuery(
        query_span_id="span-1",
        query_text="Aspirin",
        category="MED",
        start_char=0,
        end_char=7,
        semantic_candidates=[
            SemanticCandidate(
                concept_id="C0004057",
                canonical_name="Aspirin",
                similarity_score=0.95,
                cui="C0004057",
            )
        ],
        phonetic_candidates=[
            PhoneticCandidate(
                concept_id="C0004057",
                canonical_name="Aspirin",
                phonetic_distance=1.0,
                cui="C0004057",
            )
        ],
    )
    base_retrieval_input.entity_queries.append(query)

    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-1",
                "entity_text": "Aspirin",
                "category": "MED",
                "start_char": 0,
                "end_char": 7,
            },
            "average_asr_confidence": 0.90,
            "maximum_entropy": 0.10,
        }
    ]

    output = evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)
    assert len(output.detected_entities) == 1

    ent = output.detected_entities[0]
    assert len(ent.validated_candidates) == 1
    cand = ent.validated_candidates[0]

    assert cand.candidate_rank == 1
    assert cand.canonical_name == "Aspirin"
    assert cand.passes_category_threshold is True
    assert cand.candidate_source == "HYBRID"


def test_multiple_candidates_and_tie_scores(evaluator, base_retrieval_input):
    query = EntityQuery(
        query_span_id="span-2",
        query_text="Tylenol",
        category="MED",
        start_char=10,
        end_char=17,
        semantic_candidates=[
            SemanticCandidate(concept_id="C1", canonical_name="Acetaminophen", similarity_score=0.85, cui="C1"),
            SemanticCandidate(
                concept_id="C2", canonical_name="Paracetamol", similarity_score=0.85, cui="C2"
            ),  # Tie semantic
        ],
        phonetic_candidates=[
            PhoneticCandidate(
                concept_id="C1", canonical_name="Acetaminophen", phonetic_distance=1.0, cui="C1"
            )  # C1 wins tie due to phonetic
        ],
    )
    base_retrieval_input.entity_queries.append(query)

    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-2",
                "entity_text": "Tylenol",
                "category": "MED",
                "start_char": 10,
                "end_char": 17,
            },
            "average_asr_confidence": 0.85,
            "maximum_entropy": 0.20,
        }
    ]

    output = evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)
    cands = output.detected_entities[0].validated_candidates

    assert len(cands) == 2
    assert cands[0].canonical_name == "Acetaminophen"  # Won the tie
    assert cands[0].candidate_source == "HYBRID"
    assert cands[1].candidate_source == "SEMANTIC_FAISS"


def test_threshold_rejection(evaluator, base_retrieval_input):
    query = EntityQuery(
        query_span_id="span-3",
        query_text="BadDrug",
        category="MED",
        start_char=0,
        end_char=7,
        semantic_candidates=[
            SemanticCandidate(concept_id="C1", canonical_name="BadDrug", similarity_score=0.10, cui="C1")  # Sim too low
        ],
    )
    base_retrieval_input.entity_queries.append(query)

    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-3",
                "entity_text": "BadDrug",
                "category": "MED",
                "start_char": 0,
                "end_char": 7,
            },
            "average_asr_confidence": 0.90,
            "maximum_entropy": 0.10,
        }
    ]

    output = evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)
    cand = output.detected_entities[0].validated_candidates[0]
    assert cand.passes_category_threshold is False


def test_empty_retrieval(evaluator, base_retrieval_input):
    query = EntityQuery(query_span_id="span-4", query_text="Nothing", category="MED", start_char=0, end_char=7)
    base_retrieval_input.entity_queries.append(query)

    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-4",
                "entity_text": "Nothing",
                "category": "MED",
                "start_char": 0,
                "end_char": 7,
            },
            "average_asr_confidence": 0.90,
            "maximum_entropy": 0.10,
        }
    ]

    output = evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)
    assert len(output.detected_entities[0].validated_candidates) == 0


def test_unknown_category(evaluator, base_retrieval_input):
    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-5",
                "entity_text": "Unknown",
                "category": "INVALID",
                "start_char": 0,
                "end_char": 7,
            },
            "average_asr_confidence": 0.90,
            "maximum_entropy": 0.10,
        }
    ]

    with pytest.raises(ThresholdConfigurationError):
        evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)


def test_requires_recovery_low_confidence(
    evaluator: CandidateEvaluator, base_retrieval_input: RetrievalCandidatesInput
) -> None:
    """requires_recovery=True when DecisionRouter triggers on low ASR confidence."""
    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-6",
                "entity_text": "Metformin",
                "category": "MED",
                "start_char": 0,
                "end_char": 9,
            },
            "average_asr_confidence": 0.50,  # below MED min_asr_confidence (0.75)
            "maximum_entropy": 0.10,
        }
    ]

    output = evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)

    assert output.detected_entities[0].requires_recovery is True


def test_requires_recovery_high_entropy(
    evaluator: CandidateEvaluator, base_retrieval_input: RetrievalCandidatesInput
) -> None:
    """requires_recovery=True when DecisionRouter triggers on high ASR entropy."""
    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-7",
                "entity_text": "Metformin",
                "category": "MED",
                "start_char": 0,
                "end_char": 9,
            },
            "average_asr_confidence": 0.90,
            "maximum_entropy": 0.80,  # above MED max_entropy (0.45)
        }
    ]

    output = evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)

    assert output.detected_entities[0].requires_recovery is True


def test_no_requires_recovery_within_tolerance(
    evaluator: CandidateEvaluator, base_retrieval_input: RetrievalCandidatesInput
) -> None:
    """requires_recovery=False when DecisionRouter returns False for within-tolerance metrics."""
    aligned_entities = [
        {
            "entity": {
                "entity_id": "span-8",
                "entity_text": "Metformin",
                "category": "MED",
                "start_char": 0,
                "end_char": 9,
            },
            "average_asr_confidence": 0.90,
            "maximum_entropy": 0.10,
        }
    ]

    output = evaluator.evaluate_candidates(aligned_entities, base_retrieval_input)

    assert output.detected_entities[0].requires_recovery is False
