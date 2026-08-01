"""
Unit tests for Reciprocal Rank Fusion (M5 Module).
"""

from care_asr.contracts.retrieval_input import RetrievalCandidate
from src.fusion.rrf import reciprocal_rank_fusion


def _c(name: str, score: float, src: str = "semantic") -> RetrievalCandidate:
    return RetrievalCandidate(candidate=name, score=score, source=src)


def test_rrf_promotes_common_candidate():
    """Verify RRF elevates candidates appearing in both semantic and phonetic channels."""
    sem = [_c("amoxicillin", 0.9), _c("ampicillin", 0.7)]
    pho = [_c("amoxicillin", 0.8, "phonetic"), _c("amoxycillin", 0.6, "phonetic")]
    result = reciprocal_rank_fusion([sem, pho])

    assert len(result) > 0
    assert result[0].candidate.lower() == "amoxicillin"


def test_rrf_single_list_passthrough():
    """Verify RRF handles a single list correctly."""
    result = reciprocal_rank_fusion([[_c("metformin", 0.9)]])
    assert len(result) == 1
    assert result[0].candidate.lower() == "metformin"
