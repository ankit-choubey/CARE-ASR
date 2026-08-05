"""
Reciprocal Rank Fusion (M5 Module).
Merges multiple ranked lists of retrieval candidates into a unified list.
"""

from __future__ import annotations

from care_asr.contracts.retrieval_input import RetrievalCandidate


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalCandidate]],
    k: int = 60,
) -> list[RetrievalCandidate]:
    """
    Combines multiple ranked candidate lists using Reciprocal Rank Fusion (RRF).

    RRF_score(candidate) = sum( 1 / (k + rank_i) ) across all lists where candidate appears.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list, start=1):
            key = candidate.candidate.lower().strip()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    sorted_candidates = sorted(scores.items(), key=lambda x: -x[1])
    return [RetrievalCandidate(candidate=name, score=score, source="rrf") for name, score in sorted_candidates]
