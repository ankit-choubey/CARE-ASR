"""Retrieval Candidates Input Contract Schema.

Why It Exists:
    Enforces runtime validation for top-K semantic FAISS and top-K phonetic candidates
    retrieved by Divya's search engine.

Teammate Dependencies:
    - Divya (FAISS & Retrieval Lead): Produces payloads conforming to this schema.

Imported By:
    - `care_asr.validation.candidate_evaluator`

TODOs:
    - Add field validator for official CARE-ASR categories (MED, COND, ANA, TTP, PHI).
"""

import logging
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CategoryType = Literal["MED", "COND", "ANA", "TTP", "PHI"]


class SemanticCandidate(BaseModel):
    """Represents a single candidate retrieved from Semantic FAISS index.

    Attributes:
        concept_id (str): Database concept unique identifier.
        canonical_name (str): Standardized medical concept name.
        similarity_score (float): Dense vector cosine similarity score in [-1.0, 1.0].
        cui (str): Unified Medical Language System (UMLS) Concept Unique Identifier.
    """

    concept_id: str = Field(..., description="Concept unique identifier")
    canonical_name: str = Field(..., description="Canonical medical name")
    similarity_score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score")
    cui: str = Field(..., description="UMLS CUI code")


class PhoneticCandidate(BaseModel):
    """Represents a single candidate retrieved from Phonetic FAISS / String index.

    Attributes:
        concept_id (str): Database concept unique identifier.
        canonical_name (str): Standardized medical concept name.
        phonetic_distance (float): String edit / Metaphone distance (>= 0.0, lower is better).
        phonetic_encoding (str | None): Double Metaphone or Soundex encoding string.
        cui (str): UMLS Concept Unique Identifier.
    """

    concept_id: str = Field(..., description="Concept unique identifier")
    canonical_name: str = Field(..., description="Canonical medical name")
    phonetic_distance: float = Field(..., ge=0.0, description="Phonetic string edit distance")
    phonetic_encoding: str | None = Field(
        default=None, description="Phonetic encoding representation"
    )
    cui: str = Field(..., description="UMLS CUI code")


class EntityQuery(BaseModel):
    """Represents a candidate query object for a specific entity span.

    Attributes:
        query_span_id (str): Unique entity span identifier matching BioBERT detection.
        query_text (str): Entity text string extracted from raw transcript.
        category (CategoryType): Official CARE-ASR entity category (MED, COND, ANA, TTP, PHI).
        start_char (int): 0-based start character offset.
        end_char (int): 0-based end character offset.
        semantic_candidates (list[SemanticCandidate]): Top-K semantic vector search candidates.
        phonetic_candidates (list[PhoneticCandidate]): Top-K phonetic search candidates.
    """

    query_span_id: str = Field(..., description="Entity query span identifier")
    query_text: str = Field(..., description="Entity query text")
    category: CategoryType = Field(..., description="Official CARE-ASR category")
    start_char: int = Field(..., ge=0, description="Start character offset")
    end_char: int = Field(..., ge=0, description="End character offset")
    semantic_candidates: list[SemanticCandidate] = Field(
        default_factory=list, description="Semantic candidates"
    )
    phonetic_candidates: list[PhoneticCandidate] = Field(
        default_factory=list, description="Phonetic candidates"
    )


class RetrievalCandidatesInput(BaseModel):
    """Input payload contract emitted by Divya's retrieval pipeline.

    Attributes:
        transcript_id (UUID): Unique transaction identifier matching ASR payload.
        entity_queries (list[EntityQuery]): List of candidate query objects for entity spans.
    """

    transcript_id: UUID = Field(..., description="Unique transaction UUID")
    entity_queries: list[EntityQuery] = Field(
        ..., description="List of entity queries with candidates"
    )
