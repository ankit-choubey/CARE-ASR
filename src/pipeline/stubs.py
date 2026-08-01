"""
Stub implementations for T5 integration checkpoint.
Provides lightweight test stubs returning canonical Pydantic objects.
"""

from __future__ import annotations

from care_asr.contracts.asr_input import TokenScore, Transcript
from care_asr.contracts.error_analysis_output import NEREntity
from care_asr.contracts.retrieval_input import RetrievalCandidate
from care_asr.contracts.validated_output import CorrectionOutput


def stub_transcriber(audio_input: str) -> Transcript:
    """Stub transcriber returning fixed clinical transcript and token log probabilities."""
    return Transcript(
        text="patient prescribed amoxicillin five hundred milligrams",
        token_scores=[
            TokenScore(step=0, token_id=100, token="patient", log_prob=-0.01, prob=0.99),
            TokenScore(step=1, token_id=101, token="prescribed", log_prob=-0.02, prob=0.98),
            TokenScore(step=2, token_id=102, token="amoxicillin", log_prob=-2.5, prob=0.08),
            TokenScore(step=3, token_id=103, token="five", log_prob=-0.1, prob=0.90),
            TokenScore(step=4, token_id=104, token="hundred", log_prob=-0.1, prob=0.90),
            TokenScore(step=5, token_id=105, token="milligrams", log_prob=-0.3, prob=0.74),
        ],
        word_timestamps=[],
    )


def stub_entropy_gate(transcript: Transcript) -> list[bool]:
    """Stub entropy gate returning True for tokens with prob < 0.5."""
    return [ts.prob < 0.5 for ts in transcript.token_scores]


def stub_ner(transcript: Transcript) -> list[NEREntity]:
    """Stub NER tagger extracting clinical entities."""
    return [NEREntity(word="amoxicillin", category="MED", start=2, end=2, score=0.95)]


def stub_semantic_retrieve(token: str) -> list[RetrievalCandidate]:
    """Stub semantic retriever returning ranked RxNorm candidates."""
    return [
        RetrievalCandidate(candidate="amoxicillin", score=0.95, source="semantic"),
        RetrievalCandidate(candidate="ampicillin", score=0.78, source="semantic"),
    ]


def stub_phonetic_retrieve(token: str) -> list[RetrievalCandidate]:
    """Stub phonetic retriever returning ranked candidate pronunciations."""
    return [
        RetrievalCandidate(candidate="amoxicillin", score=0.88, source="phonetic"),
        RetrievalCandidate(candidate="amoxycillin", score=0.72, source="phonetic"),
    ]


def stub_corrector(token: str, candidates: list[RetrievalCandidate]) -> CorrectionOutput:
    """Stub LLM corrector picking top fused candidate."""
    best = candidates[0].candidate if candidates else token
    return CorrectionOutput(original_token=token, corrected_token=best, label="CORRECT", confidence=0.9)
