"""
Stub implementations for T5 integration checkpoint.
Provides lightweight test stubs returning canonical Pydantic objects.
"""

from __future__ import annotations

import numpy as np

from care_asr.contracts.asr_input import TokenScore, Transcript
from care_asr.contracts.error_analysis_output import NEREntity
from care_asr.contracts.retrieval_input import RetrievalCandidate
from care_asr.contracts.validated_output import CorrectionOutput


def stub_transcriber(audio_input) -> Transcript:
    """Stub transcriber: if audio_input is a clinical text string, pass through directly.
    For audio arrays/dicts or non-clinical strings, return standard clinical stub transcript.
    On Kaggle GPU, this is replaced by real Whisper inference.
    """
    # Detect if this is a real clinical text string (has spaces and medical words)
    _CLINICAL_MARKERS = {
        "patient", "prescribed", "amoxicillin", "metformin", "amlodipine", "lisinopril",
        "hypertension", "diabetes", "epilepsy", "valproate", "furosemide", "crocin",
        "combiflam", "dolo", "warfarin", "heparin", "abdomen", "thorax", "clopidogrel",
        "cetirizine", "salbutamol", "levetiracetam", "aspirin", "losartan", "insulin",
    }
    is_clinical_text = (
        isinstance(audio_input, str)
        and " " in audio_input
        and any(w in audio_input.lower() for w in _CLINICAL_MARKERS)
    )

    if is_clinical_text:
        text = audio_input
        words = text.split()
        token_scores = [
            TokenScore(
                step=i,
                token_id=100 + i,
                token=w,
                log_prob=-2.5 if i % 4 == 2 else -0.1,  # Every 4th word is uncertain
                prob=0.08 if i % 4 == 2 else 0.90,
                entropy=2.5 if i % 4 == 2 else 0.1,
            )
            for i, w in enumerate(words)
        ]
        return Transcript(text=text, token_scores=token_scores, word_timestamps=[])

    # Standard stub — returns fixed clinical transcript (for tests and audio inputs)
    return Transcript(
        text="patient prescribed amoxicillin five hundred milligrams",
        token_scores=[
            TokenScore(step=0, token_id=100, token="patient",     log_prob=-0.01, prob=0.99, entropy=0.01),
            TokenScore(step=1, token_id=101, token="prescribed",  log_prob=-0.02, prob=0.98, entropy=0.02),
            TokenScore(step=2, token_id=102, token="amoxicillin", log_prob=-2.5,  prob=0.08, entropy=2.5),
            TokenScore(step=3, token_id=103, token="five",        log_prob=-0.1,  prob=0.90, entropy=0.1),
            TokenScore(step=4, token_id=104, token="hundred",     log_prob=-0.1,  prob=0.90, entropy=0.1),
            TokenScore(step=5, token_id=105, token="milligrams",  log_prob=-0.3,  prob=0.74, entropy=0.3),
        ],
        word_timestamps=[],
    )



def stub_entropy_gate(transcript: Transcript) -> list[bool]:
    """Stub entropy gate: marks tokens with prob < 0.5 as uncertain (simulates Tsallis gating)."""
    return [ts.prob < 0.5 for ts in transcript.token_scores]


_NER_TAGGER_INSTANCE = None
_SEMANTIC_RETRIEVER_INSTANCE = None
_PHONETIC_RETRIEVER_INSTANCE = None


def stub_ner(transcript: Transcript) -> list[NEREntity]:
    """Stub NER tagger: uses MedicalNERTagger for real entity extraction from transcript text."""
    global _NER_TAGGER_INSTANCE
    try:
        if _NER_TAGGER_INSTANCE is None:
            from src.ner.tagger import MedicalNERTagger
            _NER_TAGGER_INSTANCE = MedicalNERTagger()
        return _NER_TAGGER_INSTANCE.tag(transcript)
    except Exception:
        return [NEREntity(word="amoxicillin", category="MED", start=2, end=2, score=0.95)]


def stub_semantic_retrieve(token: str) -> list[RetrievalCandidate]:
    """Stub semantic retriever: returns FAISS-indexed candidates for medical terms."""
    global _SEMANTIC_RETRIEVER_INSTANCE
    try:
        if _SEMANTIC_RETRIEVER_INSTANCE is None:
            from src.retrieval.semantic import SemanticRetriever
            _SEMANTIC_RETRIEVER_INSTANCE = SemanticRetriever()
        return _SEMANTIC_RETRIEVER_INSTANCE.retrieve(token)
    except Exception:
        return [
            RetrievalCandidate(candidate=token, score=0.95, source="semantic"),
            RetrievalCandidate(candidate="amoxicillin", score=0.78, source="semantic"),
        ]


def stub_phonetic_retrieve(token: str) -> list[RetrievalCandidate]:
    """Stub phonetic retriever: uses FAISS phonetic index for candidate retrieval."""
    global _PHONETIC_RETRIEVER_INSTANCE
    try:
        if _PHONETIC_RETRIEVER_INSTANCE is None:
            from src.retrieval.phonetic import PhoneticRetriever
            _PHONETIC_RETRIEVER_INSTANCE = PhoneticRetriever()
        return _PHONETIC_RETRIEVER_INSTANCE.retrieve(token)
    except Exception:
        return [
            RetrievalCandidate(candidate=token, score=0.88, source="phonetic"),
            RetrievalCandidate(candidate="amoxicillin", score=0.72, source="phonetic"),
        ]


def stub_corrector(token: str, candidates: list[RetrievalCandidate]) -> CorrectionOutput:
    """Stub LLM corrector picking top fused candidate with high confidence."""
    best = candidates[0].candidate if candidates else token
    conf = candidates[0].score if candidates else 0.9
    return CorrectionOutput(original_token=token, corrected_token=best, label="CORRECT", confidence=float(conf))

