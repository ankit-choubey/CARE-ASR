import uuid
import pytest

from care_asr.contracts.asr_input import ASRTranscriptInput, WordAlignment
from care_asr.ner.span_aligner import SpanAligner

@pytest.fixture
def sample_asr_input():
    return ASRTranscriptInput(
        transcript_id=uuid.uuid4(),
        audio_duration_seconds=10.0,
        raw_transcript="Patient takes Aspirin 81mg daily for hypertension.",
        words=[
            WordAlignment(word="Patient", start_time=0.0, end_time=0.5, confidence=0.9, entropy=0.1, start_char=0, end_char=7),
            WordAlignment(word="takes", start_time=0.5, end_time=1.0, confidence=0.8, entropy=0.2, start_char=8, end_char=13),
            WordAlignment(word="Aspirin", start_time=1.0, end_time=1.5, confidence=0.95, entropy=0.05, start_char=14, end_char=21),
            WordAlignment(word="81mg", start_time=1.5, end_time=2.0, confidence=0.85, entropy=0.15, start_char=22, end_char=26),
            WordAlignment(word="daily", start_time=2.0, end_time=2.5, confidence=0.9, entropy=0.1, start_char=27, end_char=32),
            WordAlignment(word="for", start_time=2.5, end_time=2.8, confidence=0.9, entropy=0.1, start_char=33, end_char=36),
            WordAlignment(word="hypertension.", start_time=2.8, end_time=3.5, confidence=0.88, entropy=0.12, start_char=37, end_char=50),
        ]
    )


def test_span_aligner_single_word_exact_match(sample_asr_input):
    aligner = SpanAligner()
    entities = [
        {"entity_text": "Aspirin", "category": "MED", "start_char": 14, "end_char": 21}
    ]
    
    results = aligner.align_entities_to_words(entities, sample_asr_input)
    assert len(results) == 1
    
    res = results[0]
    assert len(res["aligned_words"]) == 1
    assert res["aligned_words"][0]["word"] == "Aspirin"
    assert res["start_time"] == 1.0
    assert res["end_time"] == 1.5
    assert res["average_asr_confidence"] == 0.95
    assert res["maximum_entropy"] == 0.05


def test_span_aligner_multi_word_overlap(sample_asr_input):
    aligner = SpanAligner()
    # Entity spanning "Aspirin 81mg" (simulating missing spaces or multi-word extraction)
    entities = [
        {"entity_text": "Aspirin 81mg", "category": "MED", "start_char": 14, "end_char": 26}
    ]
    
    results = aligner.align_entities_to_words(entities, sample_asr_input)
    assert len(results) == 1
    
    res = results[0]
    assert len(res["aligned_words"]) == 2
    assert res["aligned_words"][0]["word"] == "Aspirin"
    assert res["aligned_words"][1]["word"] == "81mg"
    assert res["start_time"] == 1.0
    assert res["end_time"] == 2.0
    assert res["average_asr_confidence"] == pytest.approx(0.9)  # (0.95 + 0.85)/2
    assert res["maximum_entropy"] == pytest.approx(0.15)


def test_span_aligner_partial_overlap(sample_asr_input):
    aligner = SpanAligner()
    # Entity captures only "hyper" out of "hypertension."
    entities = [
        {"entity_text": "hyper", "category": "COND", "start_char": 37, "end_char": 42}
    ]
    
    results = aligner.align_entities_to_words(entities, sample_asr_input)
    assert len(results) == 1
    
    res = results[0]
    assert len(res["aligned_words"]) == 1
    assert res["aligned_words"][0]["word"] == "hypertension."
    assert res["start_time"] == 2.8


def test_span_aligner_empty_alignment():
    # Empty transcript or no words but entity was extracted somehow (edge case)
    aligner = SpanAligner()
    empty_input = ASRTranscriptInput(
        transcript_id=uuid.uuid4(),
        audio_duration_seconds=10.0,
        raw_transcript="Ghost text",
        words=[]
    )
    entities = [
        {"entity_text": "Ghost", "category": "COND", "start_char": 0, "end_char": 5}
    ]
    
    results = aligner.align_entities_to_words(entities, empty_input)
    assert len(results) == 1
    assert len(results[0]["aligned_words"]) == 0
    assert results[0]["average_asr_confidence"] == 0.0
    assert results[0]["maximum_entropy"] == 0.0
    assert results[0]["start_time"] == 0.0
    assert results[0]["end_time"] == 0.0
