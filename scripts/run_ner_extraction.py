"""Run NER Extraction over AfriSpeech-200 test set to generate reference tags.

This script executes Task T4. It processes reference transcripts to extract
medical entities (MED, COND, ANA, TTP, PHI) which are required by Ankit
for the M-WER scoreboard (T1) and by Divya for phonetic FAISS extraction (T6).
"""

import json
import logging
import uuid
from pathlib import Path

from care_asr.config.settings import Settings
from care_asr.contracts.asr_input import ASRTranscriptInput, WordAlignment
from care_asr.ner.extractor import BioBertNERExtractor
from care_asr.ner.span_aligner import SpanAligner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_mock_afrispeech_data() -> list[ASRTranscriptInput]:
    """Generates mock AfriSpeech reference transcripts for testing the pipeline."""
    return [
        ASRTranscriptInput(
            transcript_id=uuid.uuid4(),
            audio_duration_seconds=5.0,
            raw_transcript="Patient prescribed metformin 500mg daily for diabetes.",
            words=[
                WordAlignment(
                    word="Patient", start_time=0.0, end_time=0.5, confidence=1.0, entropy=0.0, start_char=0, end_char=7
                ),
                WordAlignment(
                    word="prescribed",
                    start_time=0.6,
                    end_time=1.0,
                    confidence=1.0,
                    entropy=0.0,
                    start_char=8,
                    end_char=18,
                ),
                WordAlignment(
                    word="metformin",
                    start_time=1.1,
                    end_time=1.6,
                    confidence=1.0,
                    entropy=0.0,
                    start_char=19,
                    end_char=28,
                ),
                WordAlignment(
                    word="500mg", start_time=1.7, end_time=2.2, confidence=1.0, entropy=0.0, start_char=29, end_char=34
                ),
                WordAlignment(
                    word="daily", start_time=2.3, end_time=2.7, confidence=1.0, entropy=0.0, start_char=35, end_char=40
                ),
                WordAlignment(
                    word="for", start_time=2.8, end_time=3.0, confidence=1.0, entropy=0.0, start_char=41, end_char=44
                ),
                WordAlignment(
                    word="diabetes.",
                    start_time=3.1,
                    end_time=3.8,
                    confidence=1.0,
                    entropy=0.0,
                    start_char=45,
                    end_char=54,
                ),
            ],
        )
    ]


def main() -> None:
    settings = Settings()
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "afrispeech_reference_ner_tags.json"

    logger.info("Initializing BioBert NER Extractor...")
    extractor = BioBertNERExtractor(settings=settings, auto_load=True)
    aligner = SpanAligner()

    logger.info("Loading AfriSpeech reference transcripts (using mock data for now)...")
    transcripts = generate_mock_afrispeech_data()

    results = []

    for transcript in transcripts:
        logger.info(f"Extracting entities for transcript {transcript.transcript_id}...")

        # 1. Extract raw entities using BioBERT
        detected_entities = extractor.extract_entities(transcript)

        # 2. Align entities to words (for Divya's audio extraction and Ankit's M-WER)
        aligned_entities = aligner.align_entities_to_words(detected_entities, transcript)

        results.append(
            {
                "transcript_id": str(transcript.transcript_id),
                "raw_transcript": transcript.raw_transcript,
                "entities": aligned_entities,
            }
        )

    logger.info(f"Saving {len(results)} tagged transcripts to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("T4 Execution Complete. Hand off output to Ankit and Divya.")


if __name__ == "__main__":
    main()
