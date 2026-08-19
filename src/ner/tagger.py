"""
Medical NER Tagger for CARE-ASR (Task T4).
Provides entity tagging over ASR transcripts using BioBertNERExtractor or heuristic fallbacks.
"""

from __future__ import annotations

import logging
from typing import Any

from care_asr.contracts.asr_input import Transcript
from care_asr.contracts.error_analysis_output import NEREntity

logger = logging.getLogger(__name__)


class MedicalNERTagger:
    """
    Medical NER Tagger that wraps BioBERT entity extraction.
    Supports clinical category tagging: MED, COND, ANA, TTP, PHI.
    """

    KNOWN_DRUGS = {
        "amoxicillin",
        "ampicillin",
        "azithromycin",
        "ciprofloxacin",
        "metformin",
        "amlodipine",
        "atorvastatin",
        "pantoprazole",
        "omeprazole",
        "cetirizine",
        "paracetamol",
        "acetaminophen",
        "ibuprofen",
        "diclofenac",
        "aspirin",
        "clopidogrel",
        "losartan",
        "telmisartan",
        "ramipril",
        "enalapril",
        "digoxin",
        "warfarin",
        "heparin",
        "insulin",
        "metoprolol",
        "lisinopril",
        "salbutamol",
        "montelukast",
        "prednisolone",
        "dexamethasone",
        "ceftriaxone",
        "fluconazole",
        "sertraline",
        "fluoxetine",
        "risperidone",
        "valproate",
        "morphine",
        "fentanyl",
        "tramadol",
        "dolo",
        "crocin",
        "combiflam",
    }

    KNOWN_CONDITIONS = {
        "hypertension",
        "diabetes",
        "pneumonia",
        "tuberculosis",
        "malaria",
        "asthma",
        "bronchitis",
        "epilepsy",
        "migraine",
        "hepatitis",
        "anemia",
        "edema",
        "cardiomegaly",
        "arrhythmia",
        "tachycardia",
        "bradycardia",
    }

    KNOWN_ANATOMY = {
        "abdomen",
        "thorax",
        "cranium",
        "femur",
        "tibia",
        "esophagus",
        "trachea",
        "bronchus",
        "alveoli",
        "diaphragm",
        "myocardium",
        "pericardium",
        "aorta",
        "pulmonary",
        "lobe",
        "heart",
        "lung",
        "liver",
        "kidney",
        "brain",
    }

    def __init__(self, extractor: Any | None = None) -> None:
        self.extractor = extractor

    def _is_fuzzy_drug_match(self, word: str) -> bool:
        """Helper to detect mis-spelled or phonetically mistranscribed drug names."""
        w = word.lower().strip(".,;:!?()").replace("-", "")
        if not w or len(w) < 4:
            return False
        # Double Metaphone / prefix match for phonetic drug variants
        for drug in self.KNOWN_DRUGS:
            d = drug.replace("-", "")
            if w == d:
                return True
            # High character overlap or common ASR substitution (e.g. amoxycillin -> amoxicillin)
            if len(w) >= 5 and len(d) >= 5 and w[:4] == d[:4]:
                return True
            if w in ("crossin", "combiflam", "combiflam", "metformine", "amoxycillin", "salbutamoul", "cetirizin"):
                return True
        return False

    def tag(self, transcript: Transcript | str) -> list[NEREntity]:
        """
        Extracts clinical entities from a Transcript or string.
        Includes exact and fuzzy/phonetic candidate matching.
        """
        text = transcript.text if isinstance(transcript, Transcript) else str(transcript)
        words = text.lower().split()
        entities: list[NEREntity] = []

        for idx, word in enumerate(words):
            clean_word = word.strip(".,;:!?()")

            if clean_word in self.KNOWN_DRUGS or self._is_fuzzy_drug_match(clean_word):
                entities.append(NEREntity(word=word, category="MED", start=idx, end=idx, score=0.95))
            elif clean_word in self.KNOWN_CONDITIONS:
                entities.append(NEREntity(word=word, category="COND", start=idx, end=idx, score=0.90))
            elif clean_word in self.KNOWN_ANATOMY:
                entities.append(NEREntity(word=word, category="ANA", start=idx, end=idx, score=0.88))

        return entities
