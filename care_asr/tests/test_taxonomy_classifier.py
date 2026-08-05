"""Unit tests for FailureTaxonomyClassifier.

Covers:
- Deterministic classification of retrieval_miss, boundary_mismatch,
  phonetic_distortion, and oov_error categories
- Edge cases: exact matches and empty strings
- aggregate_taxonomy counting from pre-classified and raw failure dicts
- Unknown categories skipped without raising
"""

from typing import Any

from care_asr.contracts.error_analysis_output import ErrorTaxonomy
from care_asr.evaluation.taxonomy_classifier import FailureTaxonomyClassifier


def test_classify_retrieval_miss_when_no_candidates() -> None:
    """Empty retrieval candidates classify as retrieval_miss regardless of strings."""
    assert FailureTaxonomyClassifier.classify_failure("metformin", "metformin", []) == "retrieval_miss"


def test_classify_retrieval_miss_with_dissimilar_text() -> None:
    """Dissimilar text with no candidates is still a retrieval miss."""
    assert FailureTaxonomyClassifier.classify_failure("levothyroxine", "aspirin", []) == "retrieval_miss"


def test_classify_boundary_mismatch_when_pred_contains_gt() -> None:
    """Predicted span wider than ground truth classifies as boundary_mismatch."""
    candidates = [{"concept_id": "C1"}]
    assert (
        FailureTaxonomyClassifier.classify_failure("hypertension", "severe hypertension", candidates)
        == "boundary_mismatch"
    )


def test_classify_boundary_mismatch_when_gt_contains_pred() -> None:
    """Predicted span narrower than ground truth classifies as boundary_mismatch."""
    candidates = [{"concept_id": "C1"}]
    assert FailureTaxonomyClassifier.classify_failure("500mg metformin", "metformin", candidates) == "boundary_mismatch"


def test_classify_phonetic_distortion_on_close_spelling() -> None:
    """Similar-sounding spelling variants classify as phonetic_distortion."""
    candidates = [{"concept_id": "C1"}]
    assert FailureTaxonomyClassifier.classify_failure("metformin", "metformun", candidates) == "phonetic_distortion"
    assert FailureTaxonomyClassifier.classify_failure("fever", "favor", candidates) == "phonetic_distortion"


def test_classify_oov_error_on_dissimilar_text() -> None:
    """Dissimilar term with candidates present classifies as oov_error."""
    candidates = [{"concept_id": "C1"}]
    assert FailureTaxonomyClassifier.classify_failure("levothyroxine", "aspirin", candidates) == "oov_error"


def test_classify_exact_match_with_candidates_defaults_to_oov() -> None:
    """Exact matches are not distortions; with candidates they default to oov_error."""
    candidates = [{"concept_id": "C1"}]
    assert FailureTaxonomyClassifier.classify_failure("aspirin", "aspirin", candidates) == "oov_error"


def test_classify_empty_strings_default_to_oov() -> None:
    """Degenerate empty inputs with candidates default to oov_error."""
    candidates = [{"concept_id": "C1"}]
    assert FailureTaxonomyClassifier.classify_failure("", "aspirin", candidates) == "oov_error"
    assert FailureTaxonomyClassifier.classify_failure("aspirin", "", candidates) == "oov_error"


def test_aggregate_taxonomy_counts_preclassified_failures() -> None:
    """Pre-classified failure dicts are counted into the matching ErrorTaxonomy fields."""
    failures = [
        {"failure_type": "retrieval_miss"},
        {"failure_type": "boundary_mismatch"},
        {"failure_type": "phonetic_distortion"},
        {"failure_type": "phonetic_distortion"},
        {"failure_type": "oov_error"},
    ]

    tax = FailureTaxonomyClassifier.aggregate_taxonomy(failures)

    assert isinstance(tax, ErrorTaxonomy)
    assert tax.retrieval_miss_count == 1
    assert tax.ner_boundary_mismatch_count == 1
    assert tax.phonetic_distortion_count == 2
    assert tax.oov_error_count == 1


def test_aggregate_taxonomy_classifies_raw_failures_on_the_fly() -> None:
    """Failure dicts without failure_type are classified from their fields."""
    failures: list[dict[str, Any]] = [
        {
            "ground_truth": "metformin",
            "predicted": "metformun",
            "retrieval_candidates": [{"concept_id": "C1"}],
        },
        {"ground_truth": "aspirin", "predicted": "aspirin"},
    ]

    tax = FailureTaxonomyClassifier.aggregate_taxonomy(failures)

    assert tax.phonetic_distortion_count == 1
    assert tax.retrieval_miss_count == 1
    assert tax.oov_error_count == 0
    assert tax.ner_boundary_mismatch_count == 0


def test_aggregate_taxonomy_skips_unknown_categories() -> None:
    """Unknown taxonomy categories are logged and skipped."""
    failures = [{"failure_type": "mystery"}, {"failure_type": "oov_error"}]

    tax = FailureTaxonomyClassifier.aggregate_taxonomy(failures)

    assert tax.oov_error_count == 1
    assert tax.phonetic_distortion_count == 0
    assert tax.retrieval_miss_count == 0
    assert tax.ner_boundary_mismatch_count == 0


def test_aggregate_taxonomy_empty_input() -> None:
    """An empty failure list produces an all-zero ErrorTaxonomy."""
    tax = FailureTaxonomyClassifier.aggregate_taxonomy([])

    assert isinstance(tax, ErrorTaxonomy)
    assert tax.phonetic_distortion_count == 0
    assert tax.oov_error_count == 0
    assert tax.ner_boundary_mismatch_count == 0
    assert tax.retrieval_miss_count == 0
