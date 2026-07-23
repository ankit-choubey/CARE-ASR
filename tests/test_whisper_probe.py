"""
Pytest suite for CARE-ASR Task S3: Whisper Output Scores Probe

Tests that HuggingFace Whisper models expose decoder token scores required
for Tsallis entropy computation in the uncertainty gate.
"""

import pytest
import torch
from transformers.generation.utils import (
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)
from care_asr.probes.whisper_scores_probe import load_audio, run_probe


@pytest.fixture(scope="module")
def probe_outputs():
    """
    Module-scoped fixture to run probe on a real audio sample.
    Uses 'openai/whisper-tiny' or 'openai/whisper-medium' to test score generation.
    """
    audio = load_audio()
    # Use whisper-tiny or whisper-medium for fast unit testing verification
    outputs, processor, _ = run_probe(
        model_name="openai/whisper-tiny", audio_data=audio
    )
    return outputs, processor


def test_1_probe_returns_generate_output_with_scores(probe_outputs):
    """
    Test 1: Verify probe returns GenerateDecoderOnlyOutput or GenerateEncoderDecoderOutput with scores.
    """
    outputs, _ = probe_outputs
    assert isinstance(
        outputs, (GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput)
    ), (
        f"Expected GenerateEncoderDecoderOutput or GenerateDecoderOnlyOutput, got {type(outputs)}"
    )
    assert hasattr(outputs, "scores"), "Output object does not have 'scores' attribute."
    assert outputs.scores is not None, "outputs.scores is None."


def test_2_scores_list_is_non_empty(probe_outputs):
    """
    Test 2: Verify scores list is non-empty.
    """
    outputs, _ = probe_outputs
    assert isinstance(outputs.scores, (tuple, list)), (
        "outputs.scores should be a tuple or list."
    )
    assert len(outputs.scores) > 0, (
        f"outputs.scores is empty (len={len(outputs.scores)})."
    )


def test_3_every_score_tensor_has_vocabulary_dimension(probe_outputs):
    """
    Test 3: Verify every score tensor has vocabulary dimension.
    """
    outputs, processor = probe_outputs
    expected_vocab_size = processor.tokenizer.vocab_size

    for step_idx, score_tensor in enumerate(outputs.scores):
        assert isinstance(score_tensor, torch.Tensor), (
            f"Score at step {step_idx} is not a PyTorch Tensor."
        )
        assert score_tensor.ndim == 2, (
            f"Score tensor at step {step_idx} expected rank 2 (batch_size, vocab_size), got shape {score_tensor.shape}."
        )
        assert (
            score_tensor.shape[-1] == expected_vocab_size
            or score_tensor.shape[-1] > 50000
        ), (
            f"Score tensor at step {step_idx} does not match vocab dimension. Expected {expected_vocab_size}, got {score_tensor.shape[-1]}."
        )


def test_4_generated_transcription_is_non_empty(probe_outputs):
    """
    Test 4: Verify generated transcription is non-empty.
    """
    outputs, processor = probe_outputs
    transcription = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[
        0
    ]

    assert isinstance(transcription, str), "Transcription is not a string."
    assert len(transcription.strip()) > 0, (
        "Generated transcription is empty or whitespace only."
    )
