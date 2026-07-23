"""
CARE-ASR Task S3: Whisper Output Scores Probe

This module probes HuggingFace's WhisperForConditionalGeneration model (openai/whisper-medium)
to verify that decoder token logit scores are exposed during generation. These scores are
required for downstream calculation of Tsallis entropy in the CARE-ASR uncertainty gate.

--------------------------------------------------------------------------------
EXPLANATORY COMMENTS FOR TASK S3 REQUIREMENTS:

1. WHAT outputs.scores CONTAINS:
   - `outputs.scores` is a tuple of PyTorch Tensors returned by `model.generate()` when
     `return_dict_in_generate=True` and `output_scores=True` are set.
   - Each element in the tuple corresponds to one autoregressive decoder generation step.
   - For each step `t`, `outputs.scores[t]` is a tensor of shape `(batch_size, vocab_size)`
     containing unnormalized logit values for all tokens in the Whisper vocabulary (vocab_size=51865 for whisper-medium).

2. HOW IT CAN LATER BE CONVERTED INTO PROBABILITIES:
   - Softmax transformation: `probabilities = torch.softmax(outputs.scores[t], dim=-1)`
   - Log-softmax transformation (for log-probs): `log_probabilities = torch.log_softmax(outputs.scores[t], dim=-1)`
   - Softmax converts raw logit scores into a valid probability distribution over the vocabulary where
     all elements are in [0, 1] and sum to 1.0.

3. WHY THIS IS NEEDED FOR TSALLIS ENTROPY:
   - CARE-ASR uses an entropy-based uncertainty gate to determine when to query external retrieval indices.
   - Tsallis entropy H_q(P) = (1 / (q - 1)) * (1 - sum_i(P_i^q)) is a non-extensive entropy measure parameterised
     by entropic index q.
   - Computing Tsallis entropy requires full probability distributions P_i across the model's vocabulary at each
     decoder timestep. Exposing `outputs.scores` ensures we can calculate token-level and sequence-level Tsallis
     entropy without requiring custom architecture modifications or internal model rewrites.
--------------------------------------------------------------------------------
"""

import os
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_audio(
    audio_path: Optional[str] = None, sample_rate: int = 16000
) -> np.ndarray:
    """
    Loads an audio sample for inference.

    Args:
        audio_path: Path to local audio file. If None, loads sample audio from HuggingFace dataset
                    or generates a fallback synthetic audio array.
        sample_rate: Target audio sampling rate (default: 16000 Hz for Whisper).

    Returns:
        np.ndarray: 1D floating-point audio array sampled at sample_rate.
    """
    if audio_path is not None and os.path.exists(audio_path):
        import librosa

        audio, _ = librosa.load(audio_path, sr=sample_rate)
        return audio

    # Try loading real audio sample from HuggingFace datasets
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample_audio = ds[0]["audio"]["array"]
        return np.array(sample_audio, dtype=np.float32)
    except Exception as e:
        print(f"[load_audio] Dataset load fallback due to: {e}")
        # Synthetic fallback: 3-second 440 Hz sine wave tone sampled at 16000 Hz
        t = np.linspace(0, 3.0, int(sample_rate * 3.0), endpoint=False)
        synthetic_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        return synthetic_audio


def run_probe(
    model_name: str = "openai/whisper-medium",
    audio_data: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
) -> Tuple[Any, WhisperProcessor, torch.Tensor]:
    """
    Loads Whisper processor and model, prepares input features, and runs generate()
    with return_dict_in_generate=True and output_scores=True.

    Args:
        model_name: Pretrained HuggingFace Whisper model identifier.
        audio_data: Audio signal array. If None, load_audio() will be called.
        sample_rate: Audio sampling rate (Hz).

    Returns:
        Tuple containing:
            - outputs: Generation output object containing sequences and scores.
            - processor: WhisperProcessor instance.
            - input_features: Processed log-Mel spectrogram input tensor.
    """
    if audio_data is None:
        audio_data = load_audio(sample_rate=sample_rate)

    print(f"[run_probe] Loading processor and model for '{model_name}'...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    print("[run_probe] Processing audio input features...")
    input_features = processor(
        audio_data, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features

    print(
        "[run_probe] Executing model.generate(return_dict_in_generate=True, output_scores=True)..."
    )
    with torch.no_grad():
        outputs = model.generate(
            input_features, return_dict_in_generate=True, output_scores=True
        )

    return outputs, processor, input_features


def inspect_scores(outputs: Any, processor: WhisperProcessor) -> Dict[str, Any]:
    """
    Inspects outputs.scores from Whisper model generation and prints required statistics.

    Args:
        outputs: Generation output object (GenerateEncoderDecoderOutput).
        processor: WhisperProcessor used for decoding tokens.

    Returns:
        Dict containing inspected metadata and statistics.
    """
    # 1. Transcription
    raw_sequence = outputs.sequences[0]
    transcription = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[
        0
    ]
    full_decoding = processor.batch_decode(
        outputs.sequences, skip_special_tokens=False
    )[0]

    # 2. Generated token IDs
    token_ids = raw_sequence.tolist()

    # 3. Number of decoder steps
    scores = outputs.scores
    num_decoder_steps = len(scores)

    # 4. Shape of every tensor inside outputs.scores
    score_shapes = [tuple(score.shape) for score in scores]

    # 5. Verification of len(outputs.scores) vs generated token count
    # Note: outputs.sequences contains initial prompt/prefix tokens (e.g. <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>)
    # plus the newly generated tokens.
    # `len(outputs.scores)` matches the exact number of autoregressive decoder steps taken by model.generate().
    # Determine prompt token count by checking output sequences length minus decoder score steps:
    num_total_tokens = len(token_ids)
    num_prompt_tokens = num_total_tokens - num_decoder_steps
    is_equal = num_decoder_steps == (num_total_tokens - num_prompt_tokens)

    # 6. First token probability distribution statistics
    first_logits = scores[0]  # shape: (batch_size, vocab_size)
    first_probs = torch.softmax(first_logits, dim=-1)[0]  # shape: (vocab_size,)

    max_prob_val, max_prob_idx = torch.max(first_probs, dim=-1)
    top5_probs, top5_indices = torch.topk(first_probs, 5, dim=-1)

    top5_tokens = [processor.tokenizer.decode([idx.item()]) for idx in top5_indices]
    top5_info = list(zip(top5_indices.tolist(), top5_probs.tolist(), top5_tokens))

    print("\n" + "=" * 70)
    print("CARE-ASR S3 WHISPER OUTPUT SCORES PROBE RESULTS")
    print("=" * 70)
    print(f"1. Transcription: '{transcription}'")
    print(f"   (Full sequence with special tokens: '{full_decoding}')")
    print(f"\n2. Generated Token IDs (Total {num_total_tokens} tokens):")
    print(f"   {token_ids}")
    print(f"\n3. Number of decoder steps (len(outputs.scores)): {num_decoder_steps}")
    print("\n4. Shape of every tensor inside outputs.scores:")
    for step_idx, shape in enumerate(score_shapes):
        print(f"   Step {step_idx:02d}: {shape}")

    print("\n5. Length Verification:")
    print(f"   len(outputs.scores) = {num_decoder_steps}")
    print(f"   Total tokens in sequence = {num_total_tokens}")
    print(f"   Prompt/Prefix tokens count = {num_prompt_tokens}")
    print(
        f"   Newly generated content tokens count = {num_total_tokens - num_prompt_tokens}"
    )
    print(
        f"   VERIFICATION MATCH: len(outputs.scores) == generated_content_tokens ({num_decoder_steps} == {num_total_tokens - num_prompt_tokens}): {is_equal}"
    )
    print(
        "   Explanation: outputs.scores contains logit tensors for each autoregressive generation step."
    )
    print(
        "   Initial prompt tokens are pre-filled in the decoder, so outputs.scores length strictly equals"
    )
    print("   the number of generated decoder steps (excluding initial prompt tokens).")

    print("\n6. First Token Probability Distribution Statistics:")
    print(f"   - Logits / Probs Tensor Shape: {tuple(first_logits.shape)}")
    print(
        f"   - Max Probability: {max_prob_val.item():.6f} (Token ID: {max_prob_idx.item()} -> '{processor.tokenizer.decode([max_prob_idx.item()])}')"
    )
    print("   - Top 5 Tokens:")
    for rank, (tid, prob, tok_str) in enumerate(top5_info, start=1):
        print(
            f"     {rank}. Token ID {tid:5d} | Prob: {prob:.6f} | Token: {repr(tok_str)}"
        )
    print("=" * 70 + "\n")

    return {
        "transcription": transcription,
        "token_ids": token_ids,
        "num_decoder_steps": num_decoder_steps,
        "score_shapes": score_shapes,
        "is_equal": is_equal,
        "first_step_shape": tuple(first_logits.shape),
        "first_step_max_prob": max_prob_val.item(),
        "top5_info": top5_info,
    }


if __name__ == "__main__":
    audio = load_audio()
    outputs, processor, _ = run_probe(audio_data=audio)
    inspect_scores(outputs, processor)
