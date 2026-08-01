"""
Kaggle Ablation Evaluation Script (Task T9, T15, T16).
Runs all 6 ablation rows on Kaggle GPU P100.
"""

import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "baseline",
            "naive_correction",
            "dual_retrieval",
            "entropy_gated",
            "thresholded",
            "unsure_gate",
        ],
    )
    parser.add_argument("--data-path", default="/kaggle/working/afrispeech_clinical_test")
    parser.add_argument("--ner-path", default="/kaggle/working/ner_reference_spans.json")
    parser.add_argument("--out-dir", default="/kaggle/working/ablation")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    ds = load_from_disk(args.data_path)

    from src.pipeline.pipeline import CARPipeline

    pipeline = CARPipeline()

    if args.mode in [
        "naive_correction",
        "dual_retrieval",
        "entropy_gated",
        "thresholded",
        "unsure_gate",
    ]:
        from src.correction.llm_corrector import LLMCorrector

        pipeline.corrector = LLMCorrector().correct

    if args.mode in [
        "dual_retrieval",
        "entropy_gated",
        "thresholded",
        "unsure_gate",
    ]:
        from src.retrieval.phonetic import PhoneticRetriever
        from src.retrieval.semantic import SemanticRetriever

        pipeline.semantic_retrieve = SemanticRetriever().retrieve
        pipeline.phonetic_retrieve = PhoneticRetriever().retrieve

    if args.mode in ["entropy_gated", "thresholded", "unsure_gate"]:
        from care_asr.uncertainty.gate import TsallisUncertaintyGate

        gate_obj = TsallisUncertaintyGate()
        pipeline.entropy_gate = lambda t: gate_obj.gate_tokens(t.token_scores)["uncertain_flags"]

    if args.mode == "unsure_gate":
        from src.safety.unsure_gate import UnsureGate

        pipeline.safety_gate = UnsureGate().apply

    from transformers import pipeline as hf_pipeline

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        return_timestamps=True,
        device=0,
    )

    refs, hyps, preds = [], [], []
    unsure_count, total_corrections = 0, 0

    for sample in ds.select(range(min(200, len(ds)))):
        audio = {
            "array": np.array(sample["audio"]["array"], dtype=np.float32),
            "sampling_rate": sample["audio"]["sampling_rate"],
        }
        if args.mode == "baseline":
            res = asr(audio)
            hyp = res["text"].lower().strip()
            pred_dict = {
                "audio_id": sample.get("id", "unk"),
                "prediction": hyp,
                "reference": sample["transcript"].lower().strip(),
                "attribution": [],
            }
        else:
            log = []
            res = pipeline.run(audio, attribution_log=log)
            hyp = res["corrected"].lower().strip()
            pred_dict = {
                "audio_id": sample.get("id", "unk"),
                "prediction": hyp,
                "reference": sample["transcript"].lower().strip(),
                "attribution": log,
            }
            for entry in log:
                if entry.get("module") == "M6M7_CORRECT_GATE":
                    total_corrections += 1
                    if entry.get("label") == "UNSURE":
                        unsure_count += 1

        refs.append(sample["transcript"].lower().strip())
        hyps.append(hyp)
        preds.append(pred_dict)

    wer = jiwer.wer(refs, hyps)
    unsure_rate = unsure_count / total_corrections if total_corrections > 0 else 0.0

    row = {
        "mode": args.mode,
        "wer": round(wer, 4),
        "unsure_rate": round(unsure_rate, 4),
        "num_samples": len(preds),
    }
    print(json.dumps(row, indent=2))
    with open(f"{args.out_dir}/{args.mode}_predictions.json", "w") as f:
        json.dump(preds, f, indent=2)
    with open(f"{args.out_dir}/{args.mode}_metrics.json", "w") as f:
        json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
