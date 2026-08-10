"""
Gradio web application demo for CARE-ASR.
Provides interactive UI to transcribe clinical speech and view step-by-step attribution logs.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

import sys
from pathlib import Path

# Add project root to sys.path so 'src' and 'care_asr' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.pipeline import CARPipeline
pipeline = CARPipeline()


def process_audio(audio_tuple: Any) -> tuple[str, str, str]:
    """Processes uploaded clinical speech and returns original, corrected, and trace logs."""
    if audio_tuple is None:
        return "No audio provided", "No audio provided", "[]"

    if isinstance(audio_tuple, str):
        sr = 16000
        audio_arr = np.zeros(16000, dtype=np.float32)
    else:
        sr, audio_arr = audio_tuple
        if audio_arr.dtype == np.int16:
            audio_arr = audio_arr.astype(np.float32) / 32768.0

    attribution_log: list = []
    result = pipeline.run(audio_arr, attribution_log=attribution_log)

    return (
        result["original"],
        result["corrected"],
        json.dumps(attribution_log, indent=2),
    )


def create_demo():
    """Builds Gradio demo interface."""
    import gradio as gr

    demo = gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(label="Upload Clinical Speech (.wav)", type="numpy"),
        outputs=[
            gr.Textbox(label="Whisper Original Transcription"),
            gr.Textbox(label="CARE-ASR Corrected Transcript"),
            gr.Textbox(label="Per-Layer Attribution Log", lines=10),
        ],
        title="CARE-ASR: Clinical Speech Recognition Post-Correction",
        description="Confidence-Aware Retrieval-Augmented Clinical Entity Recovery for Accented Medical Speech",
    )
    return demo


if __name__ == "__main__":
    app = create_demo()
    app.launch()
