import torch
from transformers import pipeline
import numpy as np

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device="cpu",
)

arr = np.zeros(16000, dtype=np.float32)

print("Test 1: raw")
try:
    res = asr({"raw": arr, "sampling_rate": 16000})
    print("Success 1:", res)
except Exception as e:
    print("Error 1:", type(e).__name__, e)

print("Test 2: array")
try:
    res = asr({"array": arr, "sampling_rate": 16000})
    print("Success 2:", res)
except Exception as e:
    print("Error 2:", type(e).__name__, e)

print("Test 3: raw arr")
try:
    res = asr(arr)
    print("Success 3:", res)
except Exception as e:
    print("Error 3:", type(e).__name__, e)
