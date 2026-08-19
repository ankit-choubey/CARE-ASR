import sys
import json
from src.pipeline.pipeline import CARPipeline
import pprint

def verify():
    print("=== INITIALIZING CARE-ASR PIPELINE ===")
    pipeline = CARPipeline()
    
    print("\n=== RUNNING PIPELINE WITH STUB AUDIO ===")
    log = []
    # In stubs, audio_input is ignored by stub_transcriber, it always returns:
    # "Patient shows symptoms of tachycarida and severe hypoxemeia."
    result = pipeline.run("dummy_audio.wav", attribution_log=log)
    
    print("\n=== FINAL RESULT ===")
    print(f"Original Text:  {result['original']}")
    print(f"Corrected Text: {result['corrected']}")
    
    print("\n=== MODULE ATTRIBUTION LOG ===")
    for idx, entry in enumerate(result['attribution']):
        print(f"\nStep {idx+1}: {entry['module']}")
        for k, v in entry.items():
            if k != "module":
                print(f"  - {k}: {v}")

if __name__ == "__main__":
    verify()
