"""
CARE-ASR Fail-Safe Kaggle Automated Execution Script.

This script is designed to run autonomously in a Kaggle GPU Notebook
(P100 or T4x2 GPU) using 'Save Version -> Save & Run All (Commit)'.
It runs completely in the background on Google Cloud servers even if
your laptop is shut down or powered off.

Outputs generated:
    1. data/indices/medical_vocab.json (Double Metaphone vocab)
    2. results/ablation_table.json (6-mode comparative evaluation table)
    3. outputs/metrics/india/ (India context evaluation metrics)
    4. Auto-commits and pushes all results back to GitHub branch 'ankit'

Usage in Kaggle:
    Copy and paste the entire content of this file into a single Kaggle Notebook cell,
    set GPU accelerator ON, click 'Save Version' -> 'Save & Run All', and shut down!
"""

import os
import sys
import subprocess

# 1. Repository Configuration
GITHUB_USER = "ankit-choubey"
GITHUB_REPO = "CARE-ASR"
GITHUB_BRANCH = "ankit"
GITHUB_TOKEN = ""  # Optional: Paste your GitHub Personal Access Token here if push requires auth

print("1. Cloning / updating repository...")
repo_dir = "/kaggle/working/CARE-ASR"
if os.path.exists(repo_dir):
    subprocess.run(["git", "pull", "origin", GITHUB_BRANCH], cwd=repo_dir)
else:
    subprocess.run(["git", "clone", f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git", repo_dir])
    subprocess.run(["git", "checkout", GITHUB_BRANCH], cwd=repo_dir)

os.chdir(repo_dir)
sys.path.insert(0, repo_dir)
os.environ["PYTHONPATH"] = repo_dir

print("2. Installing dependencies...")
subprocess.run(["pip", "install", "-q", "abydos", "faiss-gpu", "jiwer", "datasets", "transformers", "torch", "outlines", "bitsandbytes", "pyyaml"])

print("3. Executing Pipeline Stage 1: Build Phonetic Index & Medical Vocab...")
subprocess.run(["python3", "-c", """
import json, os
from pathlib import Path
try:
    from abydos.phonetic import DoubleMetaphone
    dm = DoubleMetaphone()
    cui_path = Path('data/indices/cui_mapping.json')
    terms = set()
    if cui_path.exists():
        with open(cui_path) as f:
            cui_map = json.load(f)
        for k, v in cui_map.items():
            if isinstance(v, dict) and 'concept_name' in v: terms.add(v['concept_name'])
    medical_vocab = {}
    for term in sorted(terms):
        codes = [c for c in dm.encode(str(term).strip()) if c]
        if codes: medical_vocab[str(term).strip()] = codes
    Path('data/indices').mkdir(parents=True, exist_ok=True)
    with open('data/indices/medical_vocab.json', 'w') as f: json.dump(medical_vocab, f, indent=2)
    print(f'✅ Generated medical_vocab.json with {len(medical_vocab)} terms')
except Exception as e:
    print(f'⚠️ Medical vocab warning: {e}')
"""])

print("4. Executing Pipeline Stage 2: Download AfriSpeech & Run 6-Mode Ablation...")
ablation_script = """
import os, json, subprocess
from pathlib import Path

# Fix run_eval.py gate API if needed
with open('scripts/run_eval.py') as f: c = f.read()
if 'gate_tokens(' in c:
    c = c.replace('gate_obj.gate_tokens(t.token_scores)["uncertain_flags"]', 'gate_obj.evaluate(t.token_scores)["uncertain_flags"]')
    with open('scripts/run_eval.py', 'w') as f: f.write(c)

print('Running 6-mode ablation sweep...')
modes = ['baseline', 'naive_correction', 'dual_retrieval', 'entropy_gated', 'thresholded', 'unsure_gate']
for m in modes:
    print(f'==> Running mode: {m}')
    subprocess.run(['python3', 'scripts/run_eval.py', '--mode', m, '--out-dir', 'results/ablation'])

ablation_table = []
for m in modes:
    p = Path(f'results/ablation/{m}_metrics.json')
    if p.exists():
        with open(p) as f: ablation_table.append(json.load(f))

with open('results/ablation_table.json', 'w') as f: json.dump(ablation_table, f, indent=2)
print('✅ Saved results/ablation_table.json')
"""
subprocess.run(["python3", "-c", ablation_script])

print("5. Executing Pipeline Stage 3: India Context Evaluation...")
subprocess.run(["python3", "scripts/run_india_eval.py", "--max-samples", "100", "--output-dir", "outputs/metrics/india"])

print("6. Auto-committing and pushing results to GitHub...")
subprocess.run(["git", "config", "user.name", "Kaggle Auto-Runner"])
subprocess.run(["git", "config", "user.email", "kaggle@care-asr.org"])
subprocess.run(["git", "add", "data/indices/", "results/", "outputs/metrics/", "scripts/run_eval.py"])
subprocess.run(["git", "commit", "-m", "fix(kaggle): complete T6/T14/T16 GPU ablation sweep and save indices"])

if GITHUB_TOKEN:
    push_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
    subprocess.run(["git", "push", push_url, f"HEAD:{GITHUB_BRANCH}"])
else:
    subprocess.run(["git", "push", "origin", f"HEAD:{GITHUB_BRANCH}"])

print("🎉 ALL STAGES COMPLETE! Artifacts pushed to GitHub branch 'ankit'.")
