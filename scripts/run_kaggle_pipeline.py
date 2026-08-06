"""
CARE-ASR Fail-Safe Kaggle Automated Execution Script.

This script runs on Kaggle GPU (P100 / T4x2) to generate all required
heavy benchmark artifacts and save them directly to /kaggle/working/
so they are captured as Kaggle output files.
"""

import os
import sys
import json
import shutil
import traceback
import subprocess
from pathlib import Path

# Setup logging to /kaggle/working/execution.log
working_dir = Path("/kaggle/working")
log_file = working_dir / "execution.log"
err_file = working_dir / "error_log.txt"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

print("=== STARTING CARE-ASR KAGGLE GPU EXECUTION ===")

try:
    # 1. Setup paths
    repo_dir = working_dir / "CARE-ASR"

    if repo_dir.exists():
        print("Pulling latest CARE-ASR repository...")
        subprocess.run(["git", "fetch", "origin"], cwd=repo_dir)
        subprocess.run(["git", "checkout", "ankit"], cwd=repo_dir)
        subprocess.run(["git", "pull", "origin", "ankit"], cwd=repo_dir)
    else:
        print("Cloning CARE-ASR repository...")
        subprocess.run(["git", "clone", "-b", "ankit", "https://github.com/ankit-choubey/CARE-ASR.git", str(repo_dir)])

    os.chdir(str(repo_dir))
    sys.path.insert(0, str(repo_dir))
    os.environ["PYTHONPATH"] = str(repo_dir)

    print("=== INSTALLING DEPENDENCIES ===")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "abydos", "faiss-gpu", "jiwer", "datasets", "transformers", "torch", "outlines", "bitsandbytes", "pyyaml"])

    # STAGE 1: Build medical_vocab.json
    print("\n=== STAGE 1: BUILDING MEDICAL VOCAB ===")
    from abydos.phonetic import DoubleMetaphone
    dm = DoubleMetaphone()
    cui_path = Path("data/indices/cui_mapping.json")
    terms = set()
    if cui_path.exists():
        with open(cui_path) as f:
            cui_map = json.load(f)
        for k, v in cui_map.items():
            if isinstance(v, dict) and "concept_name" in v:
                terms.add(v["concept_name"])
            elif isinstance(v, str):
                terms.add(v)
    
    indian_drug_names = [
        "amoxicillin", "ampicillin", "azithromycin", "ciprofloxacin", "metformin",
        "amlodipine", "atorvastatin", "pantoprazole", "omeprazole", "cetirizine",
        "paracetamol", "acetaminophen", "ibuprofen", "diclofenac", "aspirin",
        "clopidogrel", "losartan", "telmisartan", "ramipril", "enalapril",
        "digoxin", "warfarin", "heparin", "insulin", "metoprolol",
        "carvedilol", "furosemide", "spironolactone", "hydrochlorothiazide",
        "levothyroxine", "prednisolone", "dexamethasone", "methylprednisolone",
        "salbutamol", "ipratropium", "montelukast", "theophylline",
        "methotrexate", "cyclophosphamide", "cisplatin", "carboplatin",
        "doxorubicin", "vincristine", "tamoxifen", "letrozole",
        "lisinopril", "valsartan", "candesartan", "irbesartan",
        "simvastatin", "rosuvastatin", "pravastatin", "fenofibrate",
        "glimepiride", "glipizide", "glyburide", "pioglitazone", "sitagliptin",
        "ceftriaxone", "cefuroxime", "cephalexin", "clindamycin", "vancomycin",
        "fluconazole", "ketoconazole", "acyclovir", "oseltamivir",
        "ranitidine", "famotidine", "domperidone", "ondansetron",
        "alprazolam", "diazepam", "lorazepam", "clonazepam",
        "sertraline", "fluoxetine", "escitalopram", "venlafaxine",
        "risperidone", "olanzapine", "quetiapine", "haloperidol",
        "phenytoin", "carbamazepine", "valproate", "levetiracetam", "lamotrigine",
        "morphine", "fentanyl", "tramadol", "codeine", "oxycodone",
        "dolo", "crocin", "combiflam", "allegra", "montair",
        "glycomet", "telma", "stamlo", "ecosprin", "shelcal",
        "hypertension", "diabetes", "pneumonia", "tuberculosis", "malaria",
        "dengue", "typhoid", "asthma", "bronchitis", "emphysema",
        "myocardial infarction", "angina", "arrhythmia", "tachycardia",
        "bradycardia", "atrial fibrillation", "congestive heart failure",
        "stroke", "epilepsy", "migraine", "neuropathy", "meningitis",
        "hepatitis", "cirrhosis", "pancreatitis", "cholecystitis",
        "appendicitis", "peritonitis", "septicemia", "anemia",
        "thrombocytopenia", "leukemia", "lymphoma", "carcinoma",
        "abdomen", "thorax", "cranium", "femur", "tibia", "fibula",
        "humerus", "radius", "ulna", "vertebra", "sternum", "clavicle",
        "esophagus", "trachea", "bronchus", "alveoli", "diaphragm",
        "myocardium", "pericardium", "endocardium", "aorta", "vena cava",
        "cerebrum", "cerebellum", "hippocampus", "thalamus", "hypothalamus",
    ]
    terms.update(indian_drug_names)
    
    medical_vocab = {}
    for term in sorted(terms):
        t_str = str(term).strip()
        if not t_str: continue
        codes = [c for c in dm.encode(t_str) if c]
        if codes: medical_vocab[t_str] = codes
    
    Path("data/indices").mkdir(parents=True, exist_ok=True)
    with open("data/indices/medical_vocab.json", "w") as f:
        json.dump(medical_vocab, f, indent=2)
    with open(working_dir / "medical_vocab.json", "w") as f:
        json.dump(medical_vocab, f, indent=2)
    print(f"✅ Generated medical_vocab.json with {len(medical_vocab)} terms")

    # STAGE 2: Download AfriSpeech Dataset
    print("\n=== STAGE 2: DOWNLOADING AFRISPEECH-200 TEST DATASET ===")
    data_dir = working_dir / "afrispeech_clinical_test"
    from datasets import load_dataset, Dataset
    print("Downloading AfriSpeech-200 test split from HuggingFace...")
    try:
        ds = load_dataset("intronhealth/afrispeech-200", split="test", trust_remote_code=True)
    except Exception as e_hf:
        print(f"HuggingFace dataset download warning: {e_hf}; trying default load_dataset...")
        ds = load_dataset("intronhealth/afrispeech-200", split="test")

    print(f"Loaded dataset: {len(ds)} samples")
    if "domain" in ds.column_names:
        clinical_ds = ds.filter(lambda x: x.get("domain") == "clinical")
        if len(clinical_ds) > 0:
            ds = clinical_ds
            print(f"Filtered clinical domain: {len(ds)} samples")
    
    ds = ds.select(range(min(200, len(ds))))
    ds.save_to_disk(str(data_dir))
    print(f"✅ Saved AfriSpeech clinical test split to {data_dir}")

    # STAGE 3: Run 6-mode Ablation Evaluation
    print("\n=== STAGE 3: RUNNING 6-MODE ABLATION SWEEP ===")
    modes = ["baseline", "naive_correction", "dual_retrieval", "entropy_gated", "thresholded", "unsure_gate"]
    ablation_results = []

    for m in modes:
        print(f"\n---> Running Ablation Mode: {m}")
        proc = subprocess.run([sys.executable, "scripts/run_eval.py", "--mode", m, "--data-path", str(data_dir), "--out-dir", "results/ablation"])
        m_file = Path(f"results/ablation/{m}_metrics.json")
        if m_file.exists():
            with open(m_file) as f:
                res = json.load(f)
                ablation_results.append(res)
                print(f"  Result {m}: WER={res.get('wer')}, FDR={res.get('fdr')}")

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_table.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    with open(working_dir / "ablation_table.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print("✅ Saved ablation_table.json")

    # STAGE 4: Run India Context Evaluation
    print("\n=== STAGE 4: RUNNING INDIA CONTEXT EVALUATION ===")
    proc = subprocess.run([sys.executable, "scripts/run_india_eval.py", "--max-samples", "100", "--output-dir", "outputs/metrics/india"])
    ind_file = Path("outputs/metrics/india/india_context_table.json")
    if ind_file.exists():
        with open(ind_file) as f:
            ind_data = json.load(f)
        with open(working_dir / "india_context_table.json", "w") as f:
            json.dump(ind_data, f, indent=2)
        print("✅ Saved india_context_table.json")

    # STAGE 5: Copy all outputs directly to top-level /kaggle/working/
    print("\n=== STAGE 5: COPYING ALL ARTIFACTS TO TOP-LEVEL KAGGLE OUTPUT ===")
    if (repo_dir / "results").exists():
        shutil.copytree(repo_dir / "results", working_dir / "results", dirs_exist_ok=True)
    if (repo_dir / "data" / "indices").exists():
        shutil.copytree(repo_dir / "data" / "indices", working_dir / "data" / "indices", dirs_exist_ok=True)
    if (repo_dir / "outputs" / "metrics").exists():
        shutil.copytree(repo_dir / "outputs" / "metrics", working_dir / "outputs" / "metrics", dirs_exist_ok=True)
    print("✅ All result, index, and metric directories copied to /kaggle/working/")
    print("\n=== CARE-ASR KAGGLE GPU RUN COMPLETE ===")

except Exception as top_level_err:
    print(f"\n❌ FATAL TOP LEVEL EXCEPTION: {top_level_err}")
    traceback.print_exc()
    with open(err_file, "w") as ef:
        ef.write(f"Error: {top_level_err}\n")
        traceback.print_exc(file=ef)
