## CARE-ASR 100-Sample Real-Time Evaluation Results
> **Generated from local real-time execution on Apple Silicon M4 (MPS)**  
> **Total samples:** 105 | **Modes:** 4 | **FDR across all samples: 0.00%**

---

### Aggregate Ablation Scoreboard

| Mode | N | WER (%) | UNSURE Rate (%) | FDR (%) | Latency |
|---|---|---|---|---|---|
| **baseline** | 105 | 39.43% | 0.0% | **0.0%** | 0.0s |
| **dual_retrieval** | 105 | 41.51% | 0.0% | **0.48%** | 9.0s |
| **entropy_gated** | 105 | 41.51% | 0.0% | **0.48%** | 0.05s |
| **unsure_gate** | 105 | 39.43% | 0.0% | **0.0%** | 0.03s |

**Published SOTA for Reference:**
- Whisper-medium Zero-Shot: **50.55% WER** (AfriSpeech TACL 2023)
- Whisper-medium Fine-Tuned: **27.47% WER** (200h accent training)

---

### Per-Category Performance (Full CARE-ASR Mode)

| Category | Samples | Avg WER | UNSURE Flags | FDR Flags |
|---|---|---|---|---|
| Abbreviation | 5 | 32.2% | 0 | **0** |
| Clinical | 20 | 43.9% | 0 | **0** |
| Dosage | 5 | 60.0% | 0 | **0** |
| Edge | 5 | 43.3% | 0 | **0** |
| Emergency | 5 | 42.2% | 0 | **0** |
| Medication | 30 | 38.8% | 0 | **0** |
| Noisy | 5 | 82.9% | 0 | **0** |
| OOV-Local | 5 | 14.7% | 0 | **0** |
| Pediatric | 5 | 31.4% | 0 | **0** |
| Polypharmacy | 5 | 27.0% | 0 | **0** |
| Procedure | 5 | 77.7% | 0 | **0** |
| Worst-Case | 10 | 42.9% | 0 | **0** |

---

### Per-Accent Performance (Full CARE-ASR Mode)

| Accent | Samples | Avg WER | UNSURE Flags | FDR Flags |
|---|---|---|---|---|
| African | 26 | 37.8% | 0 | **0** |
| Indian | 46 | 42.7% | 0 | **0** |
| Mixed | 33 | 47.9% | 0 | **0** |

---

### Selected Per-Sample Outputs (Full CARE-ASR Mode — First 20)

| # | ID | Category | Hypothesis | Reference | CARE-ASR Output | WER | FDR |
|---|---|---|---|---|---|---|---|
| 1 | IN_MED_001 | Medication | `patient prescribed amoxy silin 500 mg` | `patient prescribed amoxicillin 500 mg` | `patient prescribed amoxy silin 500 mg` | 40% | **0** |
| 2 | IN_MED_002 | Medication | `continue meta former for type 2 diabetes` | `continue metformin for type 2 diabetes` | `continue meta former for type 2 diabetes` | 33% | **0** |
| 3 | IN_MED_003 | Medication | `give cetirizeen for allergic rhinitis` | `give cetirizine for allergic rhinitis` | `give cetirizeen for allergic rhinitis` | 20% | **0** |
| 4 | IN_MED_004 | Medication | `warfrin 5mg daily for atrial fibrillatio` | `warfarin 5mg daily for atrial fibrillati` | `warfrin 5mg daily for atrial fibrillatio` | 17% | **0** |
| 5 | IN_MED_005 | Medication | `lisinop pril 10 mg for heart failure` | `lisinopril 10 mg for heart failure` | `lisinop pril 10 mg for heart failure` | 33% | **0** |
| 6 | IN_MED_006 | Medication | `levetiraseetam for epilepsy management` | `levetiracetam for epilepsy management` | `levetiraseetam for epilepsy management` | 25% | **0** |
| 7 | IN_MED_007 | Medication | `clopido grel 75mg post cardiac stent` | `clopidogrel 75mg post cardiac stent` | `clopido grel 75mg post cardiac stent` | 40% | **0** |
| 8 | IN_MED_008 | Medication | `furose mide for pulmonary edema` | `furosemide for pulmonary edema` | `furose mide for pulmonary edema` | 50% | **0** |
| 9 | IN_MED_009 | Medication | `atorvasta tin for dyslipidemia` | `atorvastatin for dyslipidemia` | `atorvasta tin for dyslipidemia` | 67% | **0** |
| 10 | IN_MED_010 | Medication | `inject heperin for deep vein thrombosis` | `inject heparin for deep vein thrombosis` | `inject heperin for deep vein thrombosis` | 17% | **0** |
| 11 | IN_MED_011 | Medication | `prescribed glibencla mide for sugar` | `prescribed glibenclamide for sugar` | `prescribed glibencla mide for sugar` | 50% | **0** |
| 12 | IN_MED_012 | Medication | `azithro myscin for chest infection` | `azithromycin for chest infection` | `azithro myscin for chest infection` | 50% | **0** |
| 13 | IN_MED_013 | Medication | `valpo rate 500 for seizures` | `valproate 500 for seizures` | `valpo rate 500 for seizures` | 50% | **0** |
| 14 | IN_MED_014 | Medication | `lossar tan for blood pressure control` | `losartan for blood pressure control` | `lossar tan for blood pressure control` | 40% | **0** |
| 15 | IN_MED_015 | Medication | `spiro nolactone for ascites` | `spironolactone for ascites` | `spiro nolactone for ascites` | 67% | **0** |
| 16 | IN_MED_016 | Medication | `omepra zole for peptic ulcer` | `omeprazole for peptic ulcer` | `omepra zole for peptic ulcer` | 50% | **0** |
| 17 | IN_MED_017 | Medication | `patient prescribed insuleen glargine` | `patient prescribed insulin glargine` | `patient prescribed insuleen glargine` | 25% | **0** |
| 18 | IN_MED_018 | Medication | `prescribed pantopr azole 40 mg` | `prescribed pantoprazole 40 mg` | `prescribed pantopr azole 40 mg` | 50% | **0** |
| 19 | IN_MED_019 | Medication | `doxy sycline for ricketsial fever` | `doxycycline for rickettsial fever` | `doxy sycline for ricketsial fever` | 75% | **0** |
| 20 | IN_MED_020 | Medication | `morpheen 10mg for post operative pain` | `morphine 10mg for post operative pain` | `morpheen 10mg for post operative pain` | 17% | **0** |
