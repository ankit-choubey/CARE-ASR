import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure directory exists
output_dir = "/Users/theankit/Documents/AK/Projects/CARE-ASR/documentation/report"
os.makedirs(output_dir, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

# 1. F1 Score Comparison
plt.figure(figsize=(8, 6))
models = ['Whisper Zero-Shot', 'Whisper Fine-Tuned', 'CARE-ASR']
f1_scores = [0.55, 0.85, 0.98]
colors = ['#ff9999', '#66b3ff', '#99ff99']
bars = plt.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.2)
plt.title('Clinical Entity Retrieval F1 Score', fontweight='bold', fontsize=16)
plt.ylabel('F1 Score', fontsize=14)
plt.ylim(0, 1.1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"), dpi=300)
plt.close()

# 2. AUCNT Comparison
plt.figure(figsize=(8, 6))
methods = ['Max-Probability Threshold', 'Tsallis Entropy Gate (q=1/3)']
auc_scores = [21.28, 47.17]
colors_auc = ['#ffcc99', '#99ccff']
bars_auc = plt.bar(methods, auc_scores, color=colors_auc, edgecolor='black', linewidth=1.2, width=0.5)
plt.title('Area Under Curve for Negative Transfer (AUCNT)', fontweight='bold', fontsize=16)
plt.ylabel('AUCNT Score', fontsize=14)
plt.ylim(0, 55)

for bar in bars_auc:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "auc_roc_curve.png"), dpi=300)
plt.close()

# 3. False Drug Replacement (FDR) Rate
plt.figure(figsize=(8, 6))
systems = ['Standard Whisper', 'Corti Symphony', 'CARE-ASR']
fdr_rates = [2.5, 0.79, 0.0]
colors_fdr = ['#ff6666', '#ffb366', '#66cc66']
bars_fdr = plt.bar(systems, fdr_rates, color=colors_fdr, edgecolor='black', linewidth=1.2, width=0.6)
plt.title('False Drug Replacement (FDR) Rate (%)', fontweight='bold', fontsize=16)
plt.ylabel('FDR Rate (%) - Lower is Better', fontsize=14)
plt.ylim(0, 3.0)

for bar in bars_fdr:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fdr_safety_guarantee.png"), dpi=300)
plt.close()

print("Successfully generated all graphs in documentation/report/")
