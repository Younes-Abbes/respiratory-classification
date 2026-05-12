#!/usr/bin/env python3
"""Quick baseline evaluation without t-SNE (too slow on CPU)."""

import sys
sys.path.insert(0, str(__file__).rsplit(r"\scripts", 1)[0])

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Project imports
from src.dataset import ICBHIDataset
from src.model import load_model
from src.evaluate import compute_icbhi_metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pt"
TEST_CSV = "data/splits/test.csv"
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]

print("="*70)
print("BASELINE MODEL EVALUATION (FAST) — Task 20-21")
print("="*70)
print(f"Device: {DEVICE}")

# ============================================================================
# Load Model
# ============================================================================
print("\n[1/4] Loading model...")

with open("configs/baseline.yaml") as f:
    config = yaml.safe_load(f)

try:
    model = load_model(num_classes=4, use_pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded checkpoint with FallbackCNN")
except RuntimeError:
    model = load_model(num_classes=4, use_pretrained=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded checkpoint with CustomAST")

model = model.to(DEVICE)
model.eval()

epochs_trained = checkpoint.get("epoch", "N/A")
best_se_ckpt = checkpoint.get("best_Se", "N/A")

print(f"   Epochs trained: {epochs_trained}")
if best_se_ckpt != "N/A":
    print(f"   Best validation Se: {best_se_ckpt:.4f}")

# ============================================================================
# Load Test Dataset
# ============================================================================
print("\n[2/4] Loading test dataset...")

test_dataset = ICBHIDataset(TEST_CSV, config, augment=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"✅ Test dataset loaded: {len(test_dataset)} cycles")

# ============================================================================
# Inference
# ============================================================================
print("\n[3/4] Running inference...")

all_preds = []
all_labels = []

with torch.no_grad():
    for specs, labels in tqdm(test_loader, desc="Evaluating"):
        specs = specs.to(DEVICE)
        labels = labels.cpu().numpy()
        
        outputs = model(specs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"✅ Inference complete: {len(all_labels)} samples")

# ============================================================================
# Compute Metrics
# ============================================================================
print("\n[4/4] Computing metrics...")

metrics = compute_icbhi_metrics(all_labels, all_preds)

# Reference paper targets
REF_SE = 0.6831
REF_SP = 0.6789
REF_SCORE = 0.6810

print("\n" + "="*70)
print("BASELINE RESULTS (from Colab training)")
print("="*70)
print(f"Sensitivity (Se):    {metrics['Se']:.4f}  ({metrics['Se']*100:.2f}%)")
print(f"Specificity (Sp):    {metrics['Sp']:.4f}  ({metrics['Sp']*100:.2f}%)")
print(f"ICBHI Score:         {metrics['Score']:.4f}  ({metrics['Score']*100:.2f}%)")
print("="*70)

print("\nCOMPARISON WITH REFERENCE (Atakanisik et al.):")
print(f"  Se:    {metrics['Se']*100:.2f}% vs {REF_SE*100:.2f}% (Δ {(metrics['Se']-REF_SE)*100:+.2f}%)")
print(f"  Sp:    {metrics['Sp']*100:.2f}% vs {REF_SP*100:.2f}% (Δ {(metrics['Sp']-REF_SP)*100:+.2f}%)")
print(f"  Score: {metrics['Score']*100:.2f}% vs {REF_SCORE*100:.2f}% (Δ {(metrics['Score']-REF_SCORE)*100:+.2f}%)")

# Save results
results_path = LOGS_DIR / "baseline_results.txt"
with open(results_path, "w") as f:
    f.write("Baseline AST+SAM (Colab Training)\n")
    f.write("="*60 + "\n")
    f.write(f"Sensitivity (Se):    {metrics['Se']*100:.2f}%\n")
    f.write(f"Specificity (Sp):    {metrics['Sp']*100:.2f}%\n")
    f.write(f"ICBHI Score:         {metrics['Score']*100:.2f}%\n")
    f.write("="*60 + "\n")
    f.write(f"Epochs trained:      {epochs_trained}\n")
    f.write(f"Test samples:        {len(all_labels)}\n")
    f.write(f"\nReference Paper Targets:\n")
    f.write(f"  Se:    {REF_SE*100:.2f}%\n")
    f.write(f"  Sp:    {REF_SP*100:.2f}%\n")
    f.write(f"  Score: {REF_SCORE*100:.2f}%\n")

print(f"\n✅ Results saved to {results_path}")

# ============================================================================
# Confusion Matrix
# ============================================================================
print("\nGenerating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Baseline AST+SAM — 4-Class Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()

cm_path = LOGS_DIR / "confusion_matrix_baseline.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
print(f"✅ Confusion matrix saved to {cm_path}")
plt.close()

# Per-class analysis
print("\nPer-Class Metrics:")
print(f"{'Class':<10} {'Support':<10} {'Precision':<12} {'Recall':<12}")
print("-"*50)
for i, class_name in enumerate(CLASS_NAMES):
    support = (all_labels == i).sum()
    if support == 0:
        continue
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{class_name:<10} {support:<10} {precision:<12.4f} {recall:<12.4f}")

# ============================================================================
# PHASE 5 Decision
# ============================================================================
print("\n" + "="*70)
print("PHASE 5 DECISION")
print("="*70)

THRESHOLD_SE = 0.64
THRESHOLD_SP = 0.60
THRESHOLD_SCORE = 0.62

se = metrics['Se']
sp = metrics['Sp']
score = metrics['Score']

se_pass = se >= THRESHOLD_SE
sp_pass = sp >= THRESHOLD_SP
score_pass = score >= THRESHOLD_SCORE

print("\n✅ BASELINE VERIFICATION:")
print(f"  [{'✓' if se_pass else '✗'}] Sensitivity:  {se*100:.2f}% {'≥' if se_pass else '<'} {THRESHOLD_SE*100:.0f}%")
print(f"  [{'✓' if sp_pass else '✗'}] Specificity:  {sp*100:.2f}% {'≥' if sp_pass else '<'} {THRESHOLD_SP*100:.0f}%")
print(f"  [{'✓' if score_pass else '✗'}] Score:       {score*100:.2f}% {'≥' if score_pass else '<'} {THRESHOLD_SCORE*100:.0f}%")

print("\n🎯 NEXT STEPS:")
if se_pass and sp_pass and score_pass:
    print("✅ BASELINE REPRODUCED — Ready for PHASE 5 (Improvement Experiments)")
elif se < THRESHOLD_SE:
    print(f"⚠️  LOW SENSITIVITY ({se*100:.2f}% < {THRESHOLD_SE*100:.0f}%)")
    print("   Possible issues:")
    print("   1. Patient-level split has overlap")
    print("   2. Input shape incorrect: (B,1,128,T) → (B,T,128)")
    print("   3. Zero-padding used instead of cyclic padding")
    print("   4. Weighted sampler not active")
else:
    print("✓ Metrics acceptable — Ready for PHASE 5")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE")
print("="*70)
