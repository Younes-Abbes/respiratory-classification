#!/usr/bin/env python3
"""
Evaluate baseline model on test set and generate analysis plots.
Task 20-21: Baseline evaluation + confusion matrix + t-SNE visualization.
"""

import sys
sys.path.insert(0, str(__file__).rsplit(r"\scripts", 1)[0])

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import yaml

# Project imports
from src.dataset import ICBHIDataset
from src.model import load_model
from src.evaluate import compute_icbhi_metrics

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_model.pt"
CONFIG_PATH = "configs/baseline.yaml"
TEST_CSV = "data/splits/test.csv"
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
LABEL_MAP = {0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}

print("="*70)
print("BASELINE MODEL EVALUATION — Task 20-21")
print("="*70)
print(f"Device: {DEVICE}")

# ============================================================================
# 1. Load Configuration & Model
# ============================================================================
print("\n[1/6] Loading configuration and model...")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Load model with same architecture as checkpoint (check which one was trained)
# Try loading with pretrained=False first
try:
    model = load_model(num_classes=4, use_pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded checkpoint with use_pretrained=False (FallbackCNN)")
except RuntimeError:
    # Fall back to pretrained
    model = load_model(num_classes=4, use_pretrained=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print("✅ Loaded checkpoint with use_pretrained=True (CustomAST)")
model = model.to(DEVICE)
model.eval()

epochs_trained = checkpoint.get("epoch", "N/A")
best_se_ckpt = checkpoint.get("best_Se", "N/A")

print(f"✅ Model loaded from {CHECKPOINT_PATH}")
print(f"   Epochs trained: {epochs_trained}")
if best_se_ckpt != "N/A":
    print(f"   Best validation Se (during training): {best_se_ckpt:.4f}")

# ============================================================================
# 2. Load Test Dataset
# ============================================================================
print("\n[2/6] Loading test dataset...")

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
# 3. Run Inference on Test Set
# ============================================================================
print("\n[3/6] Running inference on test set...")

all_preds = []
all_labels = []

with torch.no_grad():
    for specs, labels in tqdm(test_loader, desc="Evaluating"):
        specs = specs.to(DEVICE)
        labels = labels.cpu().numpy()
        
        # Forward pass
        outputs = model(specs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"✅ Inference complete: {len(all_labels)} predictions")

# ============================================================================
# 4. Compute ICBHI Metrics
# ============================================================================
print("\n[4/6] Computing ICBHI metrics...")

metrics = compute_icbhi_metrics(all_labels, all_preds)

# Reference paper targets
REF_SE = 0.6831
REF_SP = 0.6789
REF_SCORE = 0.6810

print("\n" + "="*70)
print("BASELINE RESULTS (AST+SAM from Colab Training)")
print("="*70)
print(f"Sensitivity (Se):    {metrics['Se']:.4f}  ({metrics['Se']*100:.2f}%)")
print(f"Specificity (Sp):    {metrics['Sp']:.4f}  ({metrics['Sp']*100:.2f}%)")
print(f"ICBHI Score:         {metrics['Score']:.4f}  ({metrics['Score']*100:.2f}%)")
print("="*70)

print("\nCOMPARISON WITH REFERENCE PAPER (Atakanisik et al.):")
print(f"  Se:    {metrics['Se']*100:.2f}% vs {REF_SE*100:.2f}% (Δ {(metrics['Se']-REF_SE)*100:+.2f}%)")
print(f"  Sp:    {metrics['Sp']*100:.2f}% vs {REF_SP*100:.2f}% (Δ {(metrics['Sp']-REF_SP)*100:+.2f}%)")
print(f"  Score: {metrics['Score']*100:.2f}% vs {REF_SCORE*100:.2f}% (Δ {(metrics['Score']-REF_SCORE)*100:+.2f}%)")

# Save results to file
results_path = LOGS_DIR / "baseline_results.txt"
with open(results_path, "w") as f:
    f.write("Baseline AST+SAM\n")
    f.write("="*60 + "\n")
    f.write(f"Sensitivity (Se):    {metrics['Se']*100:.2f}%\n")
    f.write(f"Specificity (Sp):    {metrics['Sp']*100:.2f}%\n")
    f.write(f"ICBHI Score:         {metrics['Score']*100:.2f}%\n")
    f.write("="*60 + "\n")
    f.write(f"Epochs trained:      {epochs_trained}\n")
    f.write(f"Batch size:          16\n")
    f.write(f"Test samples:        {len(all_labels)}\n")
    f.write(f"\nReference Paper Targets:\n")
    f.write(f"  Se:    {REF_SE*100:.2f}%\n")
    f.write(f"  Sp:    {REF_SP*100:.2f}%\n")
    f.write(f"  Score: {REF_SCORE*100:.2f}%\n")

print(f"\n✅ Results saved to {results_path}")

# ============================================================================
# 5. Generate 4-Class Confusion Matrix
# ============================================================================
print("\n[5/6] Generating confusion matrix...")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Print per-class metrics
print("\nPer-Class Metrics (4-class):")
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
# 6. Generate t-SNE Plot (Optional - requires scikit-learn)
# ============================================================================
print("\n[6/6] Attempting to generate t-SNE visualization...")

try:
    from sklearn.manifold import TSNE
    
    # For t-SNE, we'll use model logits as features (simpler than extracting hidden layer)
    print("     Computing t-SNE from model logits...")
    
    all_logits = []
    with torch.no_grad():
        for specs, _ in tqdm(test_loader, desc="Extracting logits", leave=False):
            specs = specs.to(DEVICE)
            outputs = model(specs)
            all_logits.append(outputs.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    print(f"     Logits shape: {all_logits.shape}")
    
    print("     Running t-SNE (this may take 1-2 minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=0)
    embeddings_tsne = tsne.fit_transform(all_logits)
    
    # Plot t-SNE
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
    labels_map = {0: "Normal", 1: "Crackle", 2: "Wheeze", 3: "Both"}
    
    for class_id in range(4):
        mask = all_labels == class_id
        ax.scatter(
            embeddings_tsne[mask, 0],
            embeddings_tsne[mask, 1],
            c=colors[class_id],
            label=labels_map[class_id],
            alpha=0.6,
            s=30,
            edgecolors="none"
        )
    
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Baseline AST+SAM — t-SNE Embedding Visualization", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    tsne_path = LOGS_DIR / "tsne_baseline.png"
    plt.savefig(tsne_path, dpi=150, bbox_inches="tight")
    print(f"✅ t-SNE plot saved to {tsne_path}")
    plt.close()
    
except ImportError as e:
    print(f"⚠️  scikit-learn not available for t-SNE: {e}")
except Exception as e:
    print(f"⚠️  t-SNE generation failed: {e}")

# ============================================================================
# Final Summary & Next Steps
# ============================================================================
print("\n" + "="*70)
print("PHASE 4 COMPLETION & PHASE 5 DECISION")
print("="*70)

# Thresholds for acceptable baseline
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

print("\n📊 MISCLASSIFICATION ANALYSIS (from confusion matrix):")
print("\nTop misclassifications:")
misclass_count = 0
for i in range(4):
    for j in range(4):
        if i != j and cm[i, j] > 5:  # Only show significant errors
            pct = (cm[i, j] / cm[i, :].sum()) * 100
            print(f"  {CLASS_NAMES[i]:<10} → {CLASS_NAMES[j]:<10}: {cm[i, j]:>3} ({pct:>5.1f}%)")
            misclass_count += 1

print("\n🎯 NEXT STEPS:")
if se_pass and sp_pass and score_pass:
    print("✅ BASELINE REPRODUCED — Ready for PHASE 5 (Improvement Experiments)")
    print("   Recommendations for Phase 5:")
    print("   1. Focal Loss (γ=2) for minority class boosting (especially 'both')")
    print("   2. Stronger augmentation (SpecAugment with higher masking ratios)")
    print("   3. Threshold tuning for maximizing Se while maintaining Sp")
    print("   4. Ensemble methods if individual Se < 72%")
elif se < THRESHOLD_SE:
    print(f"⚠️  LOW SENSITIVITY ({se*100:.2f}% < {THRESHOLD_SE*100:.0f}%)")
    print("   Debugging recommended before Phase 5:")
    print("   1. Verify patient-level split has ZERO overlap between train/test")
    print("   2. Check model input shape: (B,1,128,T) correctly transposed to (B,T,128)")
    print("   3. Verify cyclic padding (not zero-padding) used during preprocessing")
    print("   4. Confirm weighted sampler active during training")
    print("   5. Review: are abnormal samples misclassified as normal?")
else:
    print(f"✓  Metrics acceptable ({se*100:.2f}%, {sp*100:.2f}%, {score*100:.2f}%)")
    print("   PHASE 5 improvements ready to begin")

print("\n" + "="*70)
print("Evaluation complete. All artifacts saved to logs/")
print("="*70)
