#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the official ICBHI test split.

Usage
-----
    python scripts/evaluate_quick.py \
        --checkpoint checkpoints/best_model.pt \
        --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import ICBHIASTDataset, ICBHIDataset
from src.metrics import icbhi_metrics
from src.model import load_model, verify_checkpoint_is_ast


CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Ckpt   : {args.checkpoint}")
    print(f"Config : {args.config}")
    print(f"Test   : {args.csv}")
    print("=" * 70)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    is_ast = verify_checkpoint_is_ast(args.checkpoint)

    # ------------------------------------------------------------------
    # Build model + dataset matching the checkpoint architecture.
    # ------------------------------------------------------------------
    if is_ast:
        try:
            from transformers import ASTFeatureExtractor as _FE
        except ImportError:
            from transformers import AutoFeatureExtractor as _FE  # type: ignore
        feature_extractor = _FE.from_pretrained(
            str(config.get("backbone_name",
                           "MIT/ast-finetuned-audioset-10-10-0.4593"))
        )
        model = load_model(
            num_classes=int(config.get("num_classes", 4)),
            use_pretrained=True,
            backbone_name=str(config.get(
                "backbone_name", "MIT/ast-finetuned-audioset-10-10-0.4593"
            )),
            head_hidden_dim=int(config.get("head_hidden_dim", 256)),
            head_dropout=float(config.get("head_dropout", 0.3)),
            freeze_backbone=False,
        )
        dataset = ICBHIASTDataset(
            csv_path=args.csv,
            config=config,
            feature_extractor=feature_extractor,
            augment=False,
        )
    else:
        model = load_model(num_classes=4, use_pretrained=False)
        dataset = ICBHIDataset(csv_path=args.csv, config=config, augment=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=device.type == "cuda")

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Evaluating"):
            specs = specs.to(device, non_blocking=True)
            logits = model(specs)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true_list.append(labels.numpy())
            y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    metrics = icbhi_metrics(y_true, y_pred)
    se, sp, score = metrics["sensitivity"], metrics["specificity"], metrics["score"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Sensitivity (Se): {se*100:6.2f}%")
    print(f"Specificity (Sp): {sp*100:6.2f}%")
    print(f"ICBHI Score:      {score*100:6.2f}%")
    print("=" * 70)

    REF = {"Se": 0.6831, "Sp": 0.6789, "Score": 0.6810}
    print("vs reference paper:")
    print(f"  Se    {se*100:6.2f}% vs {REF['Se']*100:.2f}%   (delta {(se-REF['Se'])*100:+.2f}%)")
    print(f"  Sp    {sp*100:6.2f}% vs {REF['Sp']*100:.2f}%   (delta {(sp-REF['Sp'])*100:+.2f}%)")
    print(f"  Score {score*100:6.2f}% vs {REF['Score']*100:.2f}%   (delta {(score-REF['Score'])*100:+.2f}%)")

    # -------- Save artifacts ------------------------------------------
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(logs_dir / "baseline_results.txt", "w") as f:
        f.write("Final ICBHI evaluation\n")
        f.write("=" * 40 + "\n")
        f.write(f"Sensitivity (Se): {se*100:6.2f}%\n")
        f.write(f"Specificity (Sp): {sp*100:6.2f}%\n")
        f.write(f"ICBHI Score:      {score*100:6.2f}%\n")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    plt.title(f"Confusion matrix — Score={score*100:.2f}%")
    plt.tight_layout()
    plt.savefig(logs_dir / "confusion_matrix_baseline.png", dpi=150)
    plt.close()

    print("\nPer-class recall:")
    for i, name in enumerate(CLASS_NAMES):
        support = (y_true == i).sum()
        if support == 0:
            continue
        recall = cm[i, i] / support
        print(f"  {name:8s}: {recall*100:6.2f}%  (support={support})")


if __name__ == "__main__":
    main()
