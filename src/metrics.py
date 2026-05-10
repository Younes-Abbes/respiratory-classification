"""
Evaluation metrics for the ICBHI 2017 Challenge.

The official ICBHI protocol collapses the 4 classes into a binary task:
    Normal (0)    vs   Abnormal (1, 2, 3 = Crackle, Wheeze, Both)

A prediction of any abnormal class for a sample whose true label is
abnormal counts as a True Positive (intra-abnormal confusion is not
penalised by the score).

Metrics:
    * Sensitivity (Se) = TP / (TP + FN)
        Among truly abnormal cycles, fraction predicted as abnormal.
    * Specificity (Sp) = TN / (TN + FP)
        Among truly normal cycles, fraction predicted as normal.
    * ICBHI Score = (Se + Sp) / 2

We additionally compute:
    * 4-class confusion matrix
    * Per-class precision / recall / f1
    * Harmonic mean of Se and Sp (more punishing than the arithmetic mean)
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def icbhi_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the official ICBHI Sensitivity, Specificity and Score.

    Args:
        y_true: 1-D array of integer labels in {0, 1, 2, 3}.
        y_pred: 1-D array of predicted labels in {0, 1, 2, 3}.

    Returns:
        Dictionary with keys: sensitivity, specificity, score, harmonic_score.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # Specificity: correct normal predictions / total normal samples.
    sp_denom = cm[0, :].sum()
    sp = cm[0, 0] / sp_denom if sp_denom > 0 else 0.0

    # Sensitivity: any abnormal-correct prediction / total abnormal samples.
    se_denom = cm[1:, :].sum()
    se = cm[1:, 1:].sum() / se_denom if se_denom > 0 else 0.0

    score = (se + sp) / 2.0
    harmonic = 2 * se * sp / (se + sp + 1e-12)

    return {
        "sensitivity": float(se),
        "specificity": float(sp),
        "score": float(score),
        "harmonic_score": float(harmonic),
    }


def detailed_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] | None = None
) -> Dict:
    """Full per-class report: macro / weighted F1 + classification report."""
    if class_names is None:
        class_names = ["Normal", "Crackle", "Wheeze", "Both"]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    return {
        "confusion_matrix": cm,
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "classification_report": text,
    }
