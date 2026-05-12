"""Evaluation utilities (Task 15)."""


def compute_icbhi_metrics(y_true, y_pred):
    """
    Compute ICBHI official metrics: Sensitivity (Se), Specificity (Sp), and Score.
    
    The ICBHI challenge evaluates BINARY classification: Normal vs. Abnormal
    Abnormal = crackle OR wheeze OR both (labels 1, 2, 3)
    
    Args:
        y_true: array of true labels (0, 1, 2, 3)
        y_pred: array of predicted labels (0, 1, 2, 3)
    
    Returns:
        dict with keys: "Se" (sensitivity), "Sp" (specificity), "Score" (average)
    """
    import numpy as np
    
    # Convert to binary: 0=normal, 1=abnormal
    binary_true = (y_true > 0).astype(int)
    binary_pred = (y_pred > 0).astype(int)
    
    # Compute confusion matrix
    TP = ((binary_pred == 1) & (binary_true == 1)).sum()
    TN = ((binary_pred == 0) & (binary_true == 0)).sum()
    FP = ((binary_pred == 1) & (binary_true == 0)).sum()
    FN = ((binary_pred == 0) & (binary_true == 1)).sum()
    
    # Compute Se and Sp
    Se = TP / (TP + FN + 1e-8)   # Sensitivity / Recall
    Sp = TN / (TN + FP + 1e-8)   # Specificity
    Score = (Se + Sp) / 2.0       # ICBHI Score
    
    return {"Se": Se, "Sp": Sp, "Score": Score}
