"""
Loss functions for ICBHI 4-class respiratory sound classification.

The reference paper uses cross-entropy with label smoothing 0.1 and a
WeightedRandomSampler. Two issues remain:

1. The Specificity (67.89%) is significantly lower than baselines that hit
   80%+ — the model over-corrects toward the abnormal classes.
2. Easy examples (clear majority-class normal samples) keep dominating the
   gradient even with sampling, since the network quickly becomes confident
   on them and stops learning fine-grained distinctions on the borderline.

Focal Loss addresses both points by down-weighting easy examples and
focussing gradient updates on hard misclassifications.

We provide:
    * FocalLoss  - the standard formulation with optional class alpha.
    * ClassBalancedFocalLoss - alpha is derived from effective number of
      samples (Cui et al. 2019), which is a principled choice for imbalanced
      medical datasets.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017).

    L = - alpha_y * (1 - p_y)^gamma * log(p_y)

    Args:
        alpha: per-class weighting tensor of shape (C,) or None for uniform.
        gamma: focusing parameter; gamma=0 reduces to weighted CE.
        label_smoothing: 0.0 to 1.0; smooths the one-hot target.
        reduction: 'mean', 'sum' or 'none'.
    """

    def __init__(
        self,
        alpha: Optional[Sequence[float]] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha is not None:
            alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_t)
        else:
            self.alpha = None  # type: ignore[assignment]
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)            # (B, C)
        probs = log_probs.exp()

        if self.label_smoothing > 0.0:
            with torch.no_grad():
                soft_target = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
                soft_target.scatter_(
                    1, target.unsqueeze(1), 1.0 - self.label_smoothing
                )
            ce_per_class = -soft_target * log_probs            # (B, C)
            modulator = (1.0 - probs) ** self.gamma            # (B, C)
            loss = (modulator * ce_per_class).sum(dim=-1)      # (B,)
        else:
            ce = F.nll_loss(log_probs, target, reduction="none")  # (B,)
            pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)
            loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            alpha_factor = self.alpha.to(logits.device)[target]
            loss = alpha_factor * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def class_balanced_alpha(
    class_counts: Sequence[int], beta: float = 0.999
) -> torch.Tensor:
    """Compute per-class alpha using effective number of samples.

    Cui et al., 'Class-Balanced Loss Based on Effective Number of Samples', CVPR 2019.
        E_n = (1 - beta^n_c) / (1 - beta)
        alpha_c = 1 / E_n  (then normalised so sum(alpha) = num_classes)

    Returns a (C,)-tensor suitable for FocalLoss(alpha=...).
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
    weights = 1.0 / (effective_num + 1e-12)
    weights = weights * (len(counts) / weights.sum())
    return weights.float()
