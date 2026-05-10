"""
Audio Spectrogram Transformer (AST) wrapper for ICBHI 4-class classification.

We start from the AudioSet-pretrained checkpoint
    "MIT/ast-finetuned-audioset-10-10-0.4593"
and replace the classification head with a small MLP for
    {0: Normal, 1: Crackle, 2: Wheeze, 3: Both}.

Key design choices vs. the reference paper:
    * Optional differential learning rate via `get_param_groups`:
      the pretrained backbone uses a smaller LR than the new head.
    * Optional layer-wise dropout in the head to regularise the limited
      ICBHI training set.
    * `get_features` returns the pooled embedding (768-d) so we can
      run t-SNE / kNN evaluation without retraining.
"""

from typing import List, Dict

import torch
import torch.nn as nn
from transformers import ASTModel


class CustomAST(nn.Module):
    """Audio Spectrogram Transformer with a 4-way classification head."""

    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ASTModel.from_pretrained(backbone_name)
        self.feature_dim = self.backbone.config.hidden_size  # 768 for AST-base

        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(head_dropout),
            nn.Linear(self.feature_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def get_features(self, input_values: torch.Tensor) -> torch.Tensor:
        """Return the pooled CLS embedding (B, feature_dim)."""
        outputs = self.backbone(input_values=input_values)
        # AST returns last_hidden_state of shape (B, num_patches+2, hidden);
        # `pooler_output` is the post-MLP-tanh CLS embedding.
        return outputs.pooler_output

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(input_values)
        return self.head(feats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_param_groups(
        self, base_lr: float, head_lr_multiplier: float = 10.0
    ) -> List[Dict]:
        """Return parameter groups with differential learning rates.

        The pretrained AudioSet backbone gets `base_lr` while the new
        classification head gets `base_lr * head_lr_multiplier`. This
        prevents the strong AudioSet features from being overwritten by
        the noisy gradients coming from the freshly-initialised head.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_multiplier},
        ]
