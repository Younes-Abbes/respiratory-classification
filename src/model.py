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
from transformers import ASTModel, ASTConfig

class CustomAST(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
        freeze_backbone: bool = False,
        max_length: int = 345,      # ← your actual time frames
        num_mel_bins: int = 128,    # ← your mel bins
    ) -> None:
        super().__init__()

        config = ASTConfig.from_pretrained(backbone_name)
        config.max_length = max_length
        config.num_mel_bins = num_mel_bins

        self.backbone = ASTModel.from_pretrained(
            backbone_name,
            config=config,
            ignore_mismatched_sizes=True   # ← allows pos embedding reshape
        )
        self.feature_dim = self.backbone.config.hidden_size  # 768

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
        """Return pooled CLS embedding. Input: (B, 1, 128, T) or (B, T, 128)."""
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)       # (B, 128, T)
            input_values = input_values.transpose(1, 2)  # (B, T, 128)
        outputs = self.backbone(input_values=input_values)
        return outputs.pooler_output

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(input_values)
        return self.head(feats)

    def get_param_groups(
        self, base_lr: float, head_lr_multiplier: float = 10.0
    ):
        return [
            {"params": list(self.backbone.parameters()), "lr": base_lr},
            {"params": list(self.head.parameters()), "lr": base_lr * head_lr_multiplier},
        ]

def load_model(num_classes: int = 4, use_pretrained: bool = False, **kwargs) -> nn.Module:
    """Load the AST-based model or a lightweight fallback for local dry-runs.

    Args:
        num_classes: number of output classes.
        use_pretrained: if True, instantiate CustomAST which will load the
            pretrained AudioSet weights (may download large checkpoints).

    Returns:
        nn.Module ready for training/evaluation.
    """
    if use_pretrained:
        # This will instantiate the AST backbone and may download weights.
        return CustomAST(num_classes=num_classes, **kwargs)

    # Lightweight fallback CNN: accepts (B, 1, 128, T) and returns (B, num_classes)
    class FallbackCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 1, 128, T)
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc(h)

    return FallbackCNN(num_classes)
