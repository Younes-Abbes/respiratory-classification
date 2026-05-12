"""
Audio Spectrogram Transformer (AST) wrapper for ICBHI 4-class classification.

We start from the AudioSet-pretrained checkpoint
    ``MIT/ast-finetuned-audioset-10-10-0.4593``
and replace the classification head with a small MLP for
    ``{0: Normal, 1: Crackle, 2: Wheeze, 3: Both}``.

Key differences vs. the previous version
----------------------------------------
* We **do not** override ``max_length`` / ``num_mel_bins`` any more — the
  AST positional embeddings are pretrained for 1024×128 inputs and any
  reshape destroys most of what makes the model useful.
* ``get_param_groups`` returns layer-wise-decayed learning rates for the
  AST backbone plus a larger LR for the new head.
* ``unfreeze_backbone`` lets the training script flip the frozen state
  after the warm-up phase.
"""

from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn as nn
from transformers import ASTModel


# ---------------------------------------------------------------------------
# AST + new head
# ---------------------------------------------------------------------------

class CustomAST(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        # IMPORTANT: load with the native config — do NOT change max_length
        # or num_mel_bins or the pretrained positional embeddings get
        # silently re-initialised.
        self.backbone = ASTModel.from_pretrained(backbone_name)
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
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Accept the various shapes the dataset / dataloader may produce
        and return ``(B, T, F)`` ready for ``ASTModel``."""

        if x.dim() == 4:
            # (B, 1, F, T) — legacy spectrogram layout from the CNN path.
            x = x.squeeze(1)        # (B, F, T)
            x = x.transpose(1, 2)   # (B, T, F)
        elif x.dim() == 3:
            # Already (B, T, F) — AST expects time first, freq second.
            pass
        else:
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}")
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_inputs(x)
        outputs = self.backbone(input_values=x)
        return outputs.pooler_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.head(feats)

    # ------------------------------------------------------------------
    # Optimizer parameter groups
    # ------------------------------------------------------------------
    def get_param_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 0.0,
        layerwise_lr_decay: float = 1.0,
    ) -> List[Dict]:
        """Return per-parameter-group settings for an optimizer.

        Layers closer to the output get a higher LR, deeper layers get a
        smaller LR (``layerwise_lr_decay ** depth``). This stabilises
        fine-tuning of a transformer that was pretrained on AudioSet.
        """

        groups: List[Dict] = []
        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight")

        # --- AST backbone ------------------------------------------------
        # 1. Embeddings (deepest, smallest LR).
        embed_params = list(self.backbone.embeddings.parameters())
        num_layers = len(self.backbone.encoder.layer)
        embed_lr = backbone_lr * (layerwise_lr_decay ** (num_layers + 1))
        groups.append({"params": [p for n, p in self.backbone.embeddings.named_parameters()
                                  if not any(nd in n for nd in no_decay_keys)],
                       "lr": embed_lr, "weight_decay": weight_decay})
        groups.append({"params": [p for n, p in self.backbone.embeddings.named_parameters()
                                  if any(nd in n for nd in no_decay_keys)],
                       "lr": embed_lr, "weight_decay": 0.0})

        # 2. Encoder layers — earlier layers get smaller LRs.
        for layer_idx, layer in enumerate(self.backbone.encoder.layer):
            depth = num_layers - layer_idx
            lr_layer = backbone_lr * (layerwise_lr_decay ** depth)
            groups.append({"params": [p for n, p in layer.named_parameters()
                                      if not any(nd in n for nd in no_decay_keys)],
                           "lr": lr_layer, "weight_decay": weight_decay})
            groups.append({"params": [p for n, p in layer.named_parameters()
                                      if any(nd in n for nd in no_decay_keys)],
                           "lr": lr_layer, "weight_decay": 0.0})

        # 3. Final layernorm of the backbone (closest to the head).
        if hasattr(self.backbone, "layernorm"):
            groups.append({"params": list(self.backbone.layernorm.parameters()),
                           "lr": backbone_lr, "weight_decay": 0.0})

        # --- Head --------------------------------------------------------
        groups.append({"params": [p for n, p in self.head.named_parameters()
                                  if not any(nd in n for nd in no_decay_keys)],
                       "lr": head_lr, "weight_decay": weight_decay})
        groups.append({"params": [p for n, p in self.head.named_parameters()
                                  if any(nd in n for nd in no_decay_keys)],
                       "lr": head_lr, "weight_decay": 0.0})

        # Drop empty groups (some named_parameters() lists may be empty).
        groups = [g for g in groups if len(list(g["params"])) > 0]
        return groups


# ---------------------------------------------------------------------------
# Loader (with a lightweight CPU fallback for local dry-runs)
# ---------------------------------------------------------------------------

def load_model(
    num_classes: int = 4,
    use_pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """Return ``CustomAST`` (if ``use_pretrained=True``) or a tiny CNN
    fallback for local dry-runs without GPU."""

    if use_pretrained:
        backbone_name = kwargs.pop(
            "backbone_name", "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        # Strip any keys that no longer apply (legacy YAML may still have them).
        for legacy_key in ("max_length", "num_mel_bins"):
            kwargs.pop(legacy_key, None)
        return CustomAST(
            num_classes=num_classes,
            backbone_name=backbone_name,
            **kwargs,
        )

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
            h = self.conv(x)
            h = h.view(h.size(0), -1)
            return self.fc(h)

    return FallbackCNN(num_classes)


def verify_checkpoint_is_ast(checkpoint_path: str) -> bool:
    """Print a small report and return ``True`` if the checkpoint stores
    AST weights, ``False`` if it stores the FallbackCNN."""

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    keys = list(state.keys())

    is_ast = any("backbone" in k or "head" in k for k in keys)
    is_fallback = any(k.startswith(("conv.", "fc.")) for k in keys)

    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"  AST keys present     : {is_ast}")
    print(f"  Fallback CNN present : {is_fallback}")
    print(f"  Epoch saved at       : {ckpt.get('epoch', 'unknown')}")
    print(f"  Best val Score       : {ckpt.get('best_score', 'n/a')}")
    print(f"  Best val Se          : {ckpt.get('best_Se', 'n/a')}")
    print(f"  Architecture         : {'AST' if is_ast else 'FALLBACK CNN — DO NOT USE'}")
    print("=" * 50)
    return is_ast
