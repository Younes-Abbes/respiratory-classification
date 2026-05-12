"""Training entry point for the ICBHI 2017 AST baseline.

Pipeline overview
-----------------
1.  Load YAML config from ``configs/baseline.yaml``.
2.  Build a patient-level train / val split from ``data/splits/train.csv``.
    The official ``data/splits/test.csv`` is used **only** for the final
    evaluation, never for checkpoint selection.
3.  Build the AST model and run a short warm-up where only the head
    is trained (backbone frozen). This lets the random head settle
    before we start moving the pretrained weights.
4.  Unfreeze the backbone and continue training with:
      * SAM (sharpness-aware minimization) on top of AdamW
      * Layer-wise LR decay for the backbone, larger LR for the head
      * Linear warm-up + cosine LR schedule
      * Focal Loss with class-balanced alphas + label smoothing
      * Square-root frequency sampling (less aggressive than inverse)
      * Optional Mixup + SpecAugment
      * Gradient clipping
      * Mixed precision (AMP)
5.  After each epoch we log to CSV, save ``latest.pt`` and update
    ``best_model.pt`` (selected on val Score). Early stopping kicks in
    after ``early_stopping_patience`` epochs without improvement.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .augmentations import SpecAugment, WaveformAugment, mixup_data
from .dataset import (
    ICBHIASTDataset,
    ICBHIDataset,
    make_balanced_sampler,
    patient_level_train_val_split,
)
from .losses import FocalLoss, class_balanced_alpha
from .metrics import icbhi_metrics
from .model import load_model
from .sam import SAM
from .utils import set_seed


CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


# ---------------------------------------------------------------------------
# CSV / checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  checkpoint saved -> {path}", flush=True)


def log_metrics_csv(log_path: Path, row: Dict) -> None:
    """Append a row to a CSV; rotate the file if its schema changed.

    Schema rotation guards against running a new training schema on top of
    an old ``metrics.csv`` left in the log directory by a previous run.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    # If header already exists but does not match, rotate the old file.
    if not write_header:
        with open(log_path, "r", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if list(existing_header) != list(row.keys()):
            backup = log_path.with_suffix(".prev.csv")
            try:
                log_path.rename(backup)
                print(f"  [log] schema mismatch, old CSV moved to {backup}",
                      flush=True)
            except OSError:
                log_path.unlink(missing_ok=True)
            write_header = True
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model, config: dict, head_only: bool = False):
    """Build SAM(AdamW) with the right parameter groups.

    Args:
        head_only: if True only train the new MLP head (used during warmup).
    """

    backbone_lr = float(config.get("backbone_lr", 1e-5))
    head_lr = float(config.get("head_lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 5e-4))
    layerwise_lr_decay = float(config.get("layerwise_lr_decay", 0.85))
    rho = float(config.get("rho", 0.05))

    is_ast = hasattr(model, "backbone") and hasattr(model, "head")

    if head_only and is_ast:
        warmup_head_lr = float(config.get("warmup_head_lr", head_lr * 5))
        param_groups = [
            {"params": [p for p in model.head.parameters() if p.requires_grad],
             "lr": warmup_head_lr, "weight_decay": weight_decay},
        ]
    elif is_ast:
        param_groups = model.get_param_groups(
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            weight_decay=weight_decay,
            layerwise_lr_decay=layerwise_lr_decay,
        )
    else:
        # FallbackCNN path — single group.
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                         "lr": head_lr, "weight_decay": weight_decay}]

    if rho > 0.0:
        return SAM(param_groups, torch.optim.AdamW, rho=rho), True
    return torch.optim.AdamW(param_groups), False


def cosine_warmup_lr(
    current_step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> float:
    """Return the LR multiplier for a linear warm-up then cosine decay."""
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

def _move_to_device(x, device):
    return x.to(device, non_blocking=True) if torch.is_tensor(x) else x


def _forward_loss(model, criterion, specs, labels, mixup_alpha, mixup_prob,
                  spec_augment, training: bool):
    """One forward pass, optionally with mixup + SpecAugment."""
    if training and spec_augment is not None:
        specs = spec_augment(specs)

    apply_mixup = (training and mixup_alpha > 0.0
                   and torch.rand(1).item() < mixup_prob)
    if apply_mixup:
        mixed, y_a, y_b, lam = mixup_data(specs, labels, mixup_alpha)
        logits = model(mixed)
        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        return logits, loss

    logits = model(specs)
    loss = criterion(logits, labels)
    return logits, loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    *,
    use_sam: bool,
    scaler: Optional["torch.cuda.amp.GradScaler"],
    use_amp: bool,
    grad_clip: float,
    spec_augment,
    mixup_alpha: float,
    mixup_prob: float,
    lr_scheduler_step,
    epoch_idx: int,
    total_epochs: int,
    log_every: int = 50,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    n_steps = len(loader)
    start_time = time.time()

    for i, (specs, labels) in enumerate(loader):
        specs = _move_to_device(specs, device)
        labels = _move_to_device(labels, device)

        if use_sam:
            # ---- SAM step 1 (climb to the perturbed point) ---------------
            # We wrap the forward pass in autocast() to save memory on T4 but
            # do the backward / SAM perturbation in fp32 to avoid scaling
            # issues (SAM is not compatible with GradScaler).
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = _forward_loss(
                        model, criterion, specs, labels,
                        mixup_alpha, mixup_prob, spec_augment, training=True,
                    )
                loss = loss.float()
            else:
                _, loss = _forward_loss(
                    model, criterion, specs, labels,
                    mixup_alpha, mixup_prob, spec_augment, training=True,
                )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    grad_clip,
                )
            optimizer.first_step(zero_grad=True)

            # ---- SAM step 2 (gradient at the perturbed point) ------------
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, loss2 = _forward_loss(
                        model, criterion, specs, labels,
                        mixup_alpha, mixup_prob, spec_augment, training=True,
                    )
                loss2 = loss2.float()
            else:
                logits, loss2 = _forward_loss(
                    model, criterion, specs, labels,
                    mixup_alpha, mixup_prob, spec_augment, training=True,
                )
            loss2.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    grad_clip,
                )
            optimizer.second_step(zero_grad=True)
            batch_loss = float(loss2.detach().cpu())
        else:
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, loss = _forward_loss(
                        model, criterion, specs, labels,
                        mixup_alpha, mixup_prob, spec_augment, training=True,
                    )
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = _forward_loss(
                    model, criterion, specs, labels,
                    mixup_alpha, mixup_prob, spec_augment, training=True,
                )
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        grad_clip,
                    )
                optimizer.step()
            batch_loss = float(loss.detach().cpu())

        if lr_scheduler_step is not None:
            lr_scheduler_step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

        total_loss += batch_loss
        n_batches += 1

        if (i + 1) % log_every == 0 or (i + 1) == n_steps:
            elapsed = time.time() - start_time
            it_per_s = (i + 1) / max(1e-6, elapsed)
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  ep {epoch_idx+1}/{total_epochs}  "
                f"batch {i+1}/{n_steps}  loss={batch_loss:.4f}  "
                f"avg={total_loss/n_batches:.4f}  lr={cur_lr:.2e}  "
                f"{it_per_s:.2f} it/s",
                flush=True,
            )

    y_true_arr = np.concatenate(y_true) if y_true else np.zeros(0, dtype=int)
    y_pred_arr = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=int)
    train_metrics = icbhi_metrics(y_true_arr, y_pred_arr)
    return total_loss / max(1, n_batches), train_metrics


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = False) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    for specs, labels in loader:
        specs = _move_to_device(specs, device)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(specs)
        else:
            logits = model(specs)
        preds = logits.argmax(dim=1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true_arr = np.concatenate(y_true) if y_true else np.zeros(0, dtype=int)
    y_pred_arr = np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=int)
    return icbhi_metrics(y_true_arr, y_pred_arr), y_true_arr, y_pred_arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a 2-epoch sanity check with very few batches.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <checkpoint-dir>/latest.pt if present.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args(argv)

    stdout = getattr(sys, "stdout", None)
    if stdout is not None and hasattr(stdout, "reconfigure"):
        stdout.reconfigure(line_buffering=True)

    # --- config --------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        config = cast(dict, yaml.safe_load(f) or {})

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    use_pretrained = bool(config.get("use_pretrained", True))
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"

    epochs = args.epochs or int(config.get("epochs", 40))
    if args.dry_run:
        epochs = 2

    # --- splits --------------------------------------------------------
    train_csv = Path("data/splits/train.csv")
    test_csv = Path("data/splits/test.csv")
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            "Missing data/splits/{train,test}.csv — run scripts/prepare_data.py first."
        )

    val_frac = float(config.get("val_split_frac", 0.15))
    val_seed = int(config.get("val_split_seed", 17))
    train_df, val_df = patient_level_train_val_split(str(train_csv), val_frac, val_seed)

    splits_dir = Path("data/splits")
    train_df.to_csv(splits_dir / "_train_used.csv", index=False)
    val_df.to_csv(splits_dir / "_val_used.csv", index=False)
    print(f"[data] train={len(train_df)} val={len(val_df)} test=?", flush=True)

    # --- model + feature extractor ------------------------------------
    if use_pretrained:
        # The AST processor lives in transformers, but the package layout
        # changed across versions, so we try both spellings.
        try:
            from transformers import ASTFeatureExtractor as _FE
        except ImportError:  # pragma: no cover
            from transformers import AutoFeatureExtractor as _FE  # type: ignore
        backbone_name = str(config.get(
            "backbone_name", "MIT/ast-finetuned-audioset-10-10-0.4593"
        ))
        feature_extractor = _FE.from_pretrained(backbone_name)
        sampler_labels_source = "ast"
    else:
        feature_extractor = None
        sampler_labels_source = "cnn"

    # --- datasets ------------------------------------------------------
    if use_pretrained:
        # Seed=None so each DataLoader worker gets its own RNG state; otherwise
        # the augmentations are identical across workers (and across epochs).
        wave_aug = (
            WaveformAugment(seed=None)
            if config.get("use_waveaug", True) else None
        )
        train_ds: torch.utils.data.Dataset = ICBHIASTDataset(
            csv_path=str(splits_dir / "_train_used.csv"),
            config=config,
            feature_extractor=feature_extractor,
            augment=bool(config.get("augment", True)),
            wave_aug=wave_aug,
        )
        val_ds: torch.utils.data.Dataset = ICBHIASTDataset(
            csv_path=str(splits_dir / "_val_used.csv"),
            config=config,
            feature_extractor=feature_extractor,
            augment=False,
        )
        test_ds: torch.utils.data.Dataset = ICBHIASTDataset(
            csv_path=str(test_csv),
            config=config,
            feature_extractor=feature_extractor,
            augment=False,
        )
        train_labels = cast(ICBHIASTDataset, train_ds).get_labels()
    else:
        train_ds = ICBHIDataset(str(splits_dir / "_train_used.csv"),
                                config=config, augment=bool(config.get("augment", True)))
        val_ds = ICBHIDataset(str(splits_dir / "_val_used.csv"),
                              config=config, augment=False)
        test_ds = ICBHIDataset(str(test_csv), config=config, augment=False)
        train_labels = cast(ICBHIDataset, train_ds).get_labels()

    print(f"[data] feature extractor source: {sampler_labels_source}", flush=True)

    # --- sampler / loaders --------------------------------------------
    sampler_strategy = str(config.get("sampler", "sqrt"))
    sampler = make_balanced_sampler(train_labels, strategy=sampler_strategy)
    bs = int(config.get("batch_size", 16))
    if args.dry_run:
        bs = min(bs, 2)

    num_workers = int(config.get("num_workers", 2))
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=bs,
        sampler=sampler, shuffle=sampler is None,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    # --- model --------------------------------------------------------
    model = load_model(
        num_classes=int(config.get("num_classes", 4)),
        use_pretrained=use_pretrained,
        backbone_name=str(config.get(
            "backbone_name", "MIT/ast-finetuned-audioset-10-10-0.4593"
        )),
        head_hidden_dim=int(config.get("head_hidden_dim", 256)),
        head_dropout=float(config.get("head_dropout", 0.3)),
        freeze_backbone=bool(config.get("freeze_backbone", False)),
    )
    model.to(device)

    # Sanity print on the model.
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params: total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M",
          flush=True)

    # --- loss ----------------------------------------------------------
    counts = np.bincount(train_labels, minlength=4).tolist()
    print(f"[data] train class counts: {dict(zip(CLASS_NAMES, counts))}", flush=True)

    if str(config.get("loss", "focal")).lower() == "focal":
        alpha = class_balanced_alpha(counts, beta=float(config.get("class_balanced_beta", 0.999)))
        criterion = FocalLoss(
            alpha=alpha,
            gamma=float(config.get("focal_gamma", 2.0)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )
        print(f"[loss] FocalLoss(gamma={config.get('focal_gamma', 2.0)}, "
              f"alpha={[round(a, 3) for a in alpha.tolist()]})", flush=True)
    else:
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=float(config.get("label_smoothing", 0.0))
        )

    # --- spec augment --------------------------------------------------
    spec_augment = None
    if use_pretrained and bool(config.get("use_specaugment", True)):
        spec_augment = SpecAugment(
            freq_mask_param=int(config.get("spec_freq_mask", 24)),
            time_mask_param=int(config.get("spec_time_mask", 96)),
            num_freq_masks=int(config.get("spec_num_freq_masks", 2)),
            num_time_masks=int(config.get("spec_num_time_masks", 2)),
        )

    # --- optimizer + scheduler ----------------------------------------
    warmup_epochs = int(config.get("warmup_epochs", 0))
    head_only = use_pretrained and warmup_epochs > 0

    if head_only and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        print(f"[warmup] backbone frozen for {warmup_epochs} epoch(s)", flush=True)

    optimizer, use_sam = build_optimizer(model, config, head_only=head_only)

    total_steps = max(1, len(train_loader)) * epochs
    warmup_steps = max(1, len(train_loader)) * int(config.get("lr_warmup_epochs", 1))
    min_lr_ratio = float(config.get("lr_min", 1e-7)) / max(
        float(config.get("backbone_lr", 1e-5)), 1e-12
    )
    global_step = {"v": 0}
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_scheduler_step():
        mult = cosine_warmup_lr(
            global_step["v"], total_steps, warmup_steps, min_lr_ratio
        )
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * mult
        global_step["v"] += 1

    # --- AMP scaler ----------------------------------------------------
    scaler = torch.cuda.amp.GradScaler() if (use_amp and not use_sam) else None

    # --- resume --------------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    log_path = log_dir / "metrics.csv"

    start_epoch = 0
    best_score = -1.0
    best_se = 0.0
    epochs_no_improve = 0
    latest_path = ckpt_dir / "latest.pt"
    if args.resume and latest_path.exists():
        ck = torch.load(latest_path, map_location=device)
        try:
            model.load_state_dict(ck["model_state"])
        except RuntimeError as exc:
            print(f"[resume] state_dict mismatch — skipping resume ({exc})", flush=True)
        else:
            start_epoch = int(ck.get("epoch", -1)) + 1
            best_score = float(ck.get("best_score", -1.0))
            best_se = float(ck.get("best_Se", 0.0))
            print(f"[resume] from epoch {start_epoch}, best_score={best_score:.4f}",
                  flush=True)
            # If we resume past the warm-up boundary, unfreeze the backbone
            # and rebuild the optimizer with the full set of param groups.
            if (use_pretrained and warmup_epochs > 0
                    and start_epoch >= warmup_epochs
                    and hasattr(model, "unfreeze_backbone")):
                print("[resume] past warmup boundary -> unfreezing backbone",
                      flush=True)
                model.unfreeze_backbone()
                optimizer, use_sam = build_optimizer(model, config, head_only=False)
                base_lrs = [g["lr"] for g in optimizer.param_groups]
                scaler = torch.cuda.amp.GradScaler() if (use_amp and not use_sam) else None

    # --- main loop -----------------------------------------------------
    patience = int(config.get("early_stopping_patience", 12))
    grad_clip = float(config.get("gradient_clip_norm", 1.0))
    mixup_alpha = float(config.get("mixup_alpha", 0.2)) if bool(config.get("use_mixup", True)) else 0.0
    mixup_prob = float(config.get("mixup_prob", 0.5))

    max_train_batches = 5 if args.dry_run else None
    max_val_batches = 5 if args.dry_run else None
    # Trim loaders for dry run to keep things fast.
    if max_train_batches is not None:
        from itertools import islice

        class _Limited:
            def __init__(self, loader, n):
                self.loader = loader
                self.n = n
            def __iter__(self):
                return iter(islice(self.loader, self.n))
            def __len__(self):
                return min(self.n, len(self.loader))

        train_loader = _Limited(train_loader, max_train_batches)
        val_loader = _Limited(val_loader, max_val_batches)

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*70}", flush=True)
        print(f"[epoch] {epoch+1}/{epochs}", flush=True)

        # --- end-of-warmup unfreeze --------------------------------------
        if (use_pretrained and warmup_epochs > 0
                and epoch == warmup_epochs and hasattr(model, "unfreeze_backbone")):
            print("[warmup] unfreezing backbone — rebuilding optimizer", flush=True)
            model.unfreeze_backbone()
            optimizer, use_sam = build_optimizer(model, config, head_only=False)
            base_lrs = [g["lr"] for g in optimizer.param_groups]
            # Reset the LR schedule from this point on, so the post-warmup
            # phase uses its own cosine cycle.
            global_step["v"] = 0
            total_steps_remaining = max(1, len(train_loader)) * (epochs - warmup_epochs)
            warmup_steps_remaining = max(1, len(train_loader)) * int(
                config.get("lr_warmup_epochs", 1)
            )
            min_ratio = float(config.get("lr_min", 1e-7)) / max(
                float(config.get("backbone_lr", 1e-5)), 1e-12
            )

            def lr_scheduler_step():  # noqa: F811
                mult = cosine_warmup_lr(
                    global_step["v"], total_steps_remaining,
                    warmup_steps_remaining, min_ratio,
                )
                for g, base in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = base * mult
                global_step["v"] += 1

            scaler = torch.cuda.amp.GradScaler() if (use_amp and not use_sam) else None

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_sam=use_sam,
            scaler=scaler, use_amp=use_amp,
            grad_clip=grad_clip,
            spec_augment=spec_augment,
            mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
            lr_scheduler_step=lr_scheduler_step,
            epoch_idx=epoch, total_epochs=epochs,
        )

        val_metrics, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)
        val_se = float(val_metrics["sensitivity"])
        val_sp = float(val_metrics["specificity"])
        val_score = float(val_metrics["score"])

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch+1}] "
            f"train_loss={train_loss:.4f}  "
            f"train Se={train_metrics['sensitivity']:.4f} "
            f"Sp={train_metrics['specificity']:.4f} "
            f"Score={train_metrics['score']:.4f}  | "
            f"val Se={val_se:.4f} Sp={val_sp:.4f} Score={val_score:.4f}  "
            f"lr={cur_lr:.2e}",
            flush=True,
        )

        log_metrics_csv(log_path, {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_Se": round(float(train_metrics["sensitivity"]), 4),
            "train_Sp": round(float(train_metrics["specificity"]), 4),
            "train_Score": round(float(train_metrics["score"]), 4),
            "val_Se": round(val_se, 4),
            "val_Sp": round(val_sp, 4),
            "val_Score": round(val_score, 4),
            "lr": cur_lr,
        })

        # --- checkpoints ------------------------------------------------
        improved = val_score > best_score
        if improved:
            best_score = val_score
            best_se = val_se
            epochs_no_improve = 0
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_score": best_score,
                "best_Se": best_se,
                "best_Sp": val_sp,
                "config": config,
            }, ckpt_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_score": best_score,
            "best_Se": best_se,
            "config": config,
        }, ckpt_dir / "latest.pt")

        if epochs_no_improve >= patience:
            print(f"\n[early-stop] no val Score improvement for "
                  f"{epochs_no_improve} epochs — stopping.", flush=True)
            break

    print(f"\n[done] best val Score={best_score:.4f}  Se={best_se:.4f}", flush=True)

    # --- final eval on the held-out test set --------------------------
    print("\n[final] evaluating best checkpoint on data/splits/test.csv ...",
          flush=True)
    best_ck = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ck["model_state"])
    test_metrics, y_true, y_pred = evaluate(model, test_loader, device, use_amp=use_amp)
    print(
        f"[test] Se={test_metrics['sensitivity']:.4f}  "
        f"Sp={test_metrics['specificity']:.4f}  "
        f"Score={test_metrics['score']:.4f}",
        flush=True,
    )

    # Write the final test metrics to a separate file so the per-epoch CSV
    # keeps a stable schema (mixing extra columns mid-file breaks pandas).
    with open(log_dir / "test_results.txt", "w") as f:
        f.write("Final test-set evaluation\n")
        f.write("=" * 40 + "\n")
        f.write(f"Sensitivity (Se): {test_metrics['sensitivity']*100:6.2f}%\n")
        f.write(f"Specificity (Sp): {test_metrics['specificity']*100:6.2f}%\n")
        f.write(f"ICBHI Score:      {test_metrics['score']*100:6.2f}%\n")
        f.write(f"Best val Score:   {best_score*100:6.2f}%\n")
        f.write(f"Best val Se:      {best_se*100:6.2f}%\n")

    # Save final predictions to disk for downstream analysis.
    np.savez(
        log_dir / "test_predictions.npz",
        y_true=y_true, y_pred=y_pred,
        Se=test_metrics["sensitivity"],
        Sp=test_metrics["specificity"],
        Score=test_metrics["score"],
    )


if __name__ == "__main__":
    main()
