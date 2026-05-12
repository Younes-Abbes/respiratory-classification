"""Training loop implementation (Tasks 14-17)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import ICBHIDataset, make_weighted_sampler
from src.model import load_model
from src.losses import FocalLoss, class_balanced_alpha
from src.metrics import icbhi_metrics
from src.sam import SAM


def get_optimizer(model, config):
    base_lr = float(config.get("lr", 1e-5))
    weight_decay = float(config.get("weight_decay", 1e-4))
    rho = float(config.get("rho", 0.0))

    if rho > 0.0:
        opt = SAM(
            model.parameters(), torch.optim.AdamW,
            rho=rho, lr=base_lr, weight_decay=weight_decay
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
    return opt


def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_sam=False, max_batches=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, (specs, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        specs  = specs.to(device)
        labels = labels.to(device)

        if use_sam:
            # SAM first step
            logits = model(specs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # SAM second step
            logits2 = model(specs)
            loss2   = criterion(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            logits = model(specs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches  += 1

        # ── flush every batch so Colab shows progress immediately ──
        print(f"  batch {i+1}/{len(loader) if max_batches is None else max_batches}"
              f"  loss={loss.item():.4f}", flush=True)

    return total_loss / max(1, n_batches)


def validate(model, loader, device, max_batches=None):
    import numpy as np

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for i, (specs, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            logits = model(specs.to(device))
            preds  = logits.argmax(dim=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    return icbhi_metrics(
        __import__("numpy").concatenate(y_true),
        __import__("numpy").concatenate(y_pred)
    )


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  ✅ Checkpoint saved → {path}", flush=True)


def log_metrics_csv(log_path, row):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--resume",         type=str, default="")
    parser.add_argument("--epochs",         type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir",        type=str, default="logs")
    args = parser.parse_args(argv)

    # ── force stdout flush (critical for Colab) ──
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    import yaml
    with open(Path("configs") / "baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI epochs override config
    n_epochs = args.epochs or int(config.get("epochs", 50))
    if args.dry_run:
        n_epochs = 2
    print(f"Training for {n_epochs} epochs", flush=True)

    # ── datasets ──
    train_ds = ICBHIDataset("data/splits/train.csv", config, augment=not args.dry_run)
    test_ds  = ICBHIDataset("data/splits/test.csv",  config, augment=False)
    print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples", flush=True)

    import numpy as np
    labels  = np.array(train_ds.df["label"].tolist(), dtype=int)
    sampler = make_weighted_sampler(labels)
    bs      = int(config.get("batch_size", 8))

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,    num_workers=2, pin_memory=True)

    # ── model ──
    use_pretrained  = bool(config.get("use_pretrained",  False))
    freeze_backbone = bool(config.get("freeze_backbone", True))
    print(f"Loading model (pretrained={use_pretrained}, freeze={freeze_backbone})...", flush=True)
    model = load_model(use_pretrained=use_pretrained, freeze_backbone=freeze_backbone)
    model.to(device)
    print("Model ready.", flush=True)

    # ── resume ──
    start_epoch = 0
    best_Se     = 0.0
    if args.resume:
        ckpt_path = Path(args.checkpoint_dir) / "best_model.pt"
        if ckpt_path.exists():
            ckpt        = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_Se     = ckpt.get("best_Se", 0.0)
            print(f"Resumed from epoch {start_epoch}, best Se={best_Se:.4f}", flush=True)

    # ── loss ──
    if config.get("loss", "ce") == "focal":
        counts = train_ds.df["label"].value_counts().reindex(range(4), fill_value=0).tolist()
        alpha  = class_balanced_alpha(counts)
        criterion = FocalLoss(
            alpha=alpha,
            gamma=float(config.get("focal_gamma", 2.0)),
            label_smoothing=float(config.get("label_smoothing", 0.0))
        )
    else:
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=float(config.get("label_smoothing", 0.0))
        )

    optimizer = get_optimizer(model, config)
    use_sam   = float(config.get("rho", 0.0)) > 0.0

    max_train_batches = 10 if args.dry_run else None
    max_val_batches   = 10 if args.dry_run else None

    log_path  = Path(args.log_dir)  / "metrics.csv"
    ckpt_dir  = Path(args.checkpoint_dir)

    # ── training loop ──
    for epoch in range(start_epoch, n_epochs):
        print(f"\n{'='*60}", flush=True)
        print(f"Epoch {epoch+1}/{n_epochs}", flush=True)

        train_loss  = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_sam=use_sam, max_batches=max_train_batches
        )
        val_metrics = validate(model, test_loader, device, max_batches=max_val_batches)

        Se    = val_metrics.get("sensitivity", 0.0)
        Sp    = val_metrics.get("specificity", 0.0)
        Score = val_metrics.get("score",       0.0)

        print(f"\nEpoch {epoch+1} summary:", flush=True)
        print(f"  train_loss : {train_loss:.4f}", flush=True)
        print(f"  Se         : {Se:.4f}",         flush=True)
        print(f"  Sp         : {Sp:.4f}",         flush=True)
        print(f"  Score      : {Score:.4f}",      flush=True)

        # ── log to CSV ──
        log_metrics_csv(log_path, {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_Se":     round(Se,    4),
            "val_Sp":     round(Sp,    4),
            "val_Score":  round(Score, 4),
        })

        # ── checkpoint on best Se ──
        if Se > best_Se:
            best_Se = Se
            save_checkpoint({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "best_Se":     best_Se,
                "config":      config,
            }, ckpt_dir / "best_model.pt")

        # ── always save latest ──
        save_checkpoint({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "best_Se":     best_Se,
            "config":      config,
        }, ckpt_dir / "latest.pt")

    print("\n✅ Training complete.", flush=True)
    print(f"Best Se: {best_Se:.4f}", flush=True)


if __name__ == "__main__":
    main()