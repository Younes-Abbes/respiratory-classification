"""Training loop implementation (Tasks 14-17).

This script provides a dry-run mode suitable for local verification without
running a full multi-hour training. It supports SAM wrapping and FocalLoss.
"""

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


def get_optimizer(model: torch.nn.Module, config: dict):
    base_lr = float(config.get("lr", 1e-5))
    weight_decay = float(config.get("weight_decay", 1e-4))
    rho = float(config.get("rho", 0.0))

    base_opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    if rho > 0.0:
        # Wrap in SAM
        opt = SAM(model.parameters(), torch.optim.AdamW, rho=rho, lr=base_lr, weight_decay=weight_decay)
    else:
        opt = base_opt
    return opt


def train_one_epoch(model, loader, optimizer, criterion, device, use_sam=False, max_batches=None):
    model.train()
    total_loss = 0.0
    seen = 0
    for i, (specs, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        specs = specs.to(device)
        labels = labels.to(device)

        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()

        if use_sam:
            optimizer.first_step(zero_grad=True)
            # second forward/backward
            logits2 = model(specs)
            loss2 = criterion(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += float(loss.detach().cpu().item())
        seen += labels.size(0)

    return total_loss / max(1, (i + 1))


def validate(model, loader, device, max_batches=None):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (specs, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            specs = specs.to(device)
            labels = labels.to(device)
            logits = model(specs)
            preds = logits.argmax(dim=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    import numpy as np

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return icbhi_metrics(y_true, y_pred)


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--epochs", type=int, default=2)

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs"
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    import yaml

    with open(Path("configs") / "baseline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_ds = ICBHIDataset(str(Path("data") / "splits" / "train.csv"), config, augment=False)
    test_ds = ICBHIDataset(str(Path("data") / "splits" / "test.csv"), config, augment=False)

    # Weighted sampler for training
    try:
        import numpy as np

        labels = np.array(train_ds.df["label"].tolist(), dtype=int)
        sampler = make_weighted_sampler(labels)
        train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), sampler=sampler, num_workers=0)
    except Exception:
        train_loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 8)), shuffle=True, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=int(config.get("batch_size", 8)), shuffle=False, num_workers=0)

    use_pretrained = bool(config.get("use_pretrained", False))
    freeze_backbone = bool(config.get("freeze_backbone", False))
    model = load_model(use_pretrained=use_pretrained)
    # If the loaded model is CustomAST and user requested freezing, try to freeze
    try:
        if hasattr(model, "backbone") and freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False
    except Exception:
        pass
    model.to(device)

    # Resume if requested
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            print(f"Resumed model from {ckpt_path}")
        else:
            print(f"Resume checkpoint not found: {ckpt_path}")

    # Loss
    if config.get("loss", "ce") == "focal":
        counts = train_ds.df["label"].value_counts().reindex(range(4), fill_value=0).tolist()
        alpha = class_balanced_alpha(counts)
        criterion = FocalLoss(alpha=alpha, gamma=float(config.get("focal_gamma", 2.0)), label_smoothing=float(config.get("label_smoothing", 0.0)))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, config)
    use_sam = float(config.get("rho", 0.0)) > 0.0

    # Dry-run limits
    max_train_batches = 10 if args.dry_run else None
    max_val_batches = 10 if args.dry_run else None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_sam=use_sam, max_batches=max_train_batches)
        val_metrics = validate(model, test_loader, device, max_batches=max_val_batches)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_metrics={val_metrics}")

    # Save a small checkpoint
    ckpt = {"model_state": model.state_dict(), "config": config}
    torch.save(ckpt, Path("checkpoints") / "dry_run_ckpt.pt")
    print("Dry run complete, checkpoint saved to checkpoints/dry_run_ckpt.pt")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
