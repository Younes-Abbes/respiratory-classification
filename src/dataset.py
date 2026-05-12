"""PyTorch datasets for ICBHI 2017.

This module exposes two datasets:

* ``ICBHIDataset``        — legacy log-mel spectrogram dataset (FallbackCNN path).
* ``ICBHIASTDataset``     — proper AST pipeline:
    1. Load waveform at 16 kHz.
    2. Apply waveform-level augmentation (train only).
    3. Run AST feature extractor → (1, 1024, 128) spectrogram with
       AudioSet mean/std normalisation.

We also expose ``make_balanced_sampler`` which now supports several
balancing strategies (inverse-frequency, square-root, or none).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .augmentations import WaveformAugment
from .preprocessing import preprocess_cycle, preprocess_cycle_waveform


# ---------------------------------------------------------------------------
# Sampler helpers
# ---------------------------------------------------------------------------

def make_balanced_sampler(
    labels: np.ndarray,
    strategy: str = "sqrt",
) -> WeightedRandomSampler | None:
    """Return a ``WeightedRandomSampler`` with the requested balancing
    strategy or ``None`` if ``strategy='none'``.

    * ``"inverse"`` — weights ∝ 1 / count   (very aggressive)
    * ``"sqrt"``    — weights ∝ 1 / sqrt(count)   (recommended)
    * ``"none"``    — return ``None`` (DataLoader will use ``shuffle=True``)
    """
    strategy = (strategy or "sqrt").lower()
    if strategy == "none":
        return None

    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    if strategy == "inverse":
        class_w = 1.0 / counts
    elif strategy == "sqrt":
        class_w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown sampler strategy: {strategy!r}")

    class_w = class_w / class_w.sum() * len(class_w)
    sample_w = class_w[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Backwards-compatible alias (inverse-frequency) used by older scripts."""
    sampler = make_balanced_sampler(labels, strategy="inverse")
    assert sampler is not None
    return sampler


# ---------------------------------------------------------------------------
# AST dataset (the one used for training)
# ---------------------------------------------------------------------------

class ICBHIASTDataset(Dataset):
    """ICBHI cycles → AST inputs.

    For each row of the CSV we:
      1. Load the waveform between ``start`` and ``end`` at 16 kHz.
      2. Cyclic-pad / truncate to 8 s.
      3. If ``augment=True``: apply waveform-level augmentation.
      4. Run the HuggingFace AST feature extractor to obtain a
         ``(1, 1024, 128)`` tensor normalised with AudioSet stats.

    Returns ``(spec, label)`` where ``spec`` has shape ``(1024, 128)``.
    """

    def __init__(
        self,
        csv_path: str,
        config: dict,
        feature_extractor,
        augment: bool = False,
        wave_aug: Optional[WaveformAugment] = None,
        raw_dir: str = "data/raw",
    ) -> None:
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.config = dict(config)
        # Force AST-native sample rate regardless of what's in the YAML.
        self.config["sample_rate"] = int(config.get("sample_rate", 16000))
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.raw_dir = Path(raw_dir)

        if augment:
            self.wave_aug = wave_aug or WaveformAugment()
        else:
            self.wave_aug = None

    def __len__(self) -> int:
        return len(self.df)

    def get_labels(self) -> np.ndarray:
        return self.df["label"].to_numpy(dtype=np.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        wav_path = str(self.raw_dir / f"{row['recording_file']}.wav")

        wav = preprocess_cycle_waveform(
            wav_path, float(row["start"]), float(row["end"]), self.config
        )

        if self.wave_aug is not None:
            wav = self.wave_aug(wav)

        encoded = self.feature_extractor(
            wav,
            sampling_rate=self.config["sample_rate"],
            return_tensors="pt",
        )
        spec = encoded["input_values"].squeeze(0)  # (1024, 128)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return spec, label


# ---------------------------------------------------------------------------
# Legacy spectrogram dataset (FallbackCNN path)
# ---------------------------------------------------------------------------

class ICBHIDataset(Dataset):
    """Legacy dataset that returns a ``(1, 128, T)`` log-mel spectrogram
    computed locally — used only by the FallbackCNN path for dry-runs."""

    def __init__(self, csv_path: str, config: dict, augment: bool = False) -> None:
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.config = config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def get_labels(self) -> np.ndarray:
        return self.df["label"].to_numpy(dtype=np.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        wav_path = str(Path("data") / "raw" / f"{row['recording_file']}.wav")
        spec = preprocess_cycle(
            wav_path, float(row["start"]), float(row["end"]), self.config
        )
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return spec_tensor, label


# ---------------------------------------------------------------------------
# Patient-level train/val split
# ---------------------------------------------------------------------------

def patient_level_train_val_split(
    csv_path: str,
    val_frac: float = 0.15,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a CSV (with a ``patient_id`` column) into train / val frames so
    that no patient appears in both. Stratified by the patient's dominant
    label."""

    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns:
        raise KeyError("CSV is missing 'patient_id' column")

    # Dominant label per patient — used for stratification.
    dominant = df.groupby("patient_id")["label"].agg(
        lambda s: s.value_counts().idxmax()
    )
    patients = dominant.index.to_numpy()
    strata = dominant.to_numpy()

    rng = np.random.default_rng(seed)
    val_patients: list = []
    for cls in np.unique(strata):
        candidates = patients[strata == cls]
        n_val = max(1, int(round(len(candidates) * val_frac)))
        chosen = rng.choice(candidates, size=n_val, replace=False)
        val_patients.extend(chosen.tolist())

    val_set = set(val_patients)
    val_df = df[df["patient_id"].isin(val_set)].copy()
    train_df = df[~df["patient_id"].isin(val_set)].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
