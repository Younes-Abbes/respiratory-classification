"""
PyTorch Dataset for the preprocessed ICBHI 2017 .npz file.

The preprocessing script `preprocess.py` produces a single .npz file
containing fixed-length 8-second waveforms at 16 kHz, plus integer labels
and device IDs. This module wraps that into a torch Dataset and applies
- waveform-level augmentations during training
- the AST feature extractor (log-mel spectrogram, 1024 x 128) on the fly

The reason for doing the spectrogram conversion at __getitem__ time rather
than caching it in advance is twofold:
    * it lets us apply waveform-level random augmentation each epoch,
    * the AST feature extractor's normalisation depends on AudioSet
      statistics which we should not bypass.
"""

from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import WaveformAugment


class ASTDataset(Dataset):
    """ICBHI 2017 dataset wrapping pre-padded 8-second 16 kHz waveforms."""

    def __init__(
        self,
        wavs: np.ndarray,            # (N, T)
        labels: np.ndarray,          # (N,)
        devices: np.ndarray,         # (N,)
        feature_extractor,           # transformers.ASTFeatureExtractor
        train: bool = False,
        wave_aug: Optional[WaveformAugment] = None,
        sample_rate: int = 16000,
    ) -> None:
        if not (len(wavs) == len(labels) == len(devices)):
            raise ValueError("wavs, labels and devices must have the same length")

        self.wavs = wavs
        self.labels = labels.astype(np.int64)
        self.devices = devices.astype(np.int64)
        self.feature_extractor = feature_extractor
        self.train = train
        self.sample_rate = sample_rate
        self.wave_aug = wave_aug if (train and wave_aug is not None) else None

    def __len__(self) -> int:
        return len(self.wavs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wav = self.wavs[idx]
        label = int(self.labels[idx])
        device_id = int(self.devices[idx])

        if self.wave_aug is not None:
            wav = self.wave_aug(wav)

        # AST feature extractor expects a list of 1-D numpy arrays. It returns
        # a (1, 1024, 128) log-mel spectrogram normalised with AudioSet stats.
        encoded = self.feature_extractor(
            wav,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        # Shape: (1, 1024, 128). Squeeze batch dim, keep (T, F).
        spec = encoded["input_values"].squeeze(0)

        return spec, torch.tensor(label, dtype=torch.long), torch.tensor(device_id)


def make_weighted_sampler(labels: np.ndarray) -> torch.utils.data.WeightedRandomSampler:
    """Inverse-frequency WeightedRandomSampler.

    Returns a sampler such that, in expectation, each class is sampled with
    equal probability per batch. This is the same strategy used in the
    reference paper.
    """
    counts = np.bincount(labels)
    class_w = 1.0 / np.maximum(counts, 1)
    sample_w = class_w[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
