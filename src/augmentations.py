"""
Audio and spectrogram augmentations.

We implement three families of augmentation:

1. Waveform-level (cheap, applied during data loading):
    * gaussian_noise: random additive noise SNR ~ U[10, 30] dB
    * time_shift: random circular shift up to ±20% of the clip
    * gain: random gain in [0.8, 1.2]

2. Spectrogram-level (SpecAugment - Park et al. 2019):
    * frequency_masking: zero out a random band of mel bins
    * time_masking: zero out a random time interval
    These are applied AFTER the AST feature extractor since AST consumes
    a (1, 1024, 128) log-mel spectrogram.

3. Sample-level (Mixup - Zhang et al. 2018):
    * mixup_data: convex combination of two samples and their one-hot labels
    Improvement over the reference paper, which only uses weighted sampling
    and label smoothing.

All callable classes are torch-friendly so they can run on GPU within the
DataLoader collate or inside the model forward pass.
"""

from typing import Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------
# 1. Waveform-level augmentations (numpy)
# ----------------------------------------------------------------------
class WaveformAugment:
    """Small waveform-level augmentation pipeline used during training only."""

    def __init__(
        self,
        noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (10.0, 30.0),
        shift_prob: float = 0.5,
        shift_max_frac: float = 0.2,
        gain_prob: float = 0.5,
        gain_range: Tuple[float, float] = (0.8, 1.2),
        seed: int | None = None,
    ) -> None:
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.shift_prob = shift_prob
        self.shift_max_frac = shift_max_frac
        self.gain_prob = gain_prob
        self.gain_range = gain_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        wav = wav.astype(np.float32, copy=True)

        # Random circular time shift.
        if self.rng.random() < self.shift_prob:
            max_shift = int(self.shift_max_frac * len(wav))
            if max_shift > 0:
                shift = int(self.rng.integers(-max_shift, max_shift + 1))
                wav = np.roll(wav, shift)

        # Random gain.
        if self.rng.random() < self.gain_prob:
            gain = float(self.rng.uniform(*self.gain_range))
            wav = wav * gain

        # Additive Gaussian noise with controlled SNR.
        if self.rng.random() < self.noise_prob:
            snr_db = float(self.rng.uniform(*self.noise_snr_range))
            sig_power = float(np.mean(wav ** 2)) + 1e-12
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = self.rng.normal(0.0, np.sqrt(noise_power), size=wav.shape)
            wav = wav + noise.astype(np.float32)

        # Re-normalise to avoid clipping on cyclic-padded signals.
        peak = float(np.max(np.abs(wav))) + 1e-9
        if peak > 1.0:
            wav = wav / peak
        return wav.astype(np.float32)


# ----------------------------------------------------------------------
# 2. SpecAugment (torch)
# ----------------------------------------------------------------------
class SpecAugment:
    """SpecAugment for AST-style log-mel spectrograms.

    The AST feature extractor returns tensors of shape (B, T=1024, F=128).
    We mask up to `num_freq_masks` frequency bands of width <=`freq_mask_param`
    and up to `num_time_masks` time intervals of width <=`time_mask_param`.

    Args:
        freq_mask_param: maximum width F (# mel bins) of each freq mask.
        time_mask_param: maximum width T (# frames) of each time mask.
        num_freq_masks:  number of independent frequency masks per sample.
        num_time_masks:  number of independent time masks per sample.
        prob:            probability of applying the augmentation at all.
        replace_with_mean: if True, masked region is filled with the spectrogram
                           mean rather than 0 (more stable with normalised AST input).
    """

    def __init__(
        self,
        freq_mask_param: int = 24,
        time_mask_param: int = 96,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        prob: float = 0.8,
        replace_with_mean: bool = True,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.prob = prob
        self.replace_with_mean = replace_with_mean

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to a batch of spectrograms.

        Args:
            spec: tensor of shape (B, T, F) or (B, 1, T, F).
        """
        if spec.dim() == 4:
            squeezed = True
            spec = spec.squeeze(1)
        else:
            squeezed = False

        B, T, F = spec.shape
        out = spec.clone()
        fill_value = (
            spec.mean(dim=(1, 2), keepdim=True) if self.replace_with_mean else 0.0
        )

        for b in range(B):
            if torch.rand(1).item() > self.prob:
                continue
            fv = fill_value[b] if isinstance(fill_value, torch.Tensor) else fill_value

            # Frequency masks
            for _ in range(self.num_freq_masks):
                f = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
                if f == 0:
                    continue
                f0 = int(torch.randint(0, max(F - f, 1), (1,)).item())
                out[b, :, f0 : f0 + f] = fv

            # Time masks
            for _ in range(self.num_time_masks):
                t = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
                if t == 0:
                    continue
                t0 = int(torch.randint(0, max(T - t, 1), (1,)).item())
                out[b, t0 : t0 + t, :] = fv

        if squeezed:
            out = out.unsqueeze(1)
        return out


# ----------------------------------------------------------------------
# 3. Mixup
# ----------------------------------------------------------------------
def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation (Zhang et al., 2018).

    Returns:
        mixed_x: convex combination x' = lam*x + (1-lam)*x[perm]
        y_a, y_b: original and permuted labels
        lam: mixing coefficient sampled from Beta(alpha, alpha)
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
