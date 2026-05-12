"""Preprocessing utilities for the ICBHI 2017 dataset.

Two pipelines live side-by-side:

1) ``preprocess_cycle`` returns a NumPy log-mel spectrogram (legacy path,
   used by the old ICBHIDataset).
2) ``preprocess_cycle_waveform`` returns a 1-D NumPy waveform at the
   correct sample rate, ready to be passed to the AST feature extractor.
   This is the path we now use for training because AST positional
   embeddings only make sense if the spectrogram is built with the
   AudioSet-native settings (16 kHz, 128 mels, 1024 frames).

Key fixes vs. the previous version
----------------------------------
* Sample rate **forced to 16 kHz** by default (AST native).
* Cyclic padding to exactly ``target_duration * sample_rate`` samples.
* Helper ``parse_icbhi_annotations`` is unchanged but stricter on parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

def parse_icbhi_annotations(wav_path: str, txt_path: str) -> list[dict[str, float | int]]:
    """Parse a single ICBHI annotation file into cycle dictionaries.

    Args:
        wav_path: Path to the parent recording (kept for API symmetry).
        txt_path: Path to the matching annotation file.

    Returns:
        A list of dictionaries with ``start``, ``end`` and ``label`` keys
        where ``label`` ∈ {0 normal, 1 crackle, 2 wheeze, 3 both}.
    """
    _ = wav_path
    annotation_path = Path(txt_path)
    cycles: list[dict[str, float | int]] = []

    with annotation_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 4:
                LOGGER.warning(
                    "Skipping malformed annotation line %s in %s: %s",
                    line_number, annotation_path, line,
                )
                continue

            try:
                start, end, crackle, wheeze = map(float, parts)
            except ValueError:
                LOGGER.warning(
                    "Skipping malformed annotation line %s in %s: %s",
                    line_number, annotation_path, line,
                )
                continue

            if crackle == 0 and wheeze == 0:
                label = 0
            elif crackle == 1 and wheeze == 0:
                label = 1
            elif crackle == 0 and wheeze == 1:
                label = 2
            else:
                label = 3

            cycles.append({"start": start, "end": end, "label": label})

    return cycles


# ---------------------------------------------------------------------------
# Waveform extraction (the path we now use for training)
# ---------------------------------------------------------------------------

def _load_and_cycle_pad(
    wav_path: str,
    start: float,
    end: float,
    sample_rate: int,
    target_duration: float,
) -> np.ndarray:
    """Load a cycle from ``wav_path`` between ``start`` and ``end`` seconds
    and cyclic-pad / truncate it to exactly ``target_duration`` seconds."""

    audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    start_index = int(start * sample_rate)
    end_index = int(end * sample_rate)
    cycle = audio[start_index:end_index]

    if cycle.size == 0:
        raise ValueError(
            f"Empty cycle extracted from {wav_path} between {start} and {end}"
        )

    target_length = int(target_duration * sample_rate)
    if cycle.size < target_length:
        repeats = (target_length // cycle.size) + 1
        cycle = np.tile(cycle, repeats)[:target_length]
    else:
        cycle = cycle[:target_length]

    return cycle.astype(np.float32, copy=False)


def preprocess_cycle_waveform(
    wav_path: str,
    start: float,
    end: float,
    config: dict,
) -> np.ndarray:
    """Return a fixed-length 1-D waveform ready for the AST feature extractor.

    Always uses ``config['sample_rate']`` (default **16 kHz**, AST native)
    and ``config['target_duration']`` seconds.
    """
    sample_rate = int(config.get("sample_rate", 16000))
    target_duration = float(config.get("target_duration", 8))
    return _load_and_cycle_pad(wav_path, start, end, sample_rate, target_duration)


# ---------------------------------------------------------------------------
# Spectrogram extraction (legacy path; kept for the FallbackCNN dry-run)
# ---------------------------------------------------------------------------

def preprocess_cycle(wav_path: str, start: float, end: float, config: dict):
    """Load a respiratory cycle and convert it to a normalized log-mel spectrogram.

    Note
    ----
    This is the **legacy** function. It is still used by the FallbackCNN
    code path. For AST training prefer ``preprocess_cycle_waveform`` and
    let the AST feature extractor build the spectrogram with AudioSet
    statistics.
    """
    sample_rate = int(config.get("sample_rate", 16000))
    target_duration = float(config.get("target_duration", 8))
    n_mels = int(config.get("n_mels", 128))
    n_fft = int(config.get("n_fft", 1024))
    hop_length = int(config.get("hop_length", 512))
    fmax = config.get("fmax", 8000)

    cycle = _load_and_cycle_pad(wav_path, start, end, sample_rate, target_duration)

    mel_spec = librosa.feature.melspectrogram(
        y=cycle,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)


def augment_spectrogram(spec, config: dict):
    """Apply simple SpecAugment-style masks and additive noise to a log-mel
    spectrogram (legacy helper, used by the FallbackCNN path)."""

    spec = spec.copy()
    n_mels, T = spec.shape

    time_mask_pct = float(config.get("time_mask_pct", 0.2))
    freq_mask_pct = float(config.get("freq_mask_pct", 0.2))
    time_masks = int(config.get("time_masks", 1))
    freq_masks = int(config.get("freq_masks", 1))
    noise_factor = float(config.get("noise_factor", 0.005))

    for _ in range(time_masks):
        t = int(T * time_mask_pct)
        if t <= 0:
            continue
        t0 = np.random.randint(0, max(1, T - t + 1))
        spec[:, t0 : t0 + t] = spec[:, t0 : t0 + t].mean()

    for _ in range(freq_masks):
        f = int(n_mels * freq_mask_pct)
        if f <= 0:
            continue
        f0 = np.random.randint(0, max(1, n_mels - f + 1))
        spec[f0 : f0 + f, :] = spec[f0 : f0 + f, :].mean()

    if noise_factor > 0:
        spec += np.random.normal(scale=noise_factor, size=spec.shape)

    return spec
