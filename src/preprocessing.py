"""Preprocessing utilities for ICBHI 2017.

Implemented in Tasks 6, 8, and 10 per tasks.md.
"""


def parse_icbhi_annotations(wav_path: str, txt_path: str):
    raise NotImplementedError("Implement in Task 6.")


def preprocess_cycle(wav_path: str, start: float, end: float, config: dict):
    raise NotImplementedError("Implement in Task 8.")


def augment_spectrogram(spec, config: dict):
    raise NotImplementedError("Implement in Task 10.")
