"""Prepare patient-level train/test CSV splits for ICBHI 2017."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import parse_icbhi_annotations


LABEL_NAMES = {0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}


def build_cycle_table(raw_dir: Path) -> pd.DataFrame:
    records = []
    for wav_path in sorted(raw_dir.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        patient_id = wav_path.stem.split("_")[0]
        recording_file = wav_path.stem
        cycles = parse_icbhi_annotations(str(wav_path), str(txt_path))

        for cycle_idx, cycle in enumerate(cycles):
            records.append(
                {
                    "patient_id": patient_id,
                    "recording_file": recording_file,
                    "cycle_idx": cycle_idx,
                    "start": cycle["start"],
                    "end": cycle["end"],
                    "label": cycle["label"],
                    "label_name": LABEL_NAMES[cycle["label"]],
                }
            )

    return pd.DataFrame(records)


def stratify_patients(cycle_table: pd.DataFrame) -> pd.DataFrame:
    patient_rows = []
    for patient_id, patient_cycles in cycle_table.groupby("patient_id"):
        label_counts = Counter(patient_cycles["label"])
        dominant_label = max(label_counts.items(), key=lambda item: (item[1], item[0]))[0]
        patient_rows.append({"patient_id": patient_id, "stratum": dominant_label})
    return pd.DataFrame(patient_rows)


def split_patients(patient_table: pd.DataFrame, test_size: float = 0.4, random_state: int = 42):
    train_patients, test_patients = train_test_split(
        patient_table["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=patient_table["stratum"],
    )
    return set(train_patients), set(test_patients)


def main() -> None:
    raw_dir = REPO_ROOT / "data" / "raw"
    splits_dir = REPO_ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    cycle_table = build_cycle_table(raw_dir)
    if cycle_table.empty:
        raise RuntimeError("No annotated cycles found in data/raw.")

    patient_table = stratify_patients(cycle_table)
    train_patients, test_patients = split_patients(patient_table)

    overlap = train_patients & test_patients
    if overlap:
        raise RuntimeError(f"Patient overlap detected: {sorted(overlap)}")

    train_df = cycle_table[cycle_table["patient_id"].isin(train_patients)].copy()
    test_df = cycle_table[cycle_table["patient_id"].isin(test_patients)].copy()

    train_df = train_df.sort_values(["patient_id", "recording_file", "cycle_idx"])
    test_df = test_df.sort_values(["patient_id", "recording_file", "cycle_idx"])

    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train patients: {len(train_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print(f"Train cycles: {len(train_df)}")
    print(f"Test cycles: {len(test_df)}")
    print("Zero patient overlap confirmed.")


if __name__ == "__main__":
    main()
