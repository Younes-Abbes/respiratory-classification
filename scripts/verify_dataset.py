"""Verify ICBHI dataset presence and annotation completeness."""

import glob
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import parse_icbhi_annotations


def main() -> None:
    raw_dir = "data/raw"
    wav_files = glob.glob(os.path.join(raw_dir, "*.wav"))
    txt_files = glob.glob(os.path.join(raw_dir, "*.txt"))

    print(f"WAV files found: {len(wav_files)}")
    print(f"TXT files found: {len(txt_files)}")

    wav_stems = {os.path.splitext(os.path.basename(f))[0] for f in wav_files}
    txt_stems = {
        os.path.splitext(os.path.basename(f))[0]
        for f in txt_files
        if "diagnosis" not in f
    }
    missing = wav_stems - txt_stems
    print(f"WAV files missing annotation: {missing}")

    assert len(wav_files) == 920, f"Expected 920 wav files, got {len(wav_files)}"
    assert len(missing) == 0, "Some WAV files have no annotation"
    print("✅ Dataset verified.")

    first_wav = sorted(wav_files)[0]
    first_txt = first_wav.replace(".wav", ".txt")
    cycles = parse_icbhi_annotations(first_wav, first_txt)
    assert all("label" in cycle for cycle in cycles)
    assert all(cycle["label"] in [0, 1, 2, 3] for cycle in cycles)
    assert all(cycle["end"] > cycle["start"] for cycle in cycles)

    all_cycles = []
    for wav_path in wav_files:
        txt_path = wav_path.replace(".wav", ".txt")
        if os.path.exists(txt_path):
            all_cycles.extend(parse_icbhi_annotations(wav_path, txt_path))

    counts = {label: sum(1 for cycle in all_cycles if cycle["label"] == label) for label in range(4)}
    print(counts)
    assert counts[0] == 3642, f"Normal: expected 3642, got {counts[0]}"
    assert counts[1] == 1864, f"Crackle: expected 1864, got {counts[1]}"
    assert counts[2] == 886, f"Wheeze: expected 886, got {counts[2]}"
    assert counts[3] == 506, f"Both: expected 506, got {counts[3]}"
    print("✅ Annotation parser verified.")


if __name__ == "__main__":
    main()
