# Handoff for Claude: respiratory-classification project state

## Current project state
We have completed the project up to the point where training is ready to be run, but the full baseline training has not been executed to completion yet.

What is already done and verified:
- Tasks 1 through 18 are completed in `tasks.md`.
- Task 19, `Baseline training run (Colab)`, is the current active step.
- `scripts/verify_dataset.py` passes.
- `scripts/prepare_data.py` passes and created patient-level `data/splits/train.csv` and `data/splits/test.csv`.
- `src/preprocessing.py` implements:
  - `parse_icbhi_annotations(...)`
  - `preprocess_cycle(...)`
  - `augment_spectrogram(...)`
- `src/dataset.py` implements `ICBHIDataset` and returns `(1, 128, T)` tensors.
- `src/model.py` has `load_model(...)` with a lightweight fallback CNN for local dry-runs.
- `src/train.py` supports:
  - dry-run training
  - SAM-style training flow
  - checkpoint saving
  - resume via `--resume`
- `notebooks/01_EDA.ipynb` is populated and executed successfully.
- `notebooks/02_Preprocessing_Debug.ipynb` is populated and executed successfully.
- `notebooks/colab_quickstart.ipynb` exists and is meant for Google Colab free-tier execution.

## Current config state
The active baseline config is `configs/baseline.yaml` and currently contains GPU-friendly defaults:
- `batch_size: 2`
- `amp: true`
- `use_pretrained: false`
- `freeze_backbone: true`
- `rho: 0.05`
- `loss: ce`
- `label_smoothing: 0.1`
- `scheduler: none`
- `use_specaugment: false`
- `use_mixup: false`
- `use_waveaug: false`

Important detail:
- With `use_pretrained: false`, the training script uses the lightweight fallback model for local dry-runs.
- If the goal is the full AST baseline, `use_pretrained` must be switched to `true` in `configs/baseline.yaml` before training on Colab.

## What has been validated locally
The following have been run successfully on the local machine:
- Dataset verification.
- Patient-level split generation.
- Preprocessing checks for one sample from each class.
- `ICBHIDataset` DataLoader smoke test.
- Model forward pass with a batch from the dataset.
- Training dry-run and resume dry-run.

## Where the project should go next
The next concrete goal is to run the baseline model on Google Colab free tier, starting from the current repo state.

Suggested execution plan for Colab:
1. Open `notebooks/colab_quickstart.ipynb` in Colab.
2. Set runtime to GPU.
3. Mount Google Drive.
4. Put the ICBHI dataset in Drive at `/MyDrive/ICBHI_2017/`.
5. Clone the repo inside Colab or upload the workspace.
6. Install dependencies from `requirements_colab.txt`.
7. Symlink `data/raw` to the Drive dataset folder.
8. Symlink `checkpoints/` to a Drive folder so checkpoints persist.
9. Run `scripts/verify_dataset.py`.
10. Run `scripts/prepare_data.py`.
11. Run a dry-run training first with `python -m src.train --dry-run`.
12. If that works and GPU memory allows it, switch `configs/baseline.yaml` to `use_pretrained: true` and keep `freeze_backbone: true` initially.
13. Run the full training command with `python -m src.train`.

## Recommended Colab settings for the free tier
Because the free tier often gives a smaller GPU and shorter session windows:
- Start with `batch_size: 1` or `2`.
- Keep `freeze_backbone: true` at first.
- Keep `amp: true`.
- Persist checkpoints to Google Drive.
- If the pretrained AST does not fit, fall back to head-only training first.

## Key files to continue from
- `tasks.md`
- `configs/baseline.yaml`
- `src/train.py`
- `src/model.py`
- `src/dataset.py`
- `src/preprocessing.py`
- `scripts/prepare_data.py`
- `scripts/verify_dataset.py`
- `notebooks/colab_quickstart.ipynb`

## Notes for the next agent
- Do not redo the completed local pipeline work.
- Focus on Colab execution and the baseline training path.
- If you need to modify the Colab notebook, keep it safe by default: dry-run first, then full training only after the dataset and GPU are confirmed.
- The local machine already passed the dry-run path, but the full model training should be done in Colab.
