# Handoff for Claude: respiratory-classification (May 12, 2026)

## Why I need your help
I need you to command me with an exact, minimal-risk plan to finish this project quickly and correctly.
Please assume I will execute your steps one-by-one and report back outputs.

The main blocker is that baseline performance is far below target, and evidence suggests we evaluated the wrong model type (fallback CNN checkpoint) instead of the intended AST baseline.

## Current status summary
- Local environment and imports are working.
- Data pipeline is implemented and runnable.
- Dry-run training path works.
- I can run inference and evaluation scripts.
- A baseline evaluation was run on `checkpoints/best_model.pt`, but results are poor and likely invalid for AST baseline claims.

## What is implemented right now
- `scripts/verify_dataset.py` passes.
- `scripts/prepare_data.py` passes and generates patient-level splits.
- `src/preprocessing.py` has parser + cycle preprocessing + augmentation.
- `src/dataset.py` provides dataset and sampler helper.
- `src/train.py` runs dry-run and supports resume/checkpoints.
- `src/evaluate.py` now implements ICBHI binary metrics `Se`, `Sp`, `Score`.
- `scripts/evaluate_quick.py` was added to evaluate and generate confusion matrix quickly on CPU.
- `notebooks/03_Results_Analysis.ipynb` now has cells for evaluation/plots, but none executed yet in this workspace session.

## Most important observed issue
From checkpoint inspection and loading behavior:
- `checkpoints/best_model.pt` contains keys: `epoch`, `model_state`, `best_Se`, `config`.
- `epoch` is `0` in this checkpoint.
- The stored state_dict matches fallback CNN layers (`conv.*`, `fc.*`) when loaded, not AST backbone/head keys.

This strongly suggests the downloaded "best" checkpoint is from dry-run or fallback model path, not from a completed pretrained AST baseline run.

## Measured performance so far
Using current `checkpoints/best_model.pt` with `scripts/evaluate_quick.py`:
- Sensitivity (Se): 29.44%
- Specificity (Sp): 65.42%
- ICBHI Score: 47.43%

Per-class recall from confusion matrix output:
- Normal: 65.42%
- Crackle: 0.16%
- Wheeze: 30.77%
- Both: 0.00%

Saved artifacts:
- `logs/baseline_results.txt`
- `logs/confusion_matrix_baseline.png`

## Task tracker reality check
`tasks.md` still has many items unchecked. In practice, partial implementation exists for several of them.
I need your directive on whether to:
1. Strictly re-verify and mark tasks sequentially from 10 onward, or
2. Prioritize producing a valid AST+SAM baseline result first, then backfill checkmarks.

## What I want you to command me to do
Please give me a strict step-by-step plan to recover and finish, with command-level detail.

I need you to cover all of this:
1. How to guarantee Colab training uses `use_pretrained: true` and actually saves AST checkpoints (not fallback).
2. How to verify checkpoint architecture immediately after each save.
3. Which exact metrics/plots/files to produce to satisfy Tasks 19, 20, 21.
4. Minimum acceptable baseline gate before moving to improvement experiments.
5. Fastest sequence of experiments (Task 22/23) most likely to increase Se above 68.31%.

## Constraints and preferences
- Colab free tier (T4, unstable runtime).
- Need checkpoint persistence on Drive.
- Prefer robust, low-complexity changes first.
- Need commands and config edits that are copy-paste ready.

## Project targets (must beat)
- Reference Se: 68.31%
- Reference Sp: 67.89%
- Reference Score: 68.10%

## Files you should assume are central
- `tasks.md`
- `configs/baseline.yaml`
- `src/model.py`
- `src/train.py`
- `src/dataset.py`
- `src/preprocessing.py`
- `src/evaluate.py`
- `scripts/evaluate_quick.py`
- `notebooks/colab_quickstart.ipynb`
- `notebooks/03_Results_Analysis.ipynb`

## Exact response format requested from you
Please respond with:
1. A numbered recovery plan (phases + success criteria per phase).
2. Exact config diff(s) for `configs/baseline.yaml`.
3. Exact Colab commands/cells in execution order.
4. A checkpoint-validation snippet that confirms AST vs fallback model.
5. A "stop/go" decision table based on Se progression by epoch.
6. The first 2 improvement experiments you recommend after a valid baseline.
