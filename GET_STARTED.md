# GET STARTED — Respiratory Sound Classification (ICBHI 2017)

Step-by-step runbook for the AST + SAM recovery training.  
**Target:** beat the reference paper (Se ≥ 68.31%, Sp ≥ 67.89%, Score ≥ 68.10%).

> Read this once, then follow the checklist on every Colab session.

---

## 0. One-time setup (do this only the very first time)

### 0.1 Google Drive — create the project folder

On your Google Drive, create this folder structure:

```
MyDrive/
└── respiratory_project/
    ├── ICBHI_final_database.zip      ← upload the full ICBHI dataset zip here
    ├── checkpoints/                  ← created automatically
    └── logs/                         ← created automatically
```

You only need to upload `ICBHI_final_database.zip` once. Everything else is created by the notebook.

### 0.2 Push the latest code to GitHub (from your local machine)

```bash
cd /Users/abderrahmenbejaoui/respiratory-classification
git add -A
git commit -m "fix: AST + SAM recovery training pipeline"
git push origin main
```

Verify the push went through at https://github.com/Younes-Abbes/respiratory-classification

---

## 1. Run on Google Colab — every session

### 1.1 Open the notebook and pick a GPU

1. Open https://colab.research.google.com  
2. `File → Open notebook → GitHub` and paste the repo URL.  
3. Open `notebooks/Deep_Learning_colab.ipynb`.  
4. `Runtime → Change runtime type → GPU → T4` → **Save**.

### 1.2 Run the cells in order

> **Do NOT skip the dry-run cell.** It catches 95% of errors in under a minute.

| Cell | Markdown title | What it does | Expected output |
|------|---------------|--------------|-----------------|
| 1 | `## 1. Mount Drive & locate the dataset` | Mounts Drive, unzips ICBHI. | `WAV files found: 920` |
| 2 | `## 2. Clone the GitHub repo` | Clones / pulls latest code. | A git commit hash. |
| 3 | `## 3. Install dependencies & verify GPU` (2 cells) | `pip install` then GPU check. | `cuda avail. : True` and `device : Tesla T4`. |
| 4 | `## 4. Wire the dataset into the repo & build splits` (2 cells) | Symlinks raw files, builds CSV splits. | `WAV files in data/raw : 920` and class distribution table. |
| 5 | `## 5. Sanity check — AST forward pass` | Loads AST, runs one forward pass. | `logits shape: (1, 4)`. **No warnings about positional-embedding mismatch.** |
| 6 | `## 6. Dry run (2 epochs × 5 batches)` | Tests training loop end-to-end. | Completes in 30–90 s with batch progress logs. |
| 7 | `## 7. Clear old checkpoints` | Deletes broken checkpoints from previous failed run. | `Clean. Ready to launch full training.` |
| 8 | `## 8. Full training run` | Real run, 40 epochs. | Per-epoch logs printed live for ~25–35 min. |
| 9 | `## 9. Training curves` | Plots train/val Score, Se, Sp. | A 3-panel plot. |
| 10 | `## 10. Final evaluation on the official test split` (2 cells) | Runs `evaluate_quick.py`, displays confusion matrix. | `Sensitivity ≥ 68%`, confusion matrix image. |
| 11 | `## 11. Stop / Go gates by epoch` | Reference table (no code). | — |
| 12 | `## 12. Backup everything to Drive` | Copies local files back to Drive. | List of `ckpt -> ...` and `log -> ...`. |

### 1.3 Decision points

After cell 5, before launching the full run, check:

- ✅ Trainable params printed > 80 M (AST is loaded).
- ✅ Spec shape is `(1024, 128)`.
- ✅ Logits shape is `(1, 4)`.
- ❌ If any check fails — **STOP** and check the error before wasting compute.

After cell 6 (dry run):

- ✅ Two epochs completed.
- ✅ A `latest.pt` and `best_model.pt` were created.
- ❌ Any traceback → fix before launching cell 8.

---

## 2. While training is running (cell 8)

### 2.1 Stop / Go gates

Watch the per-epoch lines in cell 8 output. The line format is:

```
[epoch  7] train_loss=0.85  train Se=0.74 Sp=0.62 Score=0.68  | val Se=0.66 Sp=0.65 Score=0.66  lr=8.43e-05
```

Use this table to decide if training is healthy:

| Epoch | Minimum **val Score** to keep going | If you fail it |
|------:|------------------------------------:|----------------|
| 5     | 0.55                                | Verify AST forward shape; check spec normalization. |
| 10    | 0.60                                | In `configs/baseline.yaml` set `mixup_alpha: 0.1` and `spec_time_mask: 64`. |
| 20    | 0.64                                | Set `loss: ce`, `label_smoothing: 0.1`, `sampler: none`. |
| 30    | 0.67                                | Set `class_balanced_beta: 0.9` and `head_lr: 0.0002`. |
| 40    | 0.68                                | **You beat the reference paper.** Move to improvement experiments. |

### 2.2 Degenerate collapse alarm

If you ever see **train Se ≈ 0.90 and train Sp ≈ 0.30** in cell 8 output, the model is predicting "abnormal" for everything (exactly what happened in your previous run). **Stop training** (Runtime → Interrupt) and apply these three knobs in `configs/baseline.yaml`:

```yaml
sampler: sqrt              # not "inverse"
class_balanced_beta: 0.99  # gentler reweighting
head_dropout: 0.15         # was 0.30
```

Commit + push, re-run cells 2 → 8.

### 2.3 If Colab disconnects mid-training

Don't panic — `latest.pt` is written to Drive every epoch.

1. Re-mount, re-clone (cells 1 → 4).
2. Skip the cleanup cell (7) — it would delete your progress.
3. Uncomment the **resume cell (8b)** and run it instead of cell 8:

```python
!python -u -m src.train --resume \
    --checkpoint-dir /content/drive/MyDrive/respiratory_project/checkpoints \
    --log-dir /content/drive/MyDrive/respiratory_project/logs
```

Training picks up at the next epoch.

---

## 3. After training finishes

### 3.1 Verify the artifacts

In `MyDrive/respiratory_project/` you should see:

```
checkpoints/
  ├── best_model.pt        ← the best checkpoint (selected on val Score)
  └── latest.pt            ← last-epoch snapshot
logs/
  ├── metrics.csv          ← per-epoch metrics (used for the training curves)
  ├── test_results.txt     ← final ICBHI Se/Sp/Score on the test set
  ├── baseline_results.txt ← human-readable version of the same numbers
  ├── confusion_matrix_baseline.png
  ├── training_curves.png
  └── test_predictions.npz ← y_true / y_pred arrays for analysis
```

### 3.2 Check the final numbers

Open `logs/test_results.txt` (or look at cell 10 output). You're looking for something like:

```
Sensitivity (Se):  69.42%
Specificity (Sp):  68.12%
ICBHI Score:       68.77%
```

If `Score > 68.10%`, you have beaten the reference paper.  
If `Score < 65%`, see the gates table above and tweak hyperparameters.

### 3.3 Update tasks.md

Mark these tasks as done in `tasks.md`:

- [x] 12 — AST model loading
- [x] 13 — SAM optimizer
- [x] 14 — Loss function + class weights
- [x] 15 — ICBHI evaluation metrics
- [x] 16 — Training loop (local dry run)
- [x] 17 — Checkpoint & resume system
- [x] 18 — Colab notebook setup
- [x] 19 — Baseline training run (Colab)
- [x] 20 — Baseline reproduces reference paper
- [x] 21 — Confusion matrix analysis

Push the updated `tasks.md` so your teammates see progress.

---

## 4. Local development workflow

You should not have to run Python locally except to edit files. But if you want to verify changes compile:

```bash
cd /Users/abderrahmenbejaoui/respiratory-classification
python3 -m py_compile src/*.py scripts/*.py
```

If you want to verify the config loads:

```bash
python3 -c "import yaml; print(yaml.safe_load(open('configs/baseline.yaml')))"
```

After any local edit, commit + push, then re-clone on Colab (or `git pull` in the cloned dir).

---

## 5. Improvement experiments (after a valid baseline)

Once you have `Score > 68%`, run two improvement experiments to beat your own baseline:

### Experiment A — Patch-Mix augmentation
Replace Mixup with Patch-Mix (Bae et al., "Patch-Mix Contrastive Learning..." Interspeech 2023). Mixes spectrogram patches between samples rather than whole spectrograms. Edit `src/augmentations.py` to add a `patch_mix_data` function and swap it in `src/train.py`. Expected: +1–2 % Score.

### Experiment B — Ensemble (AST + EfficientNet-B0)
Train an `EfficientNet-B0` on the same mel spectrograms (you can build a small `src/cnn_model.py`), then average the softmax probabilities of the two models on test. Expected: +1 % Score and much more robust Sp.

Track every experiment in a results table in your report:

| Run | Loss | Sampler | Mixup | SpecAug | Score (val) | Score (test) |
|-----|------|---------|-------|---------|-------------|--------------|
| baseline AST+SAM | focal | sqrt | 0.2 | yes | ... | ... |
| +patch-mix | focal | sqrt | patch-mix | yes | ... | ... |
| ensemble (AST + EffNet) | focal | sqrt | 0.2 | yes | ... | ... |

---

## 6. FAQ / troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'transformers'` | Re-run cell 3 (`pip install -r requirements_colab.txt`). |
| `WAV files found: 0` | The zip on Drive is missing or has a different name. Check `MyDrive/respiratory_project/ICBHI_final_database.zip`. |
| `CUDA out of memory` | In `configs/baseline.yaml`, drop `batch_size: 8` to `4`. Commit, push, re-clone, re-run. |
| Warnings about positional-embedding reshape | The wrong `max_length` is set somewhere. Verify `ast_max_length: 1024` and that `src/model.py` does NOT pass `max_length=` to `ASTConfig`. |
| Training is healthy but val Score plateaus at 0.62 | The model is underfitting — set `freeze_backbone: false`, `warmup_epochs: 1`, `epochs: 50`. |
| Test set numbers much worse than val | Patient overlap somewhere — re-run `scripts/prepare_data.py` and check `train.csv` vs `test.csv` for shared `patient_id`. |
| `data/splits/train.csv: file not found` | You forgot to run cell 4 (the "build splits" cell). |

---

## 7. Hand-off checklist (before submitting the project)

- [ ] `logs/test_results.txt` shows Score ≥ 68.10 %.
- [ ] `logs/confusion_matrix_baseline.png` is saved.
- [ ] `logs/training_curves.png` is saved.
- [ ] All checkpoints zipped and uploaded.
- [ ] `tasks.md` updated with completion ticks.
- [ ] Final report includes:
  - The recovery plan (this file).
  - The training curves.
  - The confusion matrix.
  - A comparison table with the reference paper.
  - Ablation table (baseline vs. improvements).
- [ ] Code zip runs `python3 -m py_compile src/*.py scripts/*.py` cleanly on a fresh checkout.
