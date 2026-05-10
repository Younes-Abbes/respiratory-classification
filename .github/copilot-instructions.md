# 🫁 Project Skill: Respiratory Sound Classification (GL4-RT4)
> **GitHub Copilot Reference — Read this before generating any code for this project.**

---

## 🎯 Project Mission

Classify respiratory sounds into **4 classes**: `normal`, `crackle`, `wheeze`, `both`  
using the **ICBHI 2017** dataset.  
Primary goal: **beat the reference paper's sensitivity (recall = 68.31%)**.

Reference paper: *Geometry-Aware Optimization for Respiratory Sound Classification*  
Reference GitHub: https://github.com/Atakanisik/ICBHI-AST-SAM  
Core architecture: **Audio Spectrogram Transformer (AST) + SAM optimizer**

---

## 📁 Project Structure (Always Respect This)

```
respiratory-classification/
├── .github/
│   └── copilot-instructions.md       ← symlink or copy of this file
├── configs/
│   └── baseline.yaml                 ← all hyperparameters live here
├── data/
│   ├── raw/                          ← ICBHI .wav + .txt files (never modify)
│   ├── processed/                    ← segmented cycles as .npy or .wav
│   └── splits/
│       ├── train.csv
│       └── test.csv                  ← official 60/40 split
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_Debug.ipynb
│   └── 03_Results_Analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py                    ← ICBHIDataset class
│   ├── preprocessing.py              ← audio → spectrogram pipeline
│   ├── model.py                      ← AST + modifications
│   ├── optimizer.py                  ← SAM implementation
│   ├── train.py                      ← training loop
│   ├── evaluate.py                   ← ICBHI metrics: Se, Sp, Score
│   └── utils.py                      ← logging, seeding, checkpointing
├── scripts/
│   ├── prepare_data.py               ← run once to build data/processed/
│   └── run_experiment.py             ← entry point for Colab
├── requirements.txt
├── requirements_colab.txt            ← pip installs safe for Colab
└── README.md
```

---

## 🗂️ Dataset: ICBHI 2017

| Property | Value |
|---|---|
| Patients | 126 |
| Audio files | 920 `.wav` recordings |
| Respiratory cycles | 6,898 annotated |
| Duration range | 10s – 90s per recording |
| Devices | AKGC417L, LittC2SE, Litt3200, Meditron |
| Official split | **60% train / 40% test** (patient-level, not sample-level) |

**Class distribution (severely imbalanced):**
```
normal  : 3642  ← majority (52.8%)
crackle : 1864  ← 27.0%
wheeze  :  886  ← 12.8%
both    :  506  ← 7.3%  ← hardest to detect
```

**Annotation file format** (`.txt` per recording):
```
<start_time> <end_time> <crackle_flag> <wheeze_flag>
# crackle=0, wheeze=0 → normal
# crackle=1, wheeze=0 → crackle
# crackle=0, wheeze=1 → wheeze
# crackle=1, wheeze=1 → both
```

**Label mapping (always use this):**
```python
LABEL_MAP = {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3}
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
```

---

## ⚙️ Preprocessing Pipeline (Critical)

### Step 1 — Cycle Segmentation
Parse `.txt` annotation files. Extract each cycle as a numpy array using the timestamps.
```python
import librosa
audio, sr = librosa.load(wav_path, sr=22050, mono=True)
cycle = audio[int(start * sr): int(end * sr)]
```

### Step 2 — Fixed-Length Padding (use CYCLIC, not zero-padding)
Target duration: **8 seconds** (as per reference paper).  
**Never use zero-padding** — it introduces silence that confuses the model.
```python
TARGET_LEN = 8 * SAMPLE_RATE
if len(cycle) < TARGET_LEN:
    # cyclic repeat
    repeats = (TARGET_LEN // len(cycle)) + 1
    cycle = np.tile(cycle, repeats)[:TARGET_LEN]
else:
    cycle = cycle[:TARGET_LEN]
```

### Step 3 — Log-Mel Spectrogram
```python
import librosa
mel = librosa.feature.melspectrogram(
    y=cycle, sr=22050,
    n_mels=128, n_fft=1024,
    hop_length=512, fmax=8000
)
log_mel = librosa.power_to_db(mel, ref=np.max)
# Shape expected by AST: (128, T) → normalize to [0, 1]
log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
```

### Step 4 — Augmentation (training only)
```python
# Time masking (mask up to 20% of time frames)
# Frequency masking (mask up to 20% of mel bands)
# Add Gaussian noise: noise_factor=0.005
# Time stretch: rate in [0.8, 1.2]
# Pitch shift: n_steps in [-2, 2]
```

---

## 🧠 Model Architecture

### Primary: AST (Audio Spectrogram Transformer)
- Pre-trained on **AudioSet** (mandatory — do not train from scratch)
- Library: `transformers` from HuggingFace or `timm`
- Checkpoint: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Input: log-Mel spectrogram treated as image patches (16×16)
- Fine-tune only the last N layers initially, then unfreeze all

```python
from transformers import ASTForAudioClassification, ASTConfig

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=4,
    ignore_mismatched_sizes=True
)
```

### Alternative Architectures (if experimenting)
- Swin Transformer + spectrogram
- EfficientNet-B0 on mel spectrogram (lighter baseline)
- CNN + BiLSTM hybrid (CRNN)

---

## 🔧 Optimizer: SAM (Sharpness-Aware Minimization)

**This is the core innovation of the reference paper. Always use SAM.**

SAM performs a two-step update per batch:
1. Compute gradient → find perturbation direction ε
2. Apply perturbation → compute gradient at (w + ε) → update weights

```python
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.rho = rho
        super().__init__(params, dict(rho=rho))

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # restore weights
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        norms = [p.grad.norm(p=2) for group in self.param_groups
                 for p in group["params"] if p.grad is not None]
        return torch.stack(norms).norm(p=2)
```

**Training loop with SAM (always follow this pattern):**
```python
for inputs, labels in dataloader:
    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.first_step(zero_grad=True)   # step 1: perturb

    criterion(model(inputs), labels).backward()
    optimizer.second_step(zero_grad=True)  # step 2: real update
```

**Key hyperparameters:**
```yaml
# configs/baseline.yaml
optimizer:
  name: SAM
  base_optimizer: AdamW
  rho: 0.05
  lr: 1e-4
  weight_decay: 1e-4

training:
  epochs: 50
  batch_size: 16
  sample_rate: 22050
  target_duration: 8     # seconds
  n_mels: 128
  warmup_epochs: 5
  early_stopping_patience: 10

scheduler:
  name: CosineAnnealingLR
  T_max: 50
```

---

## ⚖️ Class Imbalance Strategy

### Weighted Sampling (primary strategy)
```python
from torch.utils.data import WeightedRandomSampler

class_counts = [3642, 1864, 886, 506]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

### Weighted Loss (combine with sampling)
```python
weights = torch.tensor([1.0, 1.95, 4.11, 7.20]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
# OR
criterion = FocalLoss(gamma=2.0, alpha=weights)
```

---

## 📊 Evaluation Metrics (ICBHI Official Protocol)

> ⚠️ The ICBHI protocol evaluates **binary** classification: Normal vs. Abnormal  
> Abnormal = crackle OR wheeze OR both

```python
def compute_icbhi_metrics(y_true, y_pred):
    """
    y_true, y_pred: arrays of {0,1,2,3}
    Returns Se (sensitivity), Sp (specificity), Score
    """
    # Convert to binary: 0 = normal, 1 = abnormal
    binary_true = (y_true > 0).astype(int)
    binary_pred = (y_pred > 0).astype(int)

    TP = ((binary_pred == 1) & (binary_true == 1)).sum()
    TN = ((binary_pred == 0) & (binary_true == 0)).sum()
    FP = ((binary_pred == 1) & (binary_true == 0)).sum()
    FN = ((binary_pred == 0) & (binary_true == 1)).sum()

    Se = TP / (TP + FN + 1e-8)   # Sensitivity / Recall
    Sp = TN / (TN + FP + 1e-8)   # Specificity
    Score = (Se + Sp) / 2.0

    return {"Se": Se, "Sp": Sp, "Score": Score}
```

**Reference paper targets to beat:**
| Metric | Reference (AST+SAM) | Your Target |
|--------|---------------------|-------------|
| Sensitivity (Se) | 68.31% | **> 68.31%** |
| Specificity (Sp) | 67.89% | maintain |
| ICBHI Score | 68.10% | **> 68.10%** |

---

## 🔁 Train/Test Split Rule

> ⚠️ **CRITICAL**: Split must be **patient-level**, not cycle-level.  
> Cycles from the same patient must all be in the same split.  
> Using a random cycle-level split causes severe data leakage.

```python
# Official 60/40 patient split
# Use the pre-defined split files from the ICBHI dataset
# File: data/splits/train.csv  and  data/splits/test.csv
# Each row: patient_id, recording_file, cycle_idx, label
```

---

## 🧪 Experiment Tracking

Always log these per run:
```python
metrics_to_log = {
    "epoch": epoch,
    "train_loss": ...,
    "val_loss": ...,
    "val_Se": ...,      # ← most important
    "val_Sp": ...,
    "val_Score": ...,
    "val_acc": ...,
    "lr": scheduler.get_last_lr()[0]
}
# Use wandb.log(metrics_to_log) or save to CSV
```

Save checkpoints when `val_Se` improves:
```python
if val_Se > best_Se:
    best_Se = val_Se
    torch.save(model.state_dict(), "checkpoints/best_model.pt")
```

---

## 🔬 Improvement Ideas to Try (Beyond the Reference Paper)

| Idea | Expected Impact | Complexity |
|------|----------------|------------|
| Focal Loss (γ=2) | Better minority class recall | Low |
| SpecAugment (stronger masking) | Regularization | Low |
| Mixup on spectrograms | Better generalization | Medium |
| Threshold tuning (not 0.5) | Direct Se boost | Low |
| Label smoothing (ε=0.1) | Reduce overconfidence | Low |
| SMOTE on embeddings | More "both" samples | Medium |
| Ensemble (AST + CNN) | Score improvement | High |
| Patch-Mix augmentation | Bae et al. strategy | Medium |

---

## 🌐 Environment: Local ↔ Colab Workflow

```
Local (VS Code)                GitHub               Google Colab
─────────────────         ──────────────         ────────────────────
Write src/ code           git push/pull          !git clone <repo>
Debug preprocessing       version control        GPU: Tesla T4/L4
configs/ editing          experiment logs        !pip install -r requirements_colab.txt
                                                 Mount Google Drive for data+checkpoints
```

**Colab notebook header (always start with this):**
```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone/pull latest code
!git clone https://github.com/YOUR_USERNAME/respiratory-classification.git
%cd respiratory-classification
!pip install -r requirements_colab.txt

# Symlink dataset from Drive
import os
os.makedirs("data/raw", exist_ok=True)
!ln -s /content/drive/MyDrive/ICBHI_2017/ data/raw
```

---

## 📦 Required Libraries

```txt
# requirements.txt
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.35.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0
wandb>=0.15.0
timm>=0.9.0
```

---

## 🚫 Common Mistakes — Copilot Must Avoid These

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| Zero-padding short cycles | Cyclic padding to 8s |
| Random cycle-level train/test split | Patient-level official split |
| Training AST from scratch | Load AudioSet pretrained weights |
| Standard Adam optimizer | SAM wrapping AdamW |
| Evaluating with 4-class accuracy only | Always compute ICBHI Se/Sp/Score |
| Normalizing per-dataset | Normalize per-sample (mean/std per spectrogram) |
| Hardcoding hyperparameters | Load from `configs/baseline.yaml` |
| Saving model every epoch | Save only when `val_Se` improves |

---

## 📋 Deliverables Checklist

- [ ] `src/` — clean, modular Python code
- [ ] `configs/baseline.yaml` — reproducible hyperparameters
- [ ] `notebooks/01_EDA.ipynb` — class distribution, audio samples visualization
- [ ] `notebooks/03_Results_Analysis.ipynb` — confusion matrix, t-SNE, attention maps
- [ ] Comparison table: your results vs. reference paper
- [ ] Final report: architecture, preprocessing, training strategy, results
- [ ] ZIP of complete code for submission

---

*This skill file is the single source of truth for GitHub Copilot in this project.*  
*Last updated: May 2026 | Course: GL4-RT4 | Dataset: ICBHI 2017*