# 🔧 Fixing ModuleNotFoundError — respiratory-classification
> **GitHub Copilot: read this entire file before suggesting any fix.**  
> This is a recurring issue in this project. Always apply the solution at the ROOT level, not per-file.

---

## The Problem

Running any script directly like this fails:
```powershell
python scripts/inference.py
python scripts/prepare_data.py
python src/train.py
```

Error:
```
ModuleNotFoundError: No module named 'src'
ModuleNotFoundError: No module named 'src.dataset'
ModuleNotFoundError: No module named 'src.preprocessing'
```

---

## Root Cause

Python only finds `src` as a package if the **project root** is on `sys.path`.  
When you run `python scripts/inference.py`, the working directory added to `sys.path`  
is `scripts/` — not the project root — so `import src.dataset` fails.

---

## The Fix — 3 layers, all must be applied

### Layer 1 — Always run from project root with `-m`

**Never run scripts directly. Always use `-m` from the project root.**

```powershell
# ❌ Wrong — causes ModuleNotFoundError
python scripts/inference.py
python scripts/prepare_data.py

# ✅ Correct — run as module from project root
python -m scripts.inference
python -m scripts.prepare_data
python -m src.train
python -m src.train --dry-run
```

### Layer 2 — Add `__init__.py` to scripts/

Copilot must verify this file exists:
```
respiratory-classification/
├── scripts/
│   ├── __init__.py        ← must exist (can be empty)
│   ├── inference.py
│   ├── prepare_data.py
│   └── verify_dataset.py
├── src/
│   ├── __init__.py        ← must exist (can be empty)
│   └── ...
```

Create them if missing:
```powershell
# From project root
New-Item -ItemType File -Force scripts/__init__.py
New-Item -ItemType File -Force src/__init__.py
```

### Layer 3 — Add project root to PYTHONPATH (permanent fix)

**Option A — Set in PowerShell for current session:**
```powershell
$env:PYTHONPATH = "D:\younes\respiratory-classification"
```

**Option B — Set permanently in Windows environment variables:**
```powershell
[System.Environment]::SetEnvironmentVariable(
    "PYTHONPATH",
    "D:\younes\respiratory-classification",
    "User"
)
# Restart VS Code terminal after this
```

**Option C — Add `.env` file at project root (works with VS Code Python extension):**
```
# .env  (project root)
PYTHONPATH=D:\younes\respiratory-classification
```

Then in VS Code `settings.json`:
```json
{
    "python.envFile": "${workspaceFolder}/.env"
}
```

---

## Layer 4 — Add sys.path fix inside scripts (fallback)

If a script must be run directly (not with `-m`), add this at the very top  
of every file in `scripts/` — **before any src imports**:

```python
# Add project root to path — required when running script directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Now src imports work
from src.dataset import ICBHIDataset
from src.preprocessing import preprocess_cycle
```

> ⚠️ Copilot: add this block to the TOP of every file in `scripts/` that imports from `src`.  
> Do NOT add it to files inside `src/` — they import each other relatively.

---

## VS Code Run Configuration

Create `.vscode/launch.json` so the Run button always works correctly:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "inference",
            "type": "python",
            "request": "launch",
            "module": "scripts.inference",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "train (dry-run)",
            "type": "python",
            "request": "launch",
            "module": "src.train",
            "args": ["--dry-run"],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "train (full)",
            "type": "python",
            "request": "launch",
            "module": "src.train",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "prepare_data",
            "type": "python",
            "request": "launch",
            "module": "scripts.prepare_data",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "verify_dataset",
            "type": "python",
            "request": "launch",
            "module": "scripts.verify_dataset",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal"
        }
    ]
}
```

---

## VS Code Interpreter Setting

Make sure VS Code uses the `.venv` interpreter, not system Python:

1. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose `.venv\Scripts\python.exe` (the one showing Python 3.12)
3. Verify in the bottom status bar it shows `3.12.x ('.venv')`

If `.venv` doesn't appear in the list:
```powershell
# From project root in terminal
.venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## Quick Reference — Correct Commands from Project Root

```powershell
# Always activate venv first
.venv\Scripts\activate

# Verify dataset
python -m scripts.verify_dataset

# Prepare splits
python -m scripts.prepare_data

# Dry-run training
python -m src.train --dry-run

# Full training
python -m src.train

# Inference
python -m scripts.inference

# Evaluation notebook
jupyter notebook notebooks/03_Results_Analysis.ipynb
```

---

## Colab Equivalent

In Colab, the same fix applies — always run from the repo root:
```python
%cd /content/respiratory-classification
!python -u -m src.train --checkpoint-dir /content/drive/MyDrive/respiratory_project/checkpoints
```

---

## Checklist — Copilot must verify all of these before running any script

- [ ] Terminal is at project root (`D:\younes\respiratory-classification`)
- [ ] `.venv` is activated (prompt shows `(.venv)`)
- [ ] `scripts/__init__.py` exists
- [ ] `src/__init__.py` exists  
- [ ] Command uses `-m` format OR `sys.path` insert is at top of script
- [ ] VS Code interpreter is set to `.venv\Scripts\python.exe` (Python 3.12)

---

*Apply this fix once. It solves all ModuleNotFoundError issues in this project permanently.*