import torch
import yaml
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import load_model
from src.preprocessing import preprocess_cycle

# ── load config ──
with open("configs/baseline.yaml") as f:
    config = yaml.safe_load(f)

# ── load model ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(use_pretrained=True, freeze_backbone=False)
ckpt  = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

print(f"Model loaded — best Se was: {ckpt['best_Se']:.4f}")

# ── run inference on one cycle ──
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]

def predict(wav_path: str, start: float, end: float) -> str:
    spec  = preprocess_cycle(wav_path, start, end, config)        # (128, T)
    tensor = torch.tensor(spec, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)          # (1, 1, 128, T)

    with torch.no_grad():
        logits = model(tensor)                                     # (1, 4)
        probs  = torch.softmax(logits, dim=1).squeeze()
        pred   = probs.argmax().item()

    print(f"Prediction : {CLASS_NAMES[pred]}")
    print(f"Confidence : {probs[pred]:.2%}")
    print(f"All probs  : { {c: f'{p:.2%}' for c, p in zip(CLASS_NAMES, probs.tolist())} }")
    return CLASS_NAMES[pred]


# ── example usage ──
if __name__ == "__main__":
    # Replace with any wav file and cycle timestamps from your dataset
    predict(
        wav_path="data/raw/101_1b1_Al_sc_Meditron.wav",
        start=0.036,
        end=1.032
    )