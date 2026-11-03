"""Shared settings used by training, evaluation and the backend.

Keeping these in one place means the backend always preprocesses images exactly
the same way the model saw them during training (a very common source of bugs).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Order matters: index i in the model's output == CLASSES[i].
CLASSES = [
    "Atopic Dermatitis",
    "Contact Dermatitis",
    "Dyshidrotic Eczema",
    "Normal",
    "Nummular Dermatitis",
    "Seborrheic Dermatitis",
    "Stasis Dermatitis",
]
NORMAL_CLASS = "Normal"

# ImageNet statistics. Pretrained backbones expect inputs normalised this way;
# we use them for the from-scratch CNN too so both models share one pipeline.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
