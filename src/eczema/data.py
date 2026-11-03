"""Dataset + image transforms.

A PyTorch `Dataset` only needs two methods: `__len__` and `__getitem__`.
The `DataLoader` then batches, shuffles and loads items in parallel workers.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import CLASSES, MEAN, PROJECT_ROOT, STD


def load_image(path_or_file, max_side_hint: int = 512) -> Image.Image:
    """Open an image as RGB, honouring phone EXIF rotation.

    `draft` lets the JPEG decoder shrink big photos while decoding, which makes
    loading 12-megapixel phone pictures several times faster.
    """
    im = Image.open(path_or_file)
    if im.format == "JPEG":
        im.draft("RGB", (max_side_hint, max_side_hint))
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def build_transforms(img_size: int, train: bool) -> T.Compose:
    if train:
        # Data augmentation = showing the network plausible variations of each
        # photo so it learns the *lesion*, not the exact framing/lighting.
        # Hue jitter is kept tiny on purpose: redness is a real clinical cue.
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def read_split(csv_path: Path, split: str) -> list[tuple[Path, int]]:
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            p = Path(row["path"])
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            items.append((p, CLASSES.index(row["label"])))
    return items


class SkinDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = load_image(path)
        return self.transform(img), torch.tensor(label)
