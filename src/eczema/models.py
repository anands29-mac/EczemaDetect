"""Model definitions.

1. `SimpleCNN`  - a small convolutional network built layer by layer, trained
                  from random weights. Read this one to understand CNNs.
2. `efficientnet_b0` - a proven ImageNet architecture whose pretrained weights
                  we fine-tune ("transfer learning"). This is the one to deploy:
                  with ~7k training images, starting from features that already
                  know edges, textures and colours beats starting from zero.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions + BatchNorm + ReLU, then 2x2 max-pool (halves H and W).

    * Conv2d learns small filters (edges, scales, redness blobs ...).
    * BatchNorm keeps activations well-scaled -> faster, more stable training.
    * ReLU adds non-linearity.
    * MaxPool shrinks the feature map so later layers "see" a bigger area.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Input 3xHxW  ->  4 conv blocks (32,64,128,256 channels)  ->  global average pool  ->  classifier.

    With 160x160 input the spatial size goes 160 -> 80 -> 40 -> 20 -> 10, then the
    global average pool turns each of the 256 feature maps into one number.
    ~1.2 M parameters (vs ~4 M for EfficientNet-B0, ~25 M for ResNet-50).
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(arch: str, num_classes: int, pretrained_path: str | Path | None = None) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN(num_classes)

    if arch == "efficientnet_b0":
        import timm

        model = timm.create_model("efficientnet_b0", pretrained=False)
        if pretrained_path and Path(pretrained_path).exists():
            model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
            print(f"loaded ImageNet weights from {pretrained_path}")
        elif pretrained_path is not None:
            # Downloads from the internet the first time (cached afterwards).
            model = timm.create_model("efficientnet_b0", pretrained=True)
            print("loaded ImageNet weights via timm")
        model.reset_classifier(num_classes)  # swap the 1000-class ImageNet head for ours
        return model

    raise ValueError(f"unknown arch {arch!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
