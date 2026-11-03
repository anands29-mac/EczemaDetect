"""Step 2 — train a CNN.

Examples
--------
    # small CNN from scratch (the learning exercise)
    python src/train.py --arch simple_cnn --img-size 160 --epochs 30 --lr 3e-3

    # transfer learning (the model the dashboard uses)
    python src/train.py --arch efficientnet_b0 --img-size 224 --epochs 12 --lr 1e-3 --freeze-epochs 2

What happens each epoch
-----------------------
for every batch:
    logits = model(images)              # forward pass
    loss   = cross_entropy(logits, y)   # how wrong were we?
    loss.backward()                     # backprop: d(loss)/d(every weight)
    optimizer.step()                    # nudge weights downhill
then measure on the validation set (no gradient, no augmentation) and keep the
checkpoint with the best validation macro-F1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eczema.config import CLASSES, DATA_DIR, MEAN, MODELS_DIR, REPORTS_DIR, STD  # noqa: E402
from eczema.data import SkinDataset, build_transforms, read_split  # noqa: E402
from eczema.models import build_model, count_params  # noqa: E402


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Linear warm-up, then cosine decay to ~0. Smooth LR schedules train noticeably better."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y, total_loss = [], [], 0.0
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    pred = logits.argmax(1).numpy()
    return {
        "loss": total_loss / len(y),
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def set_backbone_trainable(model, arch: str, trainable: bool):
    if arch == "simple_cnn":
        return
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0", choices=["simple_cnn", "efficientnet_b0"])
    ap.add_argument("--splits", default=str(DATA_DIR / "splits.csv"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--freeze-epochs", type=int, default=0, help="train only the new head for the first N epochs")
    ap.add_argument("--patience", type=int, default=6, help="early-stop after N epochs without val improvement")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pretrained", default=str(MODELS_DIR / "efficientnet_b0_imagenet.pth"),
                    help="local ImageNet weights; if missing, timm downloads them")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training images")
    ap.add_argument("--resume", action="store_true", help="continue from models/<arch>_last.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    train_items = read_split(Path(args.splits), "train")
    val_items = read_split(Path(args.splits), "val")
    if args.limit:
        rng = np.random.default_rng(0)
        train_items = [train_items[i] for i in rng.permutation(len(train_items))[: args.limit]]
        val_items = val_items[: max(64, args.limit // 4)]
    print(f"train: {len(train_items)}  val: {len(val_items)}")

    train_dl = DataLoader(SkinDataset(train_items, build_transforms(args.img_size, True)),
                          batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(SkinDataset(val_items, build_transforms(args.img_size, False)),
                        batch_size=args.batch_size * 2, num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    model = build_model(args.arch, len(CLASSES), args.pretrained if args.arch != "simple_cnn" else None).to(device)
    print(f"{args.arch}: {count_params(model) / 1e6:.2f} M parameters")

    # Label smoothing: target 0.9 for the true class instead of 1.0 -> less over-confident predictions.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    best_path = MODELS_DIR / f"{args.arch}.pt"
    last_path = MODELS_DIR / f"{args.arch}_last.pt"
    history, best_f1, start_epoch, bad_epochs = [], -1.0, 0, 0

    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        history, best_f1, start_epoch, bad_epochs = ck["history"], ck["best_f1"], ck["epoch"] + 1, ck["bad_epochs"]
        print(f"resumed at epoch {start_epoch}")

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    meta = {"arch": args.arch, "classes": CLASSES, "img_size": args.img_size, "mean": MEAN, "std": STD}

    for epoch in range(start_epoch, args.epochs):
        frozen = epoch < args.freeze_epochs
        set_backbone_trainable(model, args.arch, not frozen)
        model.train()
        t0, running, seen, correct = time.time(), 0.0, 0, 0
        for i, (x, y) in enumerate(train_dl):
            step = epoch * steps_per_epoch + i
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup=steps_per_epoch // 2, base_lr=args.lr)
            x, y = x.to(device), y.to(device)

            logits = model(x)                 # forward
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()                   # backward
            optimizer.step()                  # update

            running += loss.item() * len(y)
            seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            if i % 20 == 0:
                print(f"  ep {epoch + 1} [{i}/{steps_per_epoch}] loss {running / seen:.3f} "
                      f"acc {correct / seen:.3f} ({time.time() - t0:.0f}s)", flush=True)

        val = evaluate(model, val_dl, device)
        rec = {"epoch": epoch + 1, "train_loss": running / seen, "train_acc": correct / seen,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
               "frozen_backbone": frozen, "seconds": round(time.time() - t0)}
        history.append(rec)
        print(json.dumps(rec), flush=True)

        if val["macro_f1"] > best_f1:
            best_f1, bad_epochs = val["macro_f1"], 0
            torch.save({"model": model.state_dict(), "meta": {**meta, "epoch": epoch + 1, "val": val}}, best_path)
            print(f"  ** new best val macro-F1 {best_f1:.4f} -> saved {best_path.name}")
        else:
            bad_epochs += 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history,
                    "best_f1": best_f1, "epoch": epoch, "bad_epochs": bad_epochs}, last_path)
        (REPORTS_DIR / f"{args.arch}_history.json").write_text(json.dumps(history, indent=2))

        if bad_epochs >= args.patience:
            print("early stopping")
            break

    plot_history(history, REPORTS_DIR / f"{args.arch}_curves.png", args.arch)
    print(f"best val macro-F1: {best_f1:.4f}")


def plot_history(history, out: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss"), ax[0].set_xlabel("epoch"), ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train acc")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val acc")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1", linestyle="--")
    ax[1].set_title("Accuracy / F1"), ax[1].set_xlabel("epoch"), ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
