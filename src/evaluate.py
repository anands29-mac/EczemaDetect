"""Step 3 — evaluate a trained model on the held-out TEST split (never used for training or model selection).

    python src/evaluate.py --ckpt models/efficientnet_b0.pt

Outputs (in reports/):
  * <arch>_test_report.json   accuracy, macro-F1, per-class precision/recall, eczema-vs-normal metrics
  * <arch>_confusion.png      which classes get confused with which
It also fits a *temperature* on the validation set (a single number that makes the
softmax confidences honest) and stores it next to the checkpoint for the backend.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eczema.config import CLASSES, DATA_DIR, NORMAL_CLASS, REPORTS_DIR  # noqa: E402
from eczema.data import SkinDataset, build_transforms, read_split  # noqa: E402
from eczema.models import build_model  # noqa: E402


@torch.no_grad()
def predict_logits(model, items, img_size, tta: bool):
    dl = DataLoader(SkinDataset(items, build_transforms(img_size, False)), batch_size=64, num_workers=2)
    out, ys = [], []
    for x, y in dl:
        logits = model(x)
        if tta:  # test-time augmentation: average with the mirrored image
            logits = (logits + model(torch.flip(x, dims=[3]))) / 2
        out.append(logits)
        ys.append(y)
    return torch.cat(out), torch.cat(ys).numpy()


def fit_temperature(logits: torch.Tensor, y: np.ndarray) -> float:
    """Find T minimising NLL of softmax(logits / T) on validation data (Guo et al., 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    yt = torch.from_numpy(y)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def plot_confusion(cm: np.ndarray, out: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / cm.sum(1, keepdims=True)
    short = [c.replace(" Dermatitis", "").replace(" Eczema", "") for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), short, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)), short)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else "black")
    fig.colorbar(im, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default=str(DATA_DIR / "splits.csv"))
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = build_model(meta["arch"], len(CLASSES))
    model.load_state_dict(ck["model"])
    model.eval()
    torch.set_grad_enabled(False)
    tta = not args.no_tta

    val_logits, val_y = predict_logits(model, read_split(Path(args.splits), "val"), meta["img_size"], tta)
    temperature = fit_temperature(val_logits, val_y)

    logits, y = predict_logits(model, read_split(Path(args.splits), "test"), meta["img_size"], tta)
    probs = torch.softmax(logits / temperature, 1).numpy()
    pred = probs.argmax(1)

    normal = CLASSES.index(NORMAL_CLASS)
    is_eczema_true = y != normal
    p_eczema = 1 - probs[:, normal]
    is_eczema_pred = pred != normal
    ecz_mask = is_eczema_true & is_eczema_pred

    report = {
        "arch": meta["arch"], "checkpoint_epoch": meta.get("epoch"), "tta": tta, "temperature": temperature,
        "test_images": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "top2_accuracy": float(np.mean([y[i] in probs[i].argsort()[-2:] for i in range(len(y))])),
        "eczema_vs_normal": {
            "accuracy": float((is_eczema_true == is_eczema_pred).mean()),
            "sensitivity (eczema caught)": float(is_eczema_pred[is_eczema_true].mean()),
            "specificity (normal kept normal)": float((~is_eczema_pred)[~is_eczema_true].mean()),
            "roc_auc": float(roc_auc_score(is_eczema_true, p_eczema)),
        },
        "subtype_accuracy_on_eczema_images": float((pred[ecz_mask] == y[ecz_mask]).mean()),
        "per_class": classification_report(y, pred, target_names=CLASSES, output_dict=True, digits=3),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{meta['arch']}_test_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(classification_report(y, pred, target_names=CLASSES, digits=3))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))

    cm = confusion_matrix(y, pred)
    plot_confusion(cm, REPORTS_DIR / f"{meta['arch']}_confusion.png",
                   f"{meta['arch']} — test acc {report['accuracy']:.1%}, macro-F1 {report['macro_f1']:.3f}")

    # store calibration with the checkpoint so the API reports honest confidences
    ck["meta"]["temperature"] = temperature
    ck["meta"]["test"] = {k: report[k] for k in ("accuracy", "macro_f1", "top2_accuracy", "eczema_vs_normal")}
    torch.save(ck, args.ckpt)
    print(f"saved {out} and temperature={temperature:.3f} into {args.ckpt}")


if __name__ == "__main__":
    main()
