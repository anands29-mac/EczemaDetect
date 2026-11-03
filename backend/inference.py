"""Model loading, prediction and Grad-CAM explanations (no web code here)."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eczema.config import NORMAL_CLASS  # noqa: E402
from eczema.data import build_transforms  # noqa: E402
from eczema.models import build_model  # noqa: E402

# Below this top-class probability we tell the user the model is unsure.
UNCERTAIN_BELOW = 0.45


class Predictor:
    def __init__(self, ckpt_path: Path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.meta = ck["meta"]
        self.classes: list[str] = self.meta["classes"]
        self.model = build_model(self.meta["arch"], len(self.classes))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.transform = build_transforms(self.meta["img_size"], train=False)
        self.normal_idx = self.classes.index(NORMAL_CLASS)
        self._target_layer = self.model.conv_head if self.meta["arch"] == "efficientnet_b0" else self.model.features[-1]

    # ------------------------------------------------------------------ predict
    def predict(self, img: Image.Image) -> dict:
        x = self.transform(img).unsqueeze(0)

        # Grad-CAM: capture the last conv feature maps and their gradients.
        acts, grads = {}, {}
        h1 = self._target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        h2 = self._target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
        try:
            logits = self.model(x)
            with torch.no_grad():  # test-time augmentation: also look at the mirror image
                logits_tta = (logits.detach() + self.model(torch.flip(x, dims=[3]))) / 2
            probs = torch.softmax(logits_tta / self.temperature, 1)[0].numpy()
            top = int(probs.argmax())
            self.model.zero_grad()
            logits[0, top].backward()
            cam = self._gradcam(acts["a"], grads["g"])
        finally:
            h1.remove()
            h2.remove()

        order = probs.argsort()[::-1]
        p_eczema = float(1 - probs[self.normal_idx])
        eczema_order = [i for i in order if i != self.normal_idx]
        return {
            "prediction": self.classes[top],
            "confidence": float(probs[top]),
            "uncertain": bool(probs[top] < UNCERTAIN_BELOW),
            "eczema_probability": p_eczema,
            "is_eczema": top != self.normal_idx,
            "most_likely_eczema_type": self.classes[eczema_order[0]],
            "probabilities": [{"label": self.classes[i], "probability": float(probs[i])} for i in order],
            "heatmap_png": self._overlay(img, cam),
        }

    @staticmethod
    def _gradcam(a: torch.Tensor, g: torch.Tensor) -> np.ndarray:
        weights = g.mean(dim=(2, 3), keepdim=True)          # importance of each channel
        cam = torch.relu((weights * a).sum(1))[0].detach()  # weighted sum of feature maps
        cam = cam / (cam.max() + 1e-8)
        return cam.numpy()

    def _overlay(self, img: Image.Image, cam: np.ndarray) -> str:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import colormaps

        size = self.meta["img_size"]
        # same resize + centre-crop the model saw
        base = img.copy()
        base.thumbnail((int(size * 1.14) * 4, int(size * 1.14) * 4))
        w, h = base.size
        s = int(size * 1.14) / min(w, h)
        base = base.resize((round(w * s), round(h * s)), Image.BICUBIC)
        l, t = (base.width - size) // 2, (base.height - size) // 2
        base = base.crop((l, t, l + size, t + size)).resize((448, 448), Image.BICUBIC)
        heat = Image.fromarray(np.uint8(cam * 255)).resize((448, 448), Image.BICUBIC)
        rgba = colormaps["jet"](np.asarray(heat) / 255.0)
        rgba[..., 3] = 0.45 * (np.asarray(heat) / 255.0) ** 0.8
        over = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(np.uint8(rgba * 255)))
        buf = io.BytesIO()
        over.convert("RGB").save(buf, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def skin_pixel_fraction(img: Image.Image) -> float:
    """Rough check that the photo actually shows skin (YCrCb skin-colour range).

    Deliberately loose so it works across skin tones; used only to show a
    warning, never to block a result.
    """
    small = img.copy()
    small.thumbnail((128, 128))
    ycbcr = np.asarray(small.convert("YCbCr")).astype(np.int16)
    cb, cr = ycbcr[..., 1], ycbcr[..., 2]
    mask = (cr >= 128) & (cr <= 180) & (cb >= 70) & (cb <= 135)
    return float(mask.mean())
