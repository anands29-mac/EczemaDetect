"""EczemaDetect API + dashboard server.

Run from the project root:
    uvicorn backend.app:app --reload --port 8000
then open http://localhost:8000
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import Predictor, skin_pixel_fraction

from eczema.data import load_image  # noqa: E402  (src/ is put on sys.path by inference.py)

ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE = json.loads((Path(__file__).parent / "knowledge.json").read_text(encoding="utf-8"))
MAX_BYTES = 12 * 1024 * 1024


def _find_checkpoint() -> Path:
    env = os.getenv("MODEL_PATH")
    candidates = [Path(env)] if env else [ROOT / "models" / "efficientnet_b0.pt", ROOT / "models" / "simple_cnn.pt"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No trained model found in models/. Run src/train.py first.")


app = FastAPI(title="EczemaDetect API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor(_find_checkpoint())


@app.get("/api/health")
def health():
    m = predictor.meta
    return {"status": "ok", "model": m["arch"], "img_size": m["img_size"], "classes": predictor.classes,
            "trained_epoch": m.get("epoch"), "test_metrics": m.get("test")}


@app.get("/api/conditions")
def conditions():
    return KNOWLEDGE


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 12 MB.")
    try:
        img = load_image(io.BytesIO(data), max_side_hint=1024)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't a readable image (use JPG, PNG or WEBP).")
    if min(img.size) < 64:
        raise HTTPException(400, "Image is too small; please upload a clearer, closer photo.")

    result = predictor.predict(img)
    skin = skin_pixel_fraction(img)

    warnings = []
    if skin < 0.15:
        warnings.append("This photo doesn't look like it is mostly skin. Results for non-skin photos are meaningless.")
    if result["uncertain"]:
        warnings.append("The model is unsure: no class is clearly ahead. Try a closer, well-lit photo of the affected area.")

    info = KNOWLEDGE["conditions"][result["prediction"]]
    return {
        **result,
        "skin_fraction": skin,
        "warnings": warnings,
        "condition": {"name": result["prediction"], **info},
        "general_care": KNOWLEDGE["general_care"],
        "red_flags": KNOWLEDGE["red_flags"],
        "disclaimer": KNOWLEDGE["disclaimer"],
        "sources": KNOWLEDGE["sources"],
        "model": {"arch": predictor.meta["arch"], "test_metrics": predictor.meta.get("test")},
    }


# ---- serve the dashboard -----------------------------------------------------------
FRONTEND = ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
