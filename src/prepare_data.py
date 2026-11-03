"""Step 1 — audit the raw dataset and build a clean, leak-free train/val/test split.

Why this script exists
----------------------
The Kaggle folders in Eczema_dataset/ have problems that silently inflate
accuracy if you train on them as-is:

* test/test/Normal is a byte-for-byte copy of train/train/Normal (1,579 images),
  so any "test accuracy" on Normal is really training accuracy.
* ~140 more images appear in BOTH train and test.
* ~150 identical images carry two different labels (mostly Contact vs Nummular).
* Many images are near-duplicates (same photo, re-saved or resized).

What it does
------------
1. Walks every image under --raw (the class is the parent folder name).
2. Exact duplicates  -> MD5 of the file bytes.
3. Near duplicates   -> 64-bit "difference hash" (dHash); images whose hashes
   differ in <= --near-dup bits are put in the same *group*.
4. Drops images/groups whose copies disagree on the label.
5. Splits by GROUP (stratified by class) into train/val/test, so a photo and its
   near-copy can never land on both sides of the split.
6. Writes data/splits.csv and data/audit.json.

Usage
-----
    python src/prepare_data.py --raw Eczema_dataset
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eczema.config import CLASSES, DATA_DIR, IMAGE_EXTS, PROJECT_ROOT  # noqa: E402


def dhash(path: Path, size: int = 16) -> np.ndarray:
    """Difference hash: shrink to 17x16 greyscale, compare neighbouring pixels.

    Robust to resizing/re-compression, so two copies of one photo get (nearly)
    the same 256-bit fingerprint. (An 8x8 / 64-bit hash was tried first and was
    too coarse on this dataset: unrelated skin close-ups collided.)
    """
    im = Image.open(path)
    im.draft("L", (size * 4, size * 4))  # fast JPEG downscale
    px = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((px[:, 1:] > px[:, :-1]).flatten())  # 32 bytes


def portable(p: Path) -> str:
    """Store paths relative to the project root (with / separators) so splits.csv works on any machine."""
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_duplicate_pairs(as_bytes: np.ndarray, max_bits: int, chunk: int = 256):
    """Yield (i, j) pairs whose packed hashes (rows of bytes) differ in <= max_bits bits."""
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    n = len(as_bytes)
    for start in range(0, n, chunk):
        block = as_bytes[start:start + chunk]
        x = np.bitwise_xor(block[:, None, :], as_bytes[None, :, :])  # (chunk, n, 32)
        dist = popcount[x].sum(-1)
        ii, jj = np.nonzero(dist <= max_bits)
        for i, j in zip(ii + start, jj):
            if i < j:
                yield int(i), int(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Folder containing <class>/<image> (any depth)")
    ap.add_argument("--out", default=str(DATA_DIR / "splits.csv"))
    ap.add_argument("--near-dup", type=int, default=12, help="max differing bits (of 256) to call two images near-duplicates")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw)
    files = [p for p in raw.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.parent.name in CLASSES]
    print(f"found {len(files)} image files")

    # ---- 1. exact duplicates (MD5) -------------------------------------------------
    by_md5: dict[str, dict] = {}
    unreadable = 0
    for k, p in enumerate(files):
        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            entry = by_md5.setdefault(digest, {"path": p, "labels": set(), "copies": 0})
            entry["labels"].add(p.parent.name)
            entry["copies"] += 1
        except OSError:
            unreadable += 1
        if (k + 1) % 2000 == 0:
            print(f"  hashed {k + 1}/{len(files)}")

    conflicting = [d for d, e in by_md5.items() if len(e["labels"]) > 1]
    unique = {d: e for d, e in by_md5.items() if len(e["labels"]) == 1}
    print(f"unique images: {len(by_md5)}  | exact copies removed: {len(files) - len(by_md5)}"
          f"  | label-conflicting images dropped: {len(conflicting)}")

    # ---- 2. near duplicates (dHash) -----------------------------------------------
    keys = list(unique)
    hashes = np.zeros((len(keys), 32), dtype=np.uint8)
    rng_bytes = np.random.default_rng(0)
    for i, d in enumerate(keys):
        try:
            hashes[i] = dhash(unique[d]["path"])
        except Exception:
            hashes[i] = rng_bytes.integers(0, 256, 32)  # unreadable: random hash, matches nothing
        if (i + 1) % 2000 == 0:
            print(f"  dHashed {i + 1}/{len(keys)}")
    uf = UnionFind(len(keys))
    n_pairs = 0
    for i, j in near_duplicate_pairs(hashes, args.near_dup):
        uf.union(i, j)
        n_pairs += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(keys)):
        groups[uf.find(i)].append(i)

    rows, mixed_groups = [], 0
    for gid, members in groups.items():
        labels = {next(iter(unique[keys[i]]["labels"])) for i in members}
        if len(labels) > 1:  # near-identical photos with different labels -> ambiguous
            mixed_groups += 1
            continue
        label = labels.pop()
        for i in members:
            rows.append({"md5": keys[i], "path": portable(unique[keys[i]]["path"]), "label": label, "group": gid})
    print(f"near-duplicate pairs: {n_pairs} | groups: {len(groups)} | mixed-label groups dropped: {mixed_groups}")

    # ---- 3. stratified GROUP split --------------------------------------------------
    rng = random.Random(args.seed)
    by_label_groups: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label_groups[r["label"]][r["group"]].append(r)
    for label, gmap in by_label_groups.items():
        gids = list(gmap)
        rng.shuffle(gids)
        total = sum(len(gmap[g]) for g in gids)
        n_test, n_val, seen = args.test * total, args.val * total, 0
        for g in gids:
            split = "test" if seen < n_test else "val" if seen < n_test + n_val else "train"
            for r in gmap[g]:
                r["split"] = split
            seen += len(gmap[g])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["split"], r["label"], r["path"]))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "group", "md5"])
        w.writeheader()
        w.writerows(rows)

    counts = Counter((r["split"], r["label"]) for r in rows)
    table = {s: {c: counts[(s, c)] for c in CLASSES} for s in ("train", "val", "test")}
    audit = {
        "files_found": len(files), "unreadable": unreadable, "unique_images": len(by_md5),
        "exact_copies_removed": len(files) - len(by_md5),
        "label_conflicting_dropped": len(conflicting), "near_duplicate_pairs": n_pairs,
        "mixed_label_groups_dropped": mixed_groups, "final_images": len(rows), "split_counts": table,
    }
    (out.parent / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(table, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
