"""Per-image dataset properties, each tied to an error hypothesis.

  gt_area          defect size (mask fraction)           -> are small defects escalated/missed?
  gt_dx, gt_dy     defect centroid offset from centre    -> are edge defects missed?
  bg_brightness    background lighting                   -> does lighting drift from train raise scores?
  angle            screw orientation, head direction     -> do rare orientations raise good-image scores?

Output: reports/dataset_features.csv (train/good + all test images, joined with
scores and out-of-fold verdicts for test images).
Usage: python -m cv.analyze_dataset
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from claims.segment import object_mask  # shared with the live inspector

DATA = Path("data/mvtec")


def _gray(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def object_angle(gray: np.ndarray) -> float:
    """Orientation of the object's long axis in degrees [0, 360), counter-clockwise from +x,
    pointing towards the wider end (the screw head)."""
    ys, xs = np.nonzero(object_mask(gray))
    pts = np.stack([xs - xs.mean(), -(ys - ys.mean())], axis=1)  # flip y so angles are counter-clockwise
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    axis = vecs[:, -1]
    along = pts @ axis
    across = np.abs(pts @ np.array([-axis[1], axis[0]]))
    lo, hi = np.quantile(along, [0.15, 0.85])
    if across[along <= lo].mean() > across[along >= hi].mean():  # head (wider end) on the negative side
        axis = -axis
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 360)


def background_brightness(gray: np.ndarray) -> float:
    return float(np.median(gray[~object_mask(gray)]))


def mask_features(mask_path: Path) -> dict:
    m = _gray(mask_path) > 127
    if not m.any():
        return {"gt_area": 0.0, "gt_dx": np.nan, "gt_dy": np.nan}
    ys, xs = np.nonzero(m)
    h, w = m.shape
    return {"gt_area": float(m.mean()), "gt_dx": float(xs.mean() / w - 0.5), "gt_dy": float(ys.mean() / h - 0.5)}


def image_features(category: str, rel: str) -> dict:
    gray = _gray(DATA / rel)
    feats = {"bg_brightness": background_brightness(gray), "contrast": float(gray.std())}
    feats["angle"] = object_angle(gray) if category == "screw" else np.nan
    return feats


def build(data_root: Path = DATA, reports: Path = Path("reports")) -> pd.DataFrame:
    rows = []
    for category in ("bottle", "screw"):
        for img in sorted((data_root / category / "train/good").glob("*.png")):
            rel = img.relative_to(data_root).as_posix()
            rows.append({"category": category, "split": "train", "defect_type": "good", "image": rel,
                         **image_features(category, rel)})
    oof = pd.read_csv(reports / "oof_verdicts.csv", keep_default_na=False)
    for r in oof.itertuples():
        feats = image_features(r.category, r.image)
        if r.mask:
            feats.update(mask_features(data_root / r.mask))
        rows.append({"category": r.category, "split": "test", "defect_type": r.defect_type, "image": r.image,
                     "label": r.label, "score": r.score, "verdict": r.verdict, **feats})
    df = pd.DataFrame(rows)
    df.to_csv(reports / "dataset_features.csv", index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(df.groupby(["category", "split", "defect_type"]).size().to_string())
