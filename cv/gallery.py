"""Per-image data and rendering for the heatmap gallery (Streamlit page: app_pages/gallery.py).

Every test image gets: photo, anomaly map, ground-truth mask, score, out-of-fold verdict, and a
"pointing game" check — does the map's hottest pixel land on the labelled defect? Pixel AUROC
pools all pixels of all images, so it can be high while individual maps point at the wrong place.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from claims.inspector import heat_alpha
from claims.policy import Band
from cv.evaluate import load_mask

ROOT = Path(__file__).parent.parent
MAP_SIZE = 256
TOLERANCE_PX = 6  # at 256 px; the peak counts as a hit if within this distance of the mask


def peak_hits_defect(amap: np.ndarray, mask: np.ndarray, tolerance_px: int = TOLERANCE_PX) -> bool | None:
    """True if the map's maximum lies on the (slightly dilated) defect mask; None for images without a defect."""
    if not mask.any():
        return None
    k = 2 * tolerance_px + 1
    near = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
    return bool(near[y, x])


def peak_location(amap: np.ndarray) -> tuple[int, int]:
    y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
    return int(x), int(y)


def render(image: np.ndarray, amap: np.ndarray, mask: np.ndarray, band: Band, scale: str = "fixed",
           region: np.ndarray | None = None, show_outside: bool = True, pred: np.ndarray | None = None) -> np.ndarray:
    """RGB photo with the anomaly map overlaid, the defect outlined in white, the map's peak marked, and
    (if given) the scored product region outlined in green: the claim score only counts heat inside it.
    show_outside=False hides heat outside the region; pred (map resolution) is outlined in magenta.

    scale="fixed": the demo's per-product scale (red = at/above the approve threshold).
    scale="per-image": stretched to this image's own min..max, so the map's shape is visible even when it is
    entirely below the threshold or entirely saturated."""
    h, w = image.shape[:2]
    amap = cv2.resize(amap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if scale == "fixed":
        norm, alpha = heat_alpha(amap, band)
    else:
        lo, hi = float(amap.min()), float(amap.max())
        norm = (amap - lo) / max(hi - lo, 1e-9)
        alpha = np.clip((norm - 0.2) / 0.3, 0, 1) * 0.6
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].astype(np.float32)
    g = None
    if region is not None and region.any():
        g = cv2.resize(region.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        if not show_outside:
            alpha = alpha * g
    a = alpha[..., None]
    out = (image.astype(np.float32) * (1 - a) + heat * a).astype(np.uint8)
    if g is not None:
        contours, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (40, 200, 90), max(1, w // 300))
    if pred is not None and pred.any():
        pm = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (230, 60, 220), max(2, w // 250))
    if mask.any():
        m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), max(2, w // 250))
    x, y = np.unravel_index(int(np.argmax(amap)), amap.shape)[::-1]
    r = max(6, w // 60)
    cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), max(2, w // 300) + 2)
    cv2.circle(out, (int(x), int(y)), r, (255, 230, 0), max(2, w // 300))
    return out


def load_image(rel: str, data_root: Path = ROOT / "data" / "mvtec") -> np.ndarray:
    img = cv2.imread(str(data_root / rel), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(data_root / rel)
    return img[..., ::-1].copy()


def load_maps(raw_dir: Path = ROOT / "reports" / "raw") -> dict[str, np.ndarray]:
    maps = {}
    for f in sorted(raw_dir.glob("maps_*.npz")):
        with np.load(f) as z:
            maps.update({k: z[k].astype(np.float32) for k in z.files})
    return maps


def build_table(maps: dict[str, np.ndarray], reports: Path = ROOT / "reports",
                data_root: Path = ROOT / "data" / "mvtec") -> pd.DataFrame:
    df = pd.read_csv(reports / "oof_verdicts.csv").fillna({"mask": ""})
    hits, peaks, areas = [], [], []
    for row in df.itertuples():
        amap = maps[row.image]
        mask = load_mask(data_root, row.mask, MAP_SIZE)
        hits.append(peak_hits_defect(amap, mask))
        peaks.append(float(amap.max()))
        areas.append(float(mask.mean()))
    df["peak_hits_defect"] = hits
    df["map_peak"] = peaks
    df["gt_area"] = areas
    vlm = reports / "vlm_crosscheck.csv"
    if vlm.exists():
        rows = [r for r in csv.DictReader(open(vlm, encoding="utf-8")) if r["set"] != "foreign"]
        desc = {r["image"].removeprefix("data/mvtec/"): r for r in rows}
        df["vlm_defect_visible"] = [desc.get(i, {}).get("defect_visible", "") for i in df.image]
        df["vlm_description"] = [desc.get(i, {}).get("description", "") for i in df.image]
    return df


def add_localization(df: pd.DataFrame, maps: dict[str, np.ndarray], bands: dict,
                     data_root: Path = ROOT / "data" / "mvtec") -> pd.DataFrame:
    """Per image: predicted damage mask (heat >= the product's approve threshold, inside the segmented product)
    compared with the ground-truth mask. iou/dice are NaN for good images (no defect to locate)."""
    from claims.segment import overlap, predicted_mask, product_region
    ious, dices, pred_px = [], [], []
    for row in df.itertuples():
        amap = maps[row.image].astype(np.float32)
        region, _ = product_region(load_image(row.image, data_root), size=amap.shape[0])
        pred = predicted_mask(amap, region, bands[row.category].hi)
        pred_px.append(int(pred.sum()))
        if row.label == 1:
            iou, dice = overlap(pred, load_mask(data_root, row.mask, amap.shape[0]))
        else:
            iou, dice = np.nan, np.nan
        ious.append(iou)
        dices.append(dice)
    return df.assign(iou=ious, dice=dices, pred_px=pred_px)


def localization_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.label == 1]
    g = d.groupby(["category", "defect_type"])
    out = pd.DataFrame({"defective images": g.size(), "median IoU": g.iou.median(), "median Dice": g.dice.median(),
                        "empty predictions": g.pred_px.apply(lambda s: int((s == 0).sum()))})
    return out


def pointing_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.label == 1]
    g = d.groupby(["category", "defect_type"])["peak_hits_defect"]
    out = pd.DataFrame({"defective images": g.size(), "peak on defect": g.sum().astype(int)})
    out["hit rate"] = out["peak on defect"] / out["defective images"]
    return out
