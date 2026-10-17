"""Product segmentation: score only the product, not the backdrop.

On MVTec screw, lint and dust on the backdrop produced anomaly-map peaks that approved 4/88 defect claims for
the wrong reason (docs/dev-log.md). The claim score is therefore the anomaly map's maximum inside the product
region. Calibration (cv/evaluate.py) and the live inspector both call these functions, so they cannot drift apart.

Segmentation can fail on real photos (clutter, unusual backdrops). The policy checks the product's coverage of
the frame against a range calibrated on training images and escalates anything outside it (fail-closed).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

MAP_SIZE = 256        # anomaly maps are 256x256
MARGIN_PX = 12        # at map resolution: the blurred map spills ~8 px off the product (measured on screw)
RANGE_LOW = 0.5       # plausible coverage = [0.5 x smallest, 1.5 x largest] seen on training images
RANGE_HIGH = 1.5


def object_mask(gray: np.ndarray) -> np.ndarray:
    """Dark product on a light backdrop: Otsu threshold, keep the largest connected component."""
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m > 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def product_region(image: np.ndarray, size: int = MAP_SIZE, margin: int | None = None) -> tuple[np.ndarray, float]:
    """(region at map resolution incl. margin, raw fraction of the frame covered by the product).
    The default margin is MARGIN_PX at 256 px, scaled to the requested map size."""
    if margin is None:
        margin = round(MARGIN_PX * size / MAP_SIZE)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = object_mask(gray)
    small = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    if margin > 0:
        small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    return small > 0, float(mask.mean())


def masked_score(amap: np.ndarray, region: np.ndarray) -> float:
    """Anomaly score of the product: the map maximum inside the region (NaN if the region is empty)."""
    return float(amap[region].max()) if region.any() else math.nan


def predicted_mask(amap: np.ndarray, region: np.ndarray, threshold: float) -> np.ndarray:
    """Predicted damage: anomaly-map pixels at or above the threshold, inside the product region.
    The deployed threshold is the decision (approve) threshold, not one tuned for segmentation."""
    return (amap >= threshold) & region


def overlap(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """(IoU, Dice) of two boolean masks; NaN when both are empty (nothing to compare)."""
    inter = float((pred & truth).sum())
    union = float((pred | truth).sum())
    total = float(pred.sum() + truth.sum())
    if union == 0:
        return math.nan, math.nan
    return inter / union, 2 * inter / total


def calibrate_object_range(fractions: Sequence[float]) -> tuple[float, float]:
    vals = [float(f) for f in fractions]
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError("need at least one finite coverage fraction")
    return RANGE_LOW * min(vals), min(1.0, RANGE_HIGH * max(vals))
