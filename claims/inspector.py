"""Vision inspector: image -> Inspection. No LLM involved.

PatchCoreInspector wraps the exported anomalib model; FakeInspector returns
scripted results so graph tests run in milliseconds without torch.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from claims.policy import Band, Inspection

MIN_SIDE = 64  # smaller images can't carry meaningful defect detail


class InspectionError(RuntimeError):
    pass


class Inspector(Protocol):
    def inspect(self, image_path: Path, product: str) -> Inspection: ...


def check_image(image_path: str | Path | None) -> bool:
    """True if the file exists, decodes as an image, and is large enough to inspect."""
    if not image_path:
        return False
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(image_path) as im:
            return min(im.size) >= MIN_SIDE
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def load_bands(path: Path = Path("config/thresholds.json")) -> dict[str, Band]:
    raw = json.loads(Path(path).read_text())
    return {cat: Band(lo=float(b["lo"]), hi=float(b["hi"]),
                      max_area=float(b["max_area"]) if b.get("max_area") is not None else None,
                      object_range=tuple(map(float, b["object_range"])) if b.get("object_range") else None)
            for cat, b in raw.items()}


def heat_alpha(amap: np.ndarray, band: Band) -> tuple[np.ndarray, np.ndarray]:
    """Colour position and opacity for an anomaly map on one fixed per-product scale.

    Scale runs from 0.75*band.lo to band.hi, so colour means the same thing on every
    photo (red = at or above the approve threshold). Pixels in the bottom fifth of the
    scale are fully transparent, so normal regions show the photo, not a blue tint."""
    lo = 0.75 * band.lo
    norm = np.clip((amap - lo) / max(band.hi - lo, 1e-9), 0, 1)
    alpha = np.clip((norm - 0.2) / 0.3, 0, 1) * 0.6
    return norm, alpha


def save_heatmap_overlay(image_path: Path, amap: np.ndarray, band: Band, out_path: Path,
                         region: np.ndarray | None = None) -> Path:
    """Heat over the photo. With a product region, only the scored region is coloured and it is outlined (green),
    and the predicted damage (heat at or above the approve threshold, inside the product) is outlined in magenta."""
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise InspectionError(f"cannot read {image_path}")
    h, w = img.shape[:2]
    norm, alpha = heat_alpha(cv2.resize(amap.astype(np.float32), (w, h)), band)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    if region is not None:
        r = cv2.resize(region.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = alpha * r
    a = alpha[..., None]
    out = (img * (1 - a) + heat * a).astype(np.uint8)
    if region is not None:
        contours, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (90, 200, 40), max(1, w // 400))  # BGR green: scored region
        from claims.segment import predicted_mask
        pred = cv2.resize(predicted_mask(amap, region, band.hi).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (220, 60, 230), max(2, w // 300))  # BGR magenta: predicted damage
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    return out_path


class PatchCoreInspector:
    def __init__(self, models_dir: Path, bands: Mapping[str, Band], heatmap_dir: Path, device: str = "cpu"):
        self._models_dir = Path(models_dir)
        self._bands = bands
        self._heatmap_dir = Path(heatmap_dir)
        self._device = device
        self._scorers = {}

    def _scorer(self, product: str):
        if product not in self._scorers:
            from cv.patchcore import PatchCoreScorer, model_path
            path = model_path(self._models_dir, product)
            if not path.exists():
                raise InspectionError(f"no model for {product!r} at {path}")
            self._scorers[product] = PatchCoreScorer(path, device=self._device)
        return self._scorers[product]

    def inspect(self, image_path: Path, product: str) -> Inspection:
        import math

        from PIL import Image

        from claims.segment import masked_score, product_region
        image_path = Path(image_path)
        _, amap = self._scorer(product).score(image_path)
        # Thresholds were calibrated on maps stored as float16; quantise identically so a borderline score
        # cannot land on the other side of a threshold (screw: 0.06 between max good and next defect).
        amap = amap.astype(np.float16).astype(np.float32)
        band = self._bands[product]
        # Score only the product: backdrop lint/dust must not decide a claim (see claims/segment.py).
        region, fraction = product_region(np.asarray(Image.open(image_path).convert("RGB")))
        score = masked_score(amap, region)
        if not math.isfinite(score):  # empty cut-out: the coverage rule escalates it; keep the score finite
            score = float(amap.max())
        # Name by content: MVTec photos share stems (good/000.png, broken_large/000.png).
        digest = hashlib.sha1(image_path.read_bytes()).hexdigest()[:12]
        heatmap = save_heatmap_overlay(image_path, amap, band,
                                       self._heatmap_dir / f"{product}_{image_path.stem}_{digest}_heat.png",
                                       region=region)
        return Inspection(score=score, heatmap_path=str(heatmap),
                          defect_area=float((amap >= band.hi).mean()),  # whole photo: wrong-object guard
                          object_fraction=fraction)


class FakeInspector:
    """Scripted inspector for tests: fixed score, per-filename scores, or an error."""

    def __init__(self, score: float | Mapping[str, float] = 0.5, error: Exception | None = None):
        self._score = score
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def inspect(self, image_path: Path, product: str) -> Inspection:
        self.calls.append((str(image_path), product))
        if self._error is not None:
            raise self._error
        score = self._score if isinstance(self._score, (int, float)) else self._score[Path(image_path).name]
        return Inspection(score=float(score), defect_area=0.0)
