"""Where does the anomaly map's heat sit: on the defect, elsewhere on the object, or on the background?

Per image (256x256 map resolution):
  zones      defect     = ground-truth mask, dilated by TOL_DEFECT px (same tolerance as the pointing game)
             object     = segmented product (Otsu + largest component), dilated by TOL_OBJECT px, minus defect
             background = everything else
  peak_zone  zone of the map's maximum. The saved map is Gaussian-blurred, so the image score is ~1.4 above the map
             maximum, but the two rank almost identically (Spearman 0.995 on screw): the peak's zone drives the verdict
  *_max      hottest value inside each zone; background_max >= band.lo means the background alone is enough
             to keep a good image out of auto-reject
  hot_*      share of "hot" pixels (map >= band.lo) in each zone

    python -m cv.heat_zones screw     # -> reports/heat_zones_<category>.csv + printed summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from claims.inspector import load_bands
from claims.segment import MARGIN_PX, object_mask
from cv.evaluate import load_mask
from cv.gallery import MAP_SIZE, load_maps

ROOT = Path(__file__).parent.parent
TOL_DEFECT = 6
INSET_HALF = 90  # full-resolution half-size of the zoomed background crop (gallery and Fig. 5)
TOL_OBJECT = MARGIN_PX  # 12; the blurred map spills ~8 px off the object; at 4 px, heat bleeding off the screw tip was
                 # counted as "background" (seen in crops). Counts are stable from 12 to 16 px.


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def zones(object_m: np.ndarray, defect_m: np.ndarray) -> dict[str, np.ndarray]:
    defect = _dilate(defect_m, TOL_DEFECT)
    obj = _dilate(object_m, TOL_OBJECT) & ~defect
    return {"defect": defect, "object": obj, "background": ~(defect | obj)}


def zone_stats(amap: np.ndarray, z: dict[str, np.ndarray], hot_threshold: float) -> dict:
    y, x = np.unravel_index(int(np.argmax(amap)), amap.shape)
    hot = amap >= hot_threshold
    n_hot = int(hot.sum())
    out = {"peak_zone": next(k for k, m in z.items() if m[y, x]), "hot_px": n_hot}
    for k, m in z.items():
        out[f"{k}_max"] = float(amap[m].max()) if m.any() else np.nan
        out[f"hot_{k}"] = float((hot & m).sum() / n_hot) if n_hot else 0.0
    return out


def object_mask_256(image_path: Path) -> np.ndarray:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(object_mask(gray).astype(np.uint8), (MAP_SIZE, MAP_SIZE),
                      interpolation=cv2.INTER_NEAREST) > 0


def build(category: str, data_root: Path = ROOT / "data" / "mvtec") -> pd.DataFrame:
    band = load_bands(ROOT / "config" / "thresholds.json")[category]
    oof = pd.read_csv(ROOT / "reports" / "oof_verdicts.csv").fillna({"mask": ""})
    oof = oof[oof.category == category]
    maps = load_maps()
    rows = []
    for r in oof.itertuples():
        z = zones(object_mask_256(data_root / r.image), load_mask(data_root, r.mask, MAP_SIZE))
        rows.append({"image": r.image, "defect_type": r.defect_type, "label": r.label, "score": r.score,
                     "verdict": r.verdict, **zone_stats(maps[r.image], z, band.lo)})
    return pd.DataFrame(rows)


def add_zone_columns(df: pd.DataFrame, maps: dict[str, np.ndarray],
                     data_root: Path = ROOT / "data" / "mvtec") -> pd.DataFrame:
    """Add peak_zone (defect / object / background) and background_max to a table with image + mask columns."""
    zone_of, bg_max = [], []
    for r in df.itertuples():
        z = zones(object_mask_256(data_root / r.image), load_mask(data_root, r.mask, MAP_SIZE))
        s = zone_stats(maps[r.image], z, hot_threshold=np.inf)
        zone_of.append(s["peak_zone"])
        bg_max.append(s["background_max"])
    return df.assign(peak_zone=zone_of, background_max=bg_max)


def masked_experiment(category: str = "screw") -> pd.DataFrame:
    """Re-run the 5-fold out-of-fold policy with the image score taken from the object region only.
    Both variants use the map maximum, so the comparison isolates the effect of the background."""
    from cv.evaluate import assign_folds, out_of_fold_verdicts
    maps = load_maps()
    oof = pd.read_csv(ROOT / "reports" / "oof_verdicts.csv").fillna({"mask": ""})
    base = oof[oof.category == category][["category", "defect_type", "image", "label", "mask"]].copy()
    obj = {i: _dilate(object_mask_256(ROOT / "data" / "mvtec" / i), TOL_OBJECT) for i in base.image}
    folds = assign_folds(base)
    out = base.copy()
    for name, score in [("whole", lambda i: maps[i].max()), ("masked", lambda i: maps[i][obj[i]].max())]:
        d = base.copy()
        d["score"] = [float(score(i)) for i in d.image]
        o = out_of_fold_verdicts(d, folds=folds)
        out[f"score_{name}"], out[f"verdict_{name}"] = d["score"].values, o["verdict"].values
    return out


def background_peak(amap: np.ndarray, object_m: np.ndarray) -> tuple[int, int]:
    """(x, y) of the hottest map pixel outside the dilated object, at map resolution."""
    bg = np.where(_dilate(object_m, TOL_OBJECT), -np.inf, amap)
    y, x = np.unravel_index(int(np.argmax(bg)), bg.shape)
    return int(x), int(y)


def hotspot_crop(image: np.ndarray, peak_xy: tuple[int, int], half: int = INSET_HALF) -> np.ndarray:
    """Full-resolution grayscale crop centred on a map-resolution peak, contrast-enhanced (CLAHE) so faint
    lint and dust on the backdrop become visible. Edges are padded, so the crop is always 2*half square."""
    x, y = peak_xy
    s = image.shape[0] / MAP_SIZE
    cx, cy = int(round((x + 0.5) * s)), int(round((y + 0.5) * s))
    pad = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REPLICATE)
    crop = cv2.cvtColor(pad[cy:cy + 2 * half, cx:cx + 2 * half], cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(3.0, (4, 4)).apply(crop)


def figure(masking: pd.DataFrame, out: Path, category: str = "screw") -> Path:
    """Images whose verdict depends on the background: overlay (top) and the photo zoomed on the
    background hotspot (bottom, contrast-enhanced)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cv.gallery import load_image
    band = load_bands(ROOT / "config" / "thresholds.json")[category]
    maps = load_maps()
    changed = masking[masking.verdict_whole != masking.verdict_masked]
    # keep the cases the background explains: defects approved via background, good images escalated via background
    keep = changed[((changed.label == 1) & (changed.verdict_whole == "APPROVE")) |
                   ((changed.label == 0) & (changed.verdict_whole == "ESCALATE"))]
    fig, axes = plt.subplots(2, len(keep), figsize=(2.3 * len(keep), 5.4), facecolor="#fcfcfb")
    for col, r in zip(axes.T, keep.itertuples()):
        img = load_image(r.image)
        amap = maps[r.image]
        obj = object_mask_256(ROOT / "data" / "mvtec" / r.image)
        x, y = background_peak(amap, obj)
        heat = cv2.resize(amap, img.shape[1::-1])
        col[0].imshow(img, cmap="gray")
        col[0].imshow(np.ma.masked_less(heat, 0.75 * band.lo), cmap="jet", alpha=0.45,
                      vmin=0.75 * band.lo, vmax=band.hi)
        s = img.shape[0] / MAP_SIZE
        col[0].add_patch(plt.Circle((x * s, y * s), 60, fill=False, ec="#ffe600", lw=2))
        col[1].imshow(hotspot_crop(img, (x, y)), cmap="gray")
        name = r.image.split("/test/")[1]
        col[0].set_title(f"{name}\n{r.verdict_whole} → {r.verdict_masked}", fontsize=8)
        col[1].set_title("background hotspot (zoomed)", fontsize=7, color="#555")
        for a in col:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{category}: verdicts that change when the background is masked out", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(h_pad=2.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def summarize(df: pd.DataFrame, band_lo: float, band_hi: float) -> None:
    print("zone of the map maximum (drives the score):")
    print(pd.crosstab([df.label.map({0: "good", 1: "defective"}), df.verdict], df.peak_zone, margins=True).to_string())
    for name, d in [("good", df[df.label == 0]), ("defective", df[df.label == 1])]:
        print(f"\n{name} (n={len(d)}):")
        print(f"  background hotter than reject threshold ({band_lo:.1f}): {(d.background_max >= band_lo).sum()}")
        print(f"  background at/above approve threshold ({band_hi:.1f}):   {(d.background_max >= band_hi).sum()}")
        print(f"  median share of hot pixels  background {d.hot_background.median():.0%}  "
              f"object {d.hot_object.median():.0%}  defect {d.hot_defect.median():.0%}")


def main() -> None:
    category = sys.argv[1] if len(sys.argv) > 1 else "screw"
    band = load_bands(ROOT / "config" / "thresholds.json")[category]
    df = build(category)
    out = ROOT / "reports" / f"heat_zones_{category}.csv"
    df.to_csv(out, index=False)
    summarize(df, band.lo, band.hi)
    print(f"\nwrote {out}")
    m = masked_experiment(category)
    m.to_csv(ROOT / "reports" / f"background_masking_{category}.csv", index=False)
    print("\nbackground masked (out-of-fold verdicts, whole image -> object only):")
    print(pd.crosstab(m.verdict_whole, m.verdict_masked).to_string())
    changed = m[m.verdict_whole != m.verdict_masked]
    print(changed[["image", "verdict_whole", "verdict_masked"]].to_string(index=False))
    print("wrote", figure(m, ROOT / "reports" / "figures" / f"07_{category}_background_contamination.png", category))


if __name__ == "__main__":
    main()
