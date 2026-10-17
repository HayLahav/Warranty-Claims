"""Evaluate PatchCore scores and the decision policy.

Inputs  (from cv/fit_patchcore.ipynb):  reports/raw/scores.csv, reports/raw/maps_<category>.npz
Outputs: data/folds.json            stratified 5-fold assignment (committed)
         config/thresholds.json     deployed band per category, calibrated on the full test set
         reports/oof_verdicts.csv   one out-of-fold verdict per test image
         reports/metrics.json       AUROC (full set, bootstrap CI) + cross-validated policy metrics (Wilson CI)

Scores: the image score is the anomaly map's maximum inside the segmented product (claims.segment), the same
score the live inspector uses, so backdrop lint and dust cannot decide a claim. The raw PatchCore score is kept
as score_patchcore for comparison. The plausible product coverage (object_range) is calibrated on training images.

Threshold-free metrics (AUROC) use every test image directly. Anything that
depends on the band is computed out-of-fold: each image is judged by a band
calibrated on the other k-1 folds, through the same claims.policy.decide the
app uses.

Usage: python -m cv.evaluate
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from claims.policy import Inspection, decide
from claims.segment import calibrate_object_range, masked_score, product_region
from cv.calibrate import calibrate_band, calibrate_max_area
from cv.folds import make_folds

K_FOLDS = 5
SEED = 0
VERDICTS = ("APPROVE", "REJECT", "ESCALATE")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval; well-behaved at 0/n and n/n, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(successes: int, n: int) -> dict:
    lo, hi = wilson_interval(successes, n)
    return {"value": successes / n if n else None, "k": successes, "n": n, "ci95": [lo, hi]}


def assign_folds(df: pd.DataFrame, k: int = K_FOLDS, seed: int = SEED) -> dict[str, int]:
    """Folds per category (stratified by defect type), so one category's folds — and
    therefore its metrics — don't change when another category is added or removed."""
    folds: dict[str, int] = {}
    for _, g in df.groupby("category"):
        folds.update(make_folds(zip(g["image"], g["defect_type"]), k=k, seed=seed))
    return folds


def _rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def segment_scores(df: pd.DataFrame, maps: dict[str, np.ndarray], data_root: Path) -> pd.DataFrame:
    """score -> product-only map maximum; raw PatchCore score kept as score_patchcore; adds object_fraction."""
    out = df.copy()
    out["score_patchcore"] = out["score"]
    scores, fractions = [], []
    for row in out.itertuples():
        amap = maps[row.image].astype(np.float32)
        region, fraction = product_region(_rgb(data_root / row.image), size=amap.shape[0])
        s = masked_score(amap, region)
        scores.append(s if math.isfinite(s) else float(amap.max()))
        fractions.append(fraction)
    out["score"], out["object_fraction"] = scores, fractions
    return out


def object_ranges(data_root: Path, categories) -> dict[str, tuple[float, float]]:
    """Plausible product coverage per category, from the training (defect-free) images: no test data used."""
    ranges = {}
    for cat in categories:
        files = sorted((data_root / cat / "train" / "good").glob("*.png"))
        if not files:
            raise FileNotFoundError(f"no training images for {cat} under {data_root}")
        ranges[cat] = calibrate_object_range([product_region(_rgb(f))[1] for f in files])
    return ranges


def out_of_fold_verdicts(df: pd.DataFrame, k: int = K_FOLDS, seed: int = SEED,
                         folds: dict[str, int] | None = None,
                         ranges: dict[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    folds = folds or assign_folds(df, k, seed)
    out = df.copy()
    out["fold"] = out["image"].map(folds)
    out["band_lo"] = np.nan
    out["band_hi"] = np.nan
    out["verdict"] = ""
    for category, cat_df in out.groupby("category"):
        for fold in range(k):
            held = cat_df[cat_df["fold"] == fold]
            train = cat_df[cat_df["fold"] != fold]
            band = calibrate_band(train["score"].tolist(), train["label"].tolist())
            if ranges is not None:
                band = replace(band, object_range=ranges[category])
            for idx, row in held.iterrows():
                fraction = row["object_fraction"] if "object_fraction" in held else None
                d = decide(image_ok=True, product=category,
                           inspection=Inspection(score=row["score"], object_fraction=fraction),
                           inspection_error=None, bands={category: band})
                out.at[idx, "verdict"] = d.verdict.value
                out.at[idx, "band_lo"] = band.lo
                out.at[idx, "band_hi"] = band.hi
    return out


def summarize(oof: pd.DataFrame) -> dict:
    result = {}
    for category, g in oof.groupby("category"):
        good, bad = g[g.label == 0], g[g.label == 1]
        approved, rejected = g[g.verdict == "APPROVE"], g[g.verdict == "REJECT"]
        by_type = {}
        for dt, t in g.groupby("defect_type"):
            counts = t["verdict"].value_counts()
            by_type[dt] = {"n": len(t), **{v: int(counts.get(v, 0)) for v in VERDICTS},
                           "escalation_rate": _rate(int(counts.get("ESCALATE", 0)), len(t)),
                           "wrong_rate": _rate(int(counts.get("REJECT" if dt != "good" else "APPROVE", 0)), len(t))}
        result[category] = {
            "n": len(g),
            "escalation_rate": _rate(int((g.verdict == "ESCALATE").sum()), len(g)),
            "approve_precision": _rate(int((approved.label == 1).sum()), len(approved)),
            "reject_precision": _rate(int((rejected.label == 0).sum()), len(rejected)),
            "false_approve_rate": _rate(int((good.verdict == "APPROVE").sum()), len(good)),
            "missed_defect_rate": _rate(int((bad.verdict == "REJECT").sum()), len(bad)),
            "by_defect_type": by_type,
        }
    return result


def auroc_with_ci(labels: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(labels, scores))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() != labels[idx].max():
            boots.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"value": point, "ci95": [float(lo), float(hi)], "n": int(len(labels))}


def load_mask(data_root: Path, rel: str, size: int) -> np.ndarray:
    from PIL import Image
    if not rel:
        return np.zeros((size, size), dtype=bool)
    m = Image.open(data_root / rel).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(m) > 127


def pixel_auroc(df: pd.DataFrame, maps: dict[str, np.ndarray], data_root: Path) -> float:
    from sklearn.metrics import roc_auc_score
    ys, ss = [], []
    for row in df.itertuples():
        amap = maps[row.image].astype(np.float32)
        ys.append(load_mask(data_root, row.mask, amap.shape[0]).ravel())
        ss.append(amap.ravel())
    return float(roc_auc_score(np.concatenate(ys), np.concatenate(ss)))


def main(raw_dir=Path("reports/raw"), data_root=Path("data/mvtec"), reports=Path("reports"),
         folds_path=Path("data/folds.json"), thresholds_path=Path("config/thresholds.json")) -> dict:
    df = pd.read_csv(raw_dir / "scores.csv", keep_default_na=False)
    if not np.isfinite(df["score"]).all():
        raise ValueError("non-finite scores in scores.csv")
    all_maps = {}
    for category in df["category"].unique():
        all_maps.update(dict(np.load(raw_dir / f"maps_{category}.npz")))
    df = segment_scores(df, all_maps, data_root)
    ranges = object_ranges(data_root, sorted(df["category"].unique()))

    folds = assign_folds(df)
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    folds_path.write_text(json.dumps({"k": K_FOLDS, "seed": SEED, "folds": folds}, indent=1, sort_keys=True))

    oof = out_of_fold_verdicts(df, folds=folds, ranges=ranges)
    reports.mkdir(parents=True, exist_ok=True)
    oof.to_csv(reports / "oof_verdicts.csv", index=False)

    metrics = {"k_folds": K_FOLDS, "seed": SEED, "scoring": "anomaly-map max inside the segmented product",
               "policy_cv": summarize(oof), "auroc": {}}
    thresholds = {}
    for category, g in df.groupby("category"):
        maps = {img: all_maps[img] for img in g["image"]}
        metrics["auroc"][category] = {
            "image": auroc_with_ci(g["label"].to_numpy(), g["score"].to_numpy()),
            "image_patchcore": auroc_with_ci(g["label"].to_numpy(), g["score_patchcore"].to_numpy()),
            "pixel": pixel_auroc(g, maps, data_root),
        }
        band = calibrate_band(g["score"].tolist(), g["label"].tolist())
        areas = [float((maps[img].astype(np.float32) >= band.hi).mean()) for img in g.loc[g.label == 1, "image"]]
        thresholds[category] = asdict(replace(band, max_area=calibrate_max_area(areas),
                                              object_range=ranges[category]))

    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(thresholds, indent=2))
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _print(metrics: dict) -> None:
    for cat, a in metrics["auroc"].items():
        p = metrics["policy_cv"][cat]

        def fmt(r):
            return "n/a" if r["value"] is None else f"{r['value']:.1%} [{r['ci95'][0]:.0%}-{r['ci95'][1]:.0%}] ({r['k']}/{r['n']})"
        print(f"\n== {cat}  (n={p['n']})")
        print(f"  image AUROC      {a['image']['value']:.4f}  CI95 [{a['image']['ci95'][0]:.3f}, {a['image']['ci95'][1]:.3f}]")
        print(f"  pixel AUROC      {a['pixel']:.4f}")
        for key in ("escalation_rate", "approve_precision", "reject_precision", "false_approve_rate", "missed_defect_rate"):
            print(f"  {key:<20}{fmt(p[key])}")


if __name__ == "__main__":
    _print(main())
