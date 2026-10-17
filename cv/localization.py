"""Localisation: how well the predicted damage mask matches the ground-truth mask.

Predicted mask = anomaly-map pixels at or above the product's approve threshold, inside the segmented product
(claims.segment.predicted_mask), the same region the live heatmap outlines in magenta. Compared with the
ground-truth mask by IoU and Dice per defective test image.

Caveat: the threshold is the decision threshold, not one tuned for segmentation, so these numbers describe the
masks the system actually shows, not the best masks PatchCore could produce.

    python -m cv.localization      # -> reports/localization.csv + printed summary
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from claims.inspector import load_bands
from cv.gallery import add_localization, build_table, load_maps, localization_summary

ROOT = Path(__file__).parent.parent


def main() -> None:
    maps = load_maps()
    bands = load_bands(ROOT / "config" / "thresholds.json")
    df = add_localization(build_table(maps), maps, bands)
    cols = ["category", "defect_type", "image", "label", "verdict", "score", "gt_area", "pred_px", "iou", "dice"]
    out = ROOT / "reports" / "localization.csv"
    df[cols].to_csv(out, index=False)
    pd.set_option("display.width", 140)
    print(localization_summary(df).round(3).to_string())
    d = df[df.label == 1]
    for cat, g in d.groupby("category"):
        print(f"{cat}: median IoU {g.iou.median():.3f}, median Dice {g.dice.median():.3f}, "
              f"empty predictions {(g.pred_px == 0).sum()}/{len(g)}, "
              f"median predicted px {g.pred_px.median():.0f} vs true px {(g.gt_area * 256 * 256).median():.0f}")
        for v, h in g.groupby("verdict"):
            print(f"   {v:8} n={len(h):3}  median IoU {h.iou.median():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
