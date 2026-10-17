"""Uncertainty-band calibration for anomaly scores.

The band is chosen so that, on the calibration data:
  - every score >= band.hi is defective  (auto-APPROVE is 100% precise)
  - every score <  band.lo is good       (auto-REJECT is 100% precise)
Scores in [lo, hi) are where the classes overlap and get ESCALATED.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from claims.policy import Band


def calibrate_band(scores: Sequence[float], labels: Sequence[int]) -> Band:
    """labels: 1 = defective, 0 = good."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite")
    good = sorted(s for s, l in zip(scores, labels) if l == 0)
    bad = sorted(s for s, l in zip(scores, labels) if l == 1)
    if not good or not bad:
        raise ValueError("calibration needs at least one good and one defective sample")

    max_good, min_bad = good[-1], bad[0]

    # hi: just above every good score — midpoint to the next defect score if there is one.
    bad_above = [s for s in bad if s > max_good]
    hi = (max_good + bad_above[0]) / 2 if bad_above else math.nextafter(max_good, math.inf)

    # lo: at or below every defect score — midpoint to the previous good score if there is one.
    good_below = [s for s in good if s < min_bad]
    lo = (good_below[-1] + min_bad) / 2 if good_below else min_bad

    return Band(lo=lo, hi=hi)


AREA_MARGIN = 1.5
AREA_FLOOR = 0.10


def calibrate_max_area(defect_areas: Sequence[float]) -> float:
    """Out-of-distribution guard: a photo flagging far more area than any real defect did
    (e.g. a screw photographed for a bottle claim flags ~90%) is escalated, never approved.
    Margin and floor keep real-but-larger defects from tripping it; a trip only escalates."""
    return min(1.0, max(AREA_MARGIN * max(defect_areas), AREA_FLOOR))
