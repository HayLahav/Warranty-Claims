"""Deterministic, fail-closed claim policy.

No LLM and no free text is read here. Any unexpected input resolves to
ESCALATE or NEEDS_INFO — never to APPROVE or REJECT.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


# Every reason code the system can emit (policy + graph). Each needs customer-facing text in the responder.
REASON_CODES = ("defect_detected", "no_defect_found", "low_confidence", "image_out_of_distribution",
                "segmentation_failed", "unsupported_product", "inspection_failed", "missing_or_bad_image",
                "router_failed")


class Verdict(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    NEEDS_INFO = "NEEDS_INFO"


@dataclass(frozen=True)
class Band:
    lo: float
    hi: float
    max_area: float | None = None  # flagged-area fraction above which the photo is not a plausible product image
    object_range: tuple[float, float] | None = None  # plausible share of the frame covered by the segmented product


@dataclass(frozen=True)
class Inspection:
    score: float
    heatmap_path: str | None = None
    defect_area: float | None = None  # fraction of image flagged anomalous
    object_fraction: float | None = None  # fraction of image covered by the segmented product


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reasons: tuple[str, ...]


def decide(*, image_ok: bool, product: str, inspection: Inspection | None,
           inspection_error: str | None, bands: Mapping[str, Band]) -> Decision:
    if not image_ok:
        return Decision(Verdict.NEEDS_INFO, ("missing_or_bad_image",))
    if product not in bands:
        return Decision(Verdict.ESCALATE, ("unsupported_product",))
    if inspection_error is not None or inspection is None or not math.isfinite(inspection.score):
        return Decision(Verdict.ESCALATE, ("inspection_failed",))

    band = bands[product]
    if band.max_area is not None:
        if inspection.defect_area is None or not math.isfinite(inspection.defect_area):
            return Decision(Verdict.ESCALATE, ("inspection_failed",))
        if inspection.defect_area > band.max_area:
            return Decision(Verdict.ESCALATE, ("image_out_of_distribution",))
    if band.object_range is not None:
        # the score only covers the segmented product; if the cut-out is implausible, so is the score
        f = inspection.object_fraction
        if f is None or not math.isfinite(f):
            return Decision(Verdict.ESCALATE, ("inspection_failed",))
        if not band.object_range[0] <= f <= band.object_range[1]:
            return Decision(Verdict.ESCALATE, ("segmentation_failed",))
    if inspection.score >= band.hi:
        reasons = ["defect_detected"]
        if inspection.defect_area is not None:
            reasons.append(f"defect_area={inspection.defect_area * 100:.1f}%")
        return Decision(Verdict.APPROVE, tuple(reasons))
    if inspection.score < band.lo:
        return Decision(Verdict.REJECT, ("no_defect_found",))
    return Decision(Verdict.ESCALATE, ("low_confidence",))
