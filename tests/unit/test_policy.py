import math

import pytest

from claims.policy import Band, Inspection, Verdict, decide

BANDS = {"bottle": Band(lo=0.4, hi=0.6), "screw": Band(lo=0.3, hi=0.8)}


def run(score=0.5, *, image_ok=True, product="bottle", error=None, inspection=True):
    insp = Inspection(score=score) if inspection else None
    return decide(image_ok=image_ok, product=product, inspection=insp,
                  inspection_error=error, bands=BANDS)


@pytest.mark.parametrize("score,verdict,reason", [
    (0.9, Verdict.APPROVE, "defect_detected"),
    (0.6, Verdict.APPROVE, "defect_detected"),     # exactly band_hi -> approve
    (0.599, Verdict.ESCALATE, "low_confidence"),
    (0.4, Verdict.ESCALATE, "low_confidence"),     # exactly band_lo -> not reject
    (0.399, Verdict.REJECT, "no_defect_found"),
    (0.0, Verdict.REJECT, "no_defect_found"),
])
def test_score_bands(score, verdict, reason):
    d = run(score)
    assert d.verdict == verdict
    assert reason in d.reasons


def test_bands_are_per_product():
    assert run(0.7, product="bottle").verdict == Verdict.APPROVE
    assert run(0.7, product="screw").verdict == Verdict.ESCALATE


def test_missing_image_needs_info_even_with_high_score():
    d = run(0.99, image_ok=False)
    assert d.verdict == Verdict.NEEDS_INFO
    assert d.reasons == ("missing_or_bad_image",)


def test_unsupported_product_escalates_even_with_high_score():
    d = run(0.99, product="toaster")
    assert d.verdict == Verdict.ESCALATE
    assert d.reasons == ("unsupported_product",)


def test_inspector_error_escalates():
    d = run(inspection=False, error="RuntimeError: boom")
    assert d.verdict == Verdict.ESCALATE
    assert d.reasons == ("inspection_failed",)


def test_missing_inspection_without_error_still_escalates():
    assert run(inspection=False).verdict == Verdict.ESCALATE


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_non_finite_score_fails_closed(score):
    d = run(score)
    assert d.verdict == Verdict.ESCALATE
    assert d.reasons == ("inspection_failed",)


def test_rule_order_image_before_product():
    assert run(image_ok=False, product="toaster").verdict == Verdict.NEEDS_INFO


GUARDED = {"bottle": Band(lo=0.4, hi=0.6, max_area=0.5)}


@pytest.mark.parametrize("score", [0.9, 0.5, 0.1])
def test_area_above_max_area_escalates_as_out_of_distribution(score):
    d = decide(image_ok=True, product="bottle", inspection=Inspection(score=score, defect_area=0.9),
               inspection_error=None, bands=GUARDED)
    assert d.verdict == Verdict.ESCALATE
    assert d.reasons == ("image_out_of_distribution",)


def test_area_at_max_area_is_still_decided():
    d = decide(image_ok=True, product="bottle", inspection=Inspection(score=0.9, defect_area=0.5),
               inspection_error=None, bands=GUARDED)
    assert d.verdict == Verdict.APPROVE


def test_missing_area_with_guard_fails_closed():
    d = decide(image_ok=True, product="bottle", inspection=Inspection(score=0.9),
               inspection_error=None, bands=GUARDED)
    assert d.verdict == Verdict.ESCALATE
    assert d.reasons == ("inspection_failed",)


def test_defect_area_is_reported_when_approving():
    d = decide(image_ok=True, product="bottle", inspection=Inspection(score=0.9, defect_area=0.034),
               inspection_error=None, bands=BANDS)
    assert any(r.startswith("defect_area=3.4%") for r in d.reasons)


# ── product segmentation coverage (score is only taken from the product region) ──────────────
SEG = {"screw": Band(lo=0.3, hi=0.8, max_area=0.5, object_range=(0.05, 0.25))}


def run_seg(score=0.9, fraction=0.13, area=0.01):
    return decide(image_ok=True, product="screw", inspection=Inspection(score=score, defect_area=area,
                                                                        object_fraction=fraction),
                  inspection_error=None, bands=SEG)


@pytest.mark.parametrize("fraction", [0.0, 0.049, 0.251, 0.9])
def test_implausible_product_coverage_escalates_even_with_high_score(fraction):
    d = run_seg(score=0.99, fraction=fraction)
    assert (d.verdict, d.reasons) == (Verdict.ESCALATE, ("segmentation_failed",))


@pytest.mark.parametrize("fraction", [0.05, 0.13, 0.25])
def test_plausible_coverage_is_decided_on_the_score(fraction):
    assert run_seg(score=0.99, fraction=fraction).verdict == Verdict.APPROVE
    assert run_seg(score=0.1, fraction=fraction).verdict == Verdict.REJECT


@pytest.mark.parametrize("fraction", [None, math.nan])
def test_missing_coverage_with_range_set_fails_closed(fraction):
    assert run_seg(fraction=fraction).reasons == ("inspection_failed",)


def test_wrong_object_reason_wins_over_segmentation():
    # a hazelnut photo trips the whole-image area guard first; that reason is the informative one
    d = run_seg(score=0.99, fraction=0.9, area=0.95)
    assert d.reasons == ("image_out_of_distribution",)


def test_no_range_means_no_coverage_check():
    d = decide(image_ok=True, product="bottle", inspection=Inspection(score=0.9, object_fraction=0.0),
               inspection_error=None, bands=BANDS)
    assert d.verdict == Verdict.APPROVE


def test_segmentation_failed_is_a_registered_reason():
    from claims.policy import REASON_CODES
    assert "segmentation_failed" in REASON_CODES
