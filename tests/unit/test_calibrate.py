import math

import pytest

from cv.calibrate import calibrate_band


def test_separable_scores_give_zero_width_band_between_classes():
    band = calibrate_band(scores=[0.1, 0.2, 0.8, 0.9], labels=[0, 0, 1, 1])
    assert band.lo == band.hi == pytest.approx(0.5)


def test_overlapping_scores_bracket_the_overlap():
    # goods: 0.1, 0.3, 0.6   defects: 0.4, 0.7, 0.9
    band = calibrate_band(scores=[0.1, 0.3, 0.6, 0.4, 0.7, 0.9], labels=[0, 0, 0, 1, 1, 1])
    assert band.lo == pytest.approx((0.3 + 0.4) / 2)  # below every defect
    assert band.hi == pytest.approx((0.6 + 0.7) / 2)  # above every good


def test_band_is_100_percent_precise_on_its_own_data():
    scores = [0.05, 0.22, 0.31, 0.47, 0.52, 0.18, 0.44, 0.61, 0.73, 0.29]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    band = calibrate_band(scores, labels)
    approved = [l for s, l in zip(scores, labels) if s >= band.hi]
    rejected = [l for s, l in zip(scores, labels) if s < band.lo]
    assert approved and all(l == 1 for l in approved)
    assert rejected and all(l == 0 for l in rejected)


def test_tied_good_and_defect_score_is_escalated():
    band = calibrate_band(scores=[0.1, 0.5, 0.5, 0.9], labels=[0, 0, 1, 1])
    assert band.lo < 0.5 < band.hi


def test_no_defect_above_all_goods_never_approves_a_good():
    band = calibrate_band(scores=[0.2, 0.9, 0.3, 0.5], labels=[0, 0, 1, 1])
    assert band.hi > 0.9


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1]])
def test_single_class_is_rejected(labels):
    with pytest.raises(ValueError):
        calibrate_band([0.1, 0.2, 0.3], labels)


def test_max_area_has_margin_and_floor():
    from cv.calibrate import calibrate_max_area
    assert calibrate_max_area([0.0, 0.2, 0.4]) == pytest.approx(0.6)
    assert calibrate_max_area([0.0, 0.01]) == pytest.approx(0.10)
    assert calibrate_max_area([0.9]) == pytest.approx(1.0)


def test_nan_score_is_rejected():
    with pytest.raises(ValueError):
        calibrate_band([0.1, math.nan, 0.3, 0.4], [0, 0, 1, 1])
