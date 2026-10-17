import numpy as np
import pytest

from claims.segment import (MAP_SIZE, MARGIN_PX, calibrate_object_range, masked_score, object_mask,
                            product_region)


def photo(size=512, box=(200, 50, 300, 450), bg=200, fg=60):
    """Light backdrop with a dark rectangular 'screw'."""
    img = np.full((size, size, 3), bg, np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = fg
    return img


def test_object_mask_finds_the_dark_product_and_drops_specks():
    img = photo()
    img[10:14, 10:14] = 60                     # dust speck: separate, smaller component
    m = object_mask(img[..., 0])
    assert m[250, 250] and not m[12, 12] and not m[5, 500]


def test_product_region_is_map_sized_with_a_margin_and_reports_raw_coverage():
    region, fraction = product_region(photo())
    assert region.shape == (MAP_SIZE, MAP_SIZE)
    assert fraction == pytest.approx(100 * 400 / 512 ** 2, rel=0.02)   # undilated coverage
    # box spans x 100..150 at map resolution; the margin extends it
    assert region[128, 150 + MARGIN_PX - 1] and not region[128, 150 + MARGIN_PX + 2]


def test_masked_score_ignores_heat_on_the_background():
    region, _ = product_region(photo())
    amap = np.zeros((MAP_SIZE, MAP_SIZE), np.float32)
    amap[20, 20] = 50.0                        # lint on the backdrop
    amap[128, 125] = 30.0                      # defect on the product
    assert masked_score(amap, region) == 30.0


def test_masked_score_of_an_empty_region_is_not_finite():
    assert not np.isfinite(masked_score(np.ones((MAP_SIZE, MAP_SIZE), np.float32),
                                        np.zeros((MAP_SIZE, MAP_SIZE), bool)))


def test_calibrated_range_widens_the_observed_coverage():
    lo, hi = calibrate_object_range([0.10, 0.12, 0.15])
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.225)
    assert calibrate_object_range([0.8, 0.9])[1] == 1.0


def test_calibrated_range_rejects_bad_input():
    with pytest.raises(ValueError):
        calibrate_object_range([])
    with pytest.raises(ValueError):
        calibrate_object_range([0.1, float("nan")])


def test_margin_scales_with_map_size():
    region_small, _ = product_region(photo(), size=64)        # box spans x 25..37 at 64 px
    assert region_small[32, 37 + 2] and not region_small[32, 37 + 6]   # margin 12 * 64/256 = 3 px


# ── predicted damage mask and its overlap with the ground truth ──────────────────────────────────
def test_predicted_mask_is_heat_above_threshold_inside_the_product_only():
    from claims.segment import predicted_mask
    region = np.zeros((MAP_SIZE, MAP_SIZE), bool)
    region[50:150, 50:150] = True
    amap = np.zeros((MAP_SIZE, MAP_SIZE), np.float32)
    amap[60:70, 60:70] = 40.0        # on the product, above threshold
    amap[5:15, 5:15] = 99.0          # lint on the backdrop
    amap[100:110, 100:110] = 20.0    # on the product, below threshold
    pred = predicted_mask(amap, region, threshold=35.0)
    assert pred.sum() == 100 and pred[65, 65] and not pred[10, 10] and not pred[105, 105]


def test_overlap_iou_and_dice():
    from claims.segment import overlap
    a = np.zeros((10, 10), bool); a[0:4, 0:5] = True     # 20 px
    b = np.zeros((10, 10), bool); b[2:6, 0:5] = True     # 20 px, 10 shared
    iou, dice = overlap(a, b)
    assert iou == pytest.approx(10 / 30) and dice == pytest.approx(20 / 40)


def test_overlap_of_empty_prediction_is_zero_and_of_two_empties_is_undefined():
    from claims.segment import overlap
    gt = np.zeros((10, 10), bool); gt[0:2, 0:2] = True
    assert overlap(np.zeros((10, 10), bool), gt) == (0.0, 0.0)
    iou, dice = overlap(np.zeros((10, 10), bool), np.zeros((10, 10), bool))
    assert np.isnan(iou) and np.isnan(dice)
