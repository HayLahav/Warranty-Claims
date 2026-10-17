import numpy as np
import pytest

from claims.policy import Band
from cv.gallery import peak_hits_defect, peak_location, pointing_summary, render


def blob(cx, cy, size=64):
    yy, xx = np.mgrid[:size, :size]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 20.0).astype(np.float32)


def square_mask(x0, y0, x1, y1, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_peak_on_defect_is_a_hit():
    assert peak_hits_defect(blob(20, 20), square_mask(15, 15, 25, 25)) is True


def test_peak_elsewhere_is_a_miss():
    assert peak_hits_defect(blob(50, 50), square_mask(5, 5, 12, 12)) is False


def test_tolerance_counts_near_misses_only():
    mask = square_mask(10, 10, 20, 20)
    assert peak_hits_defect(blob(24, 15), mask, tolerance_px=6) is True     # 4 px outside
    assert peak_hits_defect(blob(34, 15), mask, tolerance_px=6) is False    # 14 px outside


def test_good_images_have_no_verdict():
    assert peak_hits_defect(blob(20, 20), np.zeros((64, 64), bool)) is None


def test_peak_location_is_x_y():
    assert peak_location(blob(40, 10)) == (40, 10)


@pytest.mark.parametrize("scale", ["fixed", "per-image"])
def test_render_keeps_photo_size_and_outlines_defect(scale):
    photo = np.full((300, 300, 3), 128, np.uint8)
    out = render(photo, blob(32, 32) * 50, square_mask(5, 5, 15, 15), Band(lo=10, hi=40), scale)
    assert out.shape == photo.shape and out.dtype == np.uint8
    assert (out == 255).all(axis=-1).any()  # white mask outline drawn


def test_per_image_scale_shows_a_map_that_is_entirely_below_threshold():
    photo = np.full((64, 64, 3), 128, np.uint8)
    amap = blob(32, 32) * 5  # far below band.lo
    no_mask = np.zeros((64, 64), bool)
    fixed = render(photo, amap, no_mask, Band(lo=30, hi=40), "fixed")
    stretched = render(photo, amap, no_mask, Band(lo=30, hi=40), "per-image")
    # The marker is a ring, so the map's centre pixel is not drawn over.
    assert (fixed[32, 32] == 128).all()                     # fixed scale: below threshold -> transparent
    assert np.abs(stretched[32, 32].astype(int) - 128).sum() > 60   # per-image scale: peak is coloured


def test_pointing_summary_counts_defective_images_only():
    import pandas as pd
    df = pd.DataFrame({"category": ["screw"] * 3, "defect_type": ["good", "scratch", "scratch"],
                       "label": [0, 1, 1], "peak_hits_defect": [None, True, False]})
    s = pointing_summary(df)
    assert s.loc[("screw", "scratch")].tolist() == [2, 1, 0.5]
    assert ("screw", "good") not in s.index


def test_render_can_hide_heat_outside_the_product_and_outline_the_prediction():
    photo = np.full((64, 64, 3), 128, np.uint8)
    amap = np.zeros((64, 64), np.float32)
    amap[5:10, 5:10] = 50          # backdrop lint
    amap[30:36, 30:36] = 50        # damage on the product
    region = np.zeros((64, 64), bool)
    region[20:50, 20:50] = True
    no_mask = np.zeros((64, 64), bool)
    band = Band(lo=10, hi=40)
    shown = render(photo, amap, no_mask, band, "fixed", region=region, show_outside=True)
    hidden = render(photo, amap, no_mask, band, "fixed", region=region, show_outside=False)
    assert not (shown[7, 7] == 128).all() and (hidden[7, 7] == 128).all()     # lint heat hidden
    pred = (amap >= band.hi) & region
    out = render(photo, amap, no_mask, band, "fixed", region=region, pred=pred)
    magenta = (out[..., 0] > 180) & (out[..., 1] < 110) & (out[..., 2] > 170)
    assert magenta.any()


def test_localization_summary_reports_medians_and_empty_predictions():
    import pandas as pd
    from cv.gallery import localization_summary
    df = pd.DataFrame({"category": ["screw"] * 4, "defect_type": ["good", "scratch", "scratch", "scratch"],
                       "label": [0, 1, 1, 1], "iou": [np.nan, 0.5, 0.1, 0.0], "dice": [np.nan, 0.6, 0.2, 0.0],
                       "pred_px": [0, 30, 5, 0]})
    s = localization_summary(df).loc[("screw", "scratch")]
    assert s["defective images"] == 3 and s["median IoU"] == 0.1 and s["empty predictions"] == 1
    assert ("screw", "good") not in localization_summary(df).index
