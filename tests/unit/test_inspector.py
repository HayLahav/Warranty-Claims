import json

import numpy as np
import pytest
from PIL import Image

from claims.inspector import FakeInspector, check_image, load_bands, save_heatmap_overlay
from claims.policy import Band


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "photo.png"
    Image.fromarray(np.full((128, 128, 3), 200, np.uint8)).save(p)
    return p


def test_check_image_accepts_valid_photo(photo):
    assert check_image(photo)


@pytest.mark.parametrize("make", [
    lambda d: None,
    lambda d: d / "missing.png",
    lambda d: (d / "junk.png").write_bytes(b"not an image") and d / "junk.png",
    lambda d: Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(d / "tiny.png") or d / "tiny.png",
])
def test_check_image_rejects_bad_input(tmp_path, make):
    assert not check_image(make(tmp_path))


def test_load_bands(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"bottle": {"lo": 1.0, "hi": 2.0}}))
    assert load_bands(p) == {"bottle": Band(1.0, 2.0)}
    p.write_text(json.dumps({"bottle": {"lo": 1.0, "hi": 2.0, "max_area": 0.3}}))
    assert load_bands(p)["bottle"].max_area == 0.3


def test_heatmap_overlay_matches_photo_size(photo, tmp_path):
    out = save_heatmap_overlay(photo, np.random.default_rng(0).random((32, 32)).astype(np.float32),
                               Band(0.2, 0.8), tmp_path / "heat.png")
    assert Image.open(out).size == (128, 128)


def test_fake_inspector_scripts():
    assert FakeInspector(0.7).inspect("a.png", "bottle").score == 0.7
    assert FakeInspector({"a.png": 0.1}).inspect("dir/a.png", "bottle").score == 0.1
    with pytest.raises(RuntimeError):
        FakeInspector(error=RuntimeError("boom")).inspect("a.png", "bottle")


class _StubScorer:
    """Map whose hot region depends on the photo's brightness, so overlays differ per photo."""
    def score(self, image_path):
        level = np.asarray(Image.open(image_path)).mean() / 255
        return 50.0, np.full((256, 256), 40.0 * level, np.float32)


def _inspector(tmp_path):
    from claims.inspector import PatchCoreInspector
    insp = PatchCoreInspector(tmp_path / "models", {"bottle": Band(lo=30, hi=40)}, tmp_path / "heat")
    insp._scorers["bottle"] = _StubScorer()
    return insp


def test_photos_with_the_same_file_name_get_separate_heatmaps(tmp_path):
    # MVTec sample cases share stems (broken_large/000.png, good/000.png); one must not overwrite the other.
    paths = []
    for sub, level in [("broken_large", 250), ("good", 40)]:
        (tmp_path / sub).mkdir()
        p = tmp_path / sub / "000.png"
        Image.fromarray(np.full((128, 128, 3), level, np.uint8)).save(p)
        paths.append(p)
    insp = _inspector(tmp_path)
    first = insp.inspect(paths[0], "bottle").heatmap_path
    first_pixels = np.asarray(Image.open(first)).copy()
    second = insp.inspect(paths[1], "bottle").heatmap_path
    assert first != second
    assert np.array_equal(np.asarray(Image.open(first)), first_pixels)  # untouched by the second claim


def test_same_photo_reuses_its_heatmap_path(tmp_path, photo):
    insp = _inspector(tmp_path)
    assert insp.inspect(photo, "bottle").heatmap_path == insp.inspect(photo, "bottle").heatmap_path


class _LintScorer:
    """Hot spot on the backdrop (like the MVTec lint), a cooler one on the product."""
    def score(self, image_path):
        amap = np.full((256, 256), 10.0, np.float32)
        amap[20, 20] = 90.0          # backdrop lint, top-left corner
        amap[128, 128] = 50.0        # defect on the product (centre)
        return 99.0, amap


def test_score_ignores_heat_on_the_backdrop(tmp_path):
    from claims.inspector import PatchCoreInspector
    p = tmp_path / "screw.png"
    img = np.full((512, 512, 3), 200, np.uint8)
    img[200:312, 230:282] = 60                       # dark product in the middle
    Image.fromarray(img).save(p)
    insp = PatchCoreInspector(tmp_path / "m", {"screw": Band(lo=30, hi=40)}, tmp_path / "heat")
    insp._scorers["screw"] = _LintScorer()
    r = insp.inspect(p, "screw")
    assert r.score == 50.0                           # not 90 (lint) and not 99 (raw whole-image score)
    assert r.object_fraction == pytest.approx(112 * 52 / 512 ** 2, rel=0.02)
    assert r.defect_area == pytest.approx(2 / 256 ** 2)   # flagged area still measured on the whole photo


def test_load_bands_reads_object_range(tmp_path):
    f = tmp_path / "t.json"
    f.write_text('{"screw": {"lo": 1, "hi": 2, "max_area": 0.1, "object_range": [0.05, 0.2]}}')
    assert load_bands(f)["screw"].object_range == (0.05, 0.2)


def test_live_score_is_quantised_like_the_calibration_maps(tmp_path):
    # thresholds are calibrated on float16 maps (reports/raw/maps_*.npz); the live score must match bit for bit
    from claims.inspector import PatchCoreInspector

    class _Scorer:
        def score(self, image_path):
            amap = np.zeros((256, 256), np.float32)
            amap[128, 128] = 35.8601          # float32 value between two float16 steps
            return 0.0, amap

    p = tmp_path / "screw.png"
    img = np.full((256, 256, 3), 200, np.uint8)
    img[100:156, 120:136] = 60
    Image.fromarray(img).save(p)
    insp = PatchCoreInspector(tmp_path / "m", {"screw": Band(lo=30, hi=40)}, tmp_path / "heat")
    insp._scorers["screw"] = _Scorer()
    assert insp.inspect(p, "screw").score == float(np.float16(35.8601))


def _magenta_pixels(path):
    px = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return int(((px[..., 0] > 180) & (px[..., 1] < 110) & (px[..., 2] > 170)).sum())


@pytest.mark.parametrize("defect_value,expect_outline", [(50.0, True), (35.0, False)])
def test_live_heatmap_outlines_the_predicted_damage(tmp_path, defect_value, expect_outline):
    from claims.inspector import PatchCoreInspector

    class _Scorer:
        def score(self, image_path):
            amap = np.full((256, 256), 10.0, np.float32)
            amap[120:136, 120:136] = defect_value      # on the product; approve threshold is 40
            return 0.0, amap

    p = tmp_path / "screw.png"
    img = np.full((512, 512, 3), 200, np.uint8)
    img[160:352, 200:312] = 60
    Image.fromarray(img).save(p)
    insp = PatchCoreInspector(tmp_path / "m", {"screw": Band(lo=30, hi=40)}, tmp_path / "heat")
    insp._scorers["screw"] = _Scorer()
    assert (_magenta_pixels(insp.inspect(p, "screw").heatmap_path) > 0) == expect_outline
