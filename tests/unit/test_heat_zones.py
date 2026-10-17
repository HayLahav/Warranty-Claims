import numpy as np

from cv.heat_zones import TOL_OBJECT, background_peak, zone_stats, zones


def rect(x0, y0, x1, y1, size=128):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


OBJ = rect(40, 20, 60, 110)       # a vertical "screw"
DEFECT = rect(45, 30, 55, 40)     # on the screw


def test_zones_partition_the_image():
    z = zones(OBJ, DEFECT)
    total = sum(m.astype(int) for m in z.values())
    assert (total == 1).all()                 # every pixel in exactly one zone
    assert z["defect"][35, 50] and z["object"][90, 50] and z["background"][5, 5]


def test_object_zone_includes_a_margin_for_blur():
    z = zones(OBJ, np.zeros_like(OBJ))
    assert z["object"][60, 60 + TOL_OBJECT - 1]   # just off the edge still counts as object
    assert z["background"][60, 60 + TOL_OBJECT + 2]


def test_peak_zone_and_hot_shares():
    amap = np.zeros((128, 128), np.float32)
    amap[100:104, 100:104] = 10                  # background lint, hottest
    amap[33:37, 48:52] = 5                       # defect
    s = zone_stats(amap, zones(OBJ, DEFECT), hot_threshold=4)
    assert s["peak_zone"] == "background"
    assert s["background_max"] == 10 and s["defect_max"] == 5
    assert s["hot_background"] == 0.5 and s["hot_defect"] == 0.5


def test_good_image_has_no_defect_zone():
    s = zone_stats(np.ones((128, 128), np.float32), zones(OBJ, np.zeros_like(OBJ)), hot_threshold=2)
    assert np.isnan(s["defect_max"]) and s["hot_px"] == 0


def test_background_peak_ignores_the_object():
    amap = np.zeros((128, 128), np.float32)
    amap[50, 50] = 99                            # on the screw: must be ignored
    amap[10, 120] = 3
    assert background_peak(amap, OBJ) == (120, 10)


def test_hotspot_crop_is_centred_on_the_peak_at_full_resolution():
    from cv.heat_zones import hotspot_crop
    img = np.full((1024, 1024, 3), 200, np.uint8)
    img[500:520, 300:304] = 40                       # a thin dark "fibre" at full resolution
    crop = hotspot_crop(img, (302 * 256 // 1024, 510 * 256 // 1024), half=90)
    assert crop.shape == (180, 180) and crop.dtype == np.uint8
    centre = crop[80:100, 80:100]
    assert centre.min() < crop[:20, :20].mean() - 50  # the fibre is in the middle and stands out


def test_hotspot_crop_near_the_border_keeps_its_size():
    from cv.heat_zones import hotspot_crop
    img = np.full((900, 900, 3), 200, np.uint8)
    assert hotspot_crop(img, (0, 255), half=90).shape == (180, 180)
