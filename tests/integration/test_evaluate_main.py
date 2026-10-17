import json

import numpy as np
from PIL import Image

from cv.evaluate import main

SIZE = 16      # anomaly-map resolution in this synthetic set
PHOTO = 64     # photo resolution


def product_photo(path):
    """Light backdrop, dark product square covering map cells 2..10 (photo 8..40)."""
    img = np.full((PHOTO, PHOTO, 3), 200, np.uint8)
    img[8:40, 8:40] = 60
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def test_main_scores_only_the_segmented_product(tmp_path):
    raw, data, reports = tmp_path / "raw", tmp_path / "mvtec", tmp_path / "reports"
    raw.mkdir()
    for i in range(4):
        product_photo(data / f"bottle/train/good/{i:03d}.png")
    rng = np.random.default_rng(0)
    rows, maps = [], {}
    for i in range(30):
        label = int(i >= 10)
        dt = "good" if not label else ("crack" if i % 2 else "dent")
        image = f"bottle/test/{dt}/{i:03d}.png"
        product_photo(data / image)
        mask = ""
        amap = rng.random((SIZE, SIZE)).astype(np.float16) * 0.3
        if label:
            mask = f"bottle/ground_truth/{dt}/{i:03d}_mask.png"
            m = np.zeros((SIZE, SIZE), np.uint8)
            m[4:8, 4:8] = 255
            (data / mask).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(m).save(data / mask)
            amap[4:8, 4:8] += 1.0                      # defect on the product
        else:
            amap[14, 14] += 2.0                        # backdrop lint, far from the product: hotter than any defect
        maps[image] = amap
        rows.append(f"bottle,{dt},{image},{label},{mask},{float(amap.max()) + i * 1e-4}")
    (raw / "scores.csv").write_text("category,defect_type,image,label,mask,score\n" + "\n".join(rows))
    np.savez_compressed(raw / "maps_bottle.npz", **maps)

    metrics = main(raw_dir=raw, data_root=data, reports=reports,
                   folds_path=tmp_path / "folds.json", thresholds_path=tmp_path / "thresholds.json")

    # the raw (whole-image) scores put every good image above every defect; the product-only score separates them
    assert metrics["auroc"]["bottle"]["image"]["value"] == 1.0
    assert metrics["auroc"]["bottle"]["image_patchcore"]["value"] == 0.0
    assert metrics["auroc"]["bottle"]["pixel"] < 1.0                     # pixel AUROC still sees the whole map
    oof = (reports / "oof_verdicts.csv").read_text()
    assert "score_patchcore" in oof and "object_fraction" in oof
    assert json.loads((reports / "metrics.json").read_text())["k_folds"] == 5
    band = json.loads((tmp_path / "thresholds.json").read_text())["bottle"]
    assert band["lo"] <= band["hi"]
    assert 0.10 <= band["max_area"] <= 1.0
    lo, hi = band["object_range"]
    assert lo < (32 / 64) ** 2 < hi                                       # training coverage 25% inside the range
    assert len(json.loads((tmp_path / "folds.json").read_text())["folds"]) == 30
