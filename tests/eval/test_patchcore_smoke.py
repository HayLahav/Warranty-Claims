"""Fit -> export -> score on a tiny MVTec subset, on CPU.

Proves the anomalib API calls in cv/patchcore.py work end to end before the
full fit runs on Colab. Run with:  pytest -m eval tests/eval/test_patchcore_smoke.py
"""
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from cv.patchcore import MAP_SIZE, fit_and_export, model_path, score_test_set

DATA = Path("data/mvtec")
pytestmark = pytest.mark.eval


@pytest.fixture
def tiny_bottle(tmp_path: Path) -> Path:
    src = DATA / "bottle"
    if not src.is_dir():
        pytest.skip("data/mvtec/bottle missing — run python -m cv.download_mvtec")
    dst = tmp_path / "mvtec" / "bottle"
    for sub, n in [("train/good", 8), ("test/good", 2), ("test/broken_large", 2)]:
        (dst / sub).mkdir(parents=True)
        for f in sorted((src / sub).glob("*.png"))[:n]:
            shutil.copy(f, dst / sub / f.name)
    (dst / "ground_truth/broken_large").mkdir(parents=True)
    for f in (dst / "test/broken_large").glob("*.png"):
        shutil.copy(src / "ground_truth/broken_large" / f"{f.stem}_mask.png", dst / "ground_truth/broken_large")
    return tmp_path / "mvtec"


def test_fit_export_score_roundtrip(tiny_bottle: Path, tmp_path: Path):
    models = tmp_path / "models"
    exported = fit_and_export("bottle", tiny_bottle, models, num_workers=0)
    assert exported == model_path(models, "bottle")

    csv_path = score_test_set(tiny_bottle, models, tmp_path / "raw", categories=("bottle",))
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "category,defect_type,image,label,mask,score"
    assert len(lines) == 1 + 4

    scores = [float(l.rsplit(",", 1)[1]) for l in lines[1:]]
    assert all(math.isfinite(s) for s in scores)
    maps = np.load(tmp_path / "raw" / "maps_bottle.npz")
    assert len(maps.files) == 4
    assert all(maps[k].shape == (MAP_SIZE, MAP_SIZE) for k in maps.files)
