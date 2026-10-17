"""Colab (GPU) scores in reports/raw/scores.csv must match local CPU inference.

Thresholds are calibrated on the Colab scores but applied to live CPU scores in
the app; if the two disagree, every band is silently wrong.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cv.patchcore import PatchCoreScorer, model_path

DATA, MODELS, RAW = Path("data/mvtec"), Path("models"), Path("reports/raw")
pytestmark = pytest.mark.eval


@pytest.fixture(scope="module")
def scores():
    if not (RAW / "scores.csv").exists() or not MODELS.is_dir():
        pytest.skip("Colab artifacts missing")
    return pd.read_csv(RAW / "scores.csv", keep_default_na=False)


@pytest.mark.parametrize("category", ["bottle", "screw"])
def test_cpu_scores_and_maps_match_colab(scores, category):
    sample = (scores[scores.category == category]
              .groupby("defect_type", group_keys=False).head(1))  # one image per defect type
    scorer = PatchCoreScorer(model_path(MODELS, category), device="cpu")
    maps = np.load(RAW / f"maps_{category}.npz")
    for row in sample.itertuples():
        score, amap = scorer.score(DATA / row.image)
        assert score == pytest.approx(row.score, rel=1e-3), row.image
        np.testing.assert_allclose(amap, maps[row.image].astype(np.float32), rtol=1e-2, atol=1e-2 * row.score)
