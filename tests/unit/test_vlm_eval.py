import numpy as np
import pytest
from PIL import Image

from cv.vlm_eval import run
from tests.stubs import StubLLM

OK = '{"object_matches_product": true, "defect_visible": false, "description": "Intact screw."}'


@pytest.fixture
def todo(tmp_path):
    p = tmp_path / "img.png"
    Image.fromarray(np.full((96, 96, 3), 150, np.uint8)).save(p)
    return [dict(set="control", image=str(p), product="screw", defect_type="good", label=0,
                 matches_truth=1, oof_verdict="REJECT") for _ in range(5)]


def test_successful_run_writes_results_with_problems_column(tmp_path, todo):
    out = tmp_path / "vlm.csv"
    rows = run(StubLLM(OK), todo=todo, out_csv=out, workers=2)
    assert all(r["source"] == "vlm" for r in rows)
    assert out.exists() and "problems" in out.read_text().splitlines()[0]


def test_mostly_failed_run_refuses_to_overwrite_previous_results(tmp_path, todo):
    out = tmp_path / "vlm.csv"
    out.write_text("previous good results")
    with pytest.raises(RuntimeError, match="429"):
        run(StubLLM(RuntimeError("Error code: 429 rate limited")), todo=todo, out_csv=out, workers=2)
    assert out.read_text() == "previous good results"
