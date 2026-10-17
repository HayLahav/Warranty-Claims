"""Regression: fit_and_export must work inside a Jupyter kernel (i.e. on Colab).

Lightning's default RichProgressBar wraps stdout/stderr in rich FileProxy
objects; PatchCore's coreset tqdm writes through them while the progress bar
is live, and in Jupyter rich renders via IPython back into the proxy ->
RecursionError. Passing under plain pytest proves nothing, so this test runs
the fit in a real kernel. Run with:  pytest -m eval tests/eval/test_fit_in_jupyter.py
"""
import shutil
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

DATA = Path("data/mvtec/bottle")
pytestmark = pytest.mark.eval


def test_fit_and_export_runs_in_jupyter_kernel(tmp_path):
    if not DATA.is_dir():
        pytest.skip("data/mvtec/bottle missing — run python -m cv.download_mvtec")
    root = tmp_path / "mvtec"
    for sub, n in [("train/good", 8), ("test/good", 2)]:
        (root / "bottle" / sub).mkdir(parents=True)
        for f in sorted((DATA / sub).glob("*.png"))[:n]:
            shutil.copy(f, root / "bottle" / sub)

    repo = str(Path.cwd())
    cell = (f"import sys; sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "from cv.patchcore import fit_and_export\n"
            f"print('EXPORTED', fit_and_export('bottle', Path({str(root)!r}), Path({str(tmp_path / 'models')!r}), num_workers=0))")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(cell)])
    nbclient.NotebookClient(nb, kernel_name="python3", timeout=600).execute()

    out = "".join(o.get("text", "") for o in nb.cells[0].outputs)
    assert "EXPORTED" in out
