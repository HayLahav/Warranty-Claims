"""Download MVTec AD categories.

anomalib's built-in MVTec URL (mydrive.ch) returns 404 as of 2026-10, so we pull
from a Hugging Face mirror that keeps the original folder layout:
    <root>/<category>/{train/good, test/<defect_type>, ground_truth/<defect_type>}

MVTec AD is licensed CC BY-NC-SA 4.0 (see <root>/license.txt).

Usage: python -m cv.download_mvtec [--root data/mvtec] [--categories bottle screw] [--foreign]
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "foersben/mvtec-ad"
DEFAULT_CATEGORIES = ("bottle", "screw")

# Official MVTec AD test-set sizes; used to detect partial downloads.
EXPECTED_TEST_COUNTS = {"bottle": 83, "screw": 160}


def download(root: Path, categories=DEFAULT_CATEGORIES) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=[f"{c}/**" for c in categories] + ["license.txt"],
    )
    for c in categories:
        n = len(list((root / c / "test").glob("*/*.png")))
        expected = EXPECTED_TEST_COUNTS.get(c)
        if expected is not None and n != expected:
            raise RuntimeError(f"{c}: found {n} test images, expected {expected}")
        print(f"{c}: {n} test images OK")
    return root


FOREIGN_CATEGORIES = ("cable", "hazelnut", "metal_nut", "toothbrush")


def download_foreign(out: Path = Path("data/foreign"), per_category: int = 3) -> Path:
    """Held-out photos of objects we have no model for, used to test the out-of-distribution guard.
    They are never used for calibration."""
    import shutil
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    for cat in FOREIGN_CATEGORIES:
        for i in range(per_category):
            src = hf_hub_download(REPO_ID, f"{cat}/test/good/{i:03d}.png", repo_type="dataset")
            shutil.copy(src, out / f"{cat}_{i:03d}.png")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/mvtec"))
    p.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    p.add_argument("--foreign", action="store_true", help="also fetch held-out foreign-object photos")
    args = p.parse_args()
    download(args.root, args.categories)
    if args.foreign:
        print("foreign photos ->", download_foreign())
