"""PatchCore fit / export / scoring — shared by the Colab notebook and local code.

Post-processing is disabled on purpose: anomalib's post-processor min-max
normalises and clamps scores using thresholds it picks itself. We want raw
anomaly scores and do our own calibration (cv/calibrate.py), so every number
we report comes from one place.

Layout produced by export:   <models_dir>/<category>/weights/torch/model.pt
Layout produced by scoring:  <out_dir>/scores.csv, <out_dir>/maps_<category>.npz
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MAP_SIZE = 256  # anomaly maps are stored at this resolution (float16)


@dataclass(frozen=True)
class ImageRecord:
    category: str
    defect_type: str   # "good" or the MVTec defect folder name
    image: str         # path relative to the data root, posix style
    label: int         # 0 = good, 1 = defective
    mask: str          # ground-truth mask path relative to data root, "" for good


def list_test_images(data_root: Path, category: str) -> list[ImageRecord]:
    records = []
    for img in sorted((data_root / category / "test").glob("*/*.png")):
        defect_type = img.parent.name
        is_good = defect_type == "good"
        mask = "" if is_good else f"{category}/ground_truth/{defect_type}/{img.stem}_mask.png"
        if mask and not (data_root / mask).exists():
            raise FileNotFoundError(f"missing ground-truth mask {mask}")
        records.append(ImageRecord(category, defect_type, img.relative_to(data_root).as_posix(),
                                   0 if is_good else 1, mask))
    if not records:
        raise FileNotFoundError(f"no test images under {data_root / category / 'test'}")
    return records


def fit_and_export(category: str, data_root: Path, models_dir: Path,
                   coreset_sampling_ratio: float = 0.1, num_workers: int = 2) -> Path:
    """Fit a PatchCore memory bank on train/good and export a TorchInferencer model."""
    from anomalib.data import MVTecAD
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    if not (data_root / category).is_dir():
        raise FileNotFoundError(f"{data_root / category} not found — run cv.download_mvtec first")
    datamodule = MVTecAD(root=data_root, category=category, num_workers=num_workers)
    model = Patchcore(coreset_sampling_ratio=coreset_sampling_ratio,
                      post_processor=False, evaluator=False, visualizer=False)
    # No progress bar: Lightning's default RichProgressBar proxies stdout/stderr, and
    # PatchCore's coreset tqdm writing through it recurses forever inside Jupyter/Colab.
    # See tests/eval/test_fit_in_jupyter.py.
    engine = Engine(default_root_dir=models_dir / "_work", logger=False, enable_progress_bar=False)
    engine.fit(model=model, datamodule=datamodule)
    path = engine.export(model=model, export_type=ExportType.TORCH, export_root=models_dir / category)
    if path is None or not Path(path).exists():
        raise RuntimeError(f"export produced no model file for {category}")
    return Path(path)


def model_path(models_dir: Path, category: str) -> Path:
    return models_dir / category / "weights" / "torch" / "model.pt"


class PatchCoreScorer:
    """Thin wrapper over anomalib's TorchInferencer returning (score, map)."""

    def __init__(self, path: Path, device: str = "cpu"):
        from anomalib.deploy import TorchInferencer

        # Exported .pt files are pickles; anomalib refuses them unless
        # TRUST_REMOTE_CODE=1. We only load models we exported ourselves, so we
        # opt in for this load alone instead of process-wide.
        previous = os.environ.get("TRUST_REMOTE_CODE")
        os.environ["TRUST_REMOTE_CODE"] = "1"
        try:
            self._inferencer = TorchInferencer(path=path, device=device)
        finally:
            if previous is None:
                del os.environ["TRUST_REMOTE_CODE"]
            else:
                os.environ["TRUST_REMOTE_CODE"] = previous

    def score(self, image_path: Path) -> tuple[float, np.ndarray]:
        import cv2

        pred = self._inferencer.predict(image=image_path)
        score = float(pred.pred_score.detach().cpu().reshape(-1)[0])
        amap = pred.anomaly_map.detach().cpu().numpy().squeeze().astype(np.float32)
        if amap.shape != (MAP_SIZE, MAP_SIZE):
            amap = cv2.resize(amap, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_LINEAR)
        return score, amap


def score_test_set(data_root: Path, models_dir: Path, out_dir: Path,
                   categories=("bottle", "screw"), device: str = "cpu") -> Path:
    """Score every test image; write scores.csv and per-category anomaly maps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for category in categories:
        scorer = PatchCoreScorer(model_path(models_dir, category), device=device)
        maps = {}
        for rec in list_test_images(data_root, category):
            score, amap = scorer.score(data_root / rec.image)
            rows.append({**asdict(rec), "score": score})
            maps[rec.image] = amap.astype(np.float16)
        np.savez_compressed(out_dir / f"maps_{category}.npz", **maps)
        print(f"{category}: scored {len(maps)} images")

    csv_path = out_dir / "scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
