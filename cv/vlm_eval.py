"""Offline evaluation of the VLM cross-check (live API calls; ~100 images, a few cents).

Sets:
  escalated  every out-of-fold ESCALATE (all screw): does the VLM see the defect PatchCore was unsure about?
  control    screw images PatchCore decided confidently, 10 good + 10 defective: baseline VLM accuracy.
  foreign    12 other-category photos claimed as bottle and as screw: does it catch the wrong object?
             Plus the escalated/control images, which all DO match their product (false-alarm rate).

    python -m cv.vlm_eval            # writes reports/vlm_crosscheck.csv and reports/vlm_metrics.json
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from claims.cross_check import cross_check
from cv.evaluate import wilson_interval

ROOT = Path(__file__).parent.parent
OUT_CSV = ROOT / "reports" / "vlm_crosscheck.csv"
OUT_JSON = ROOT / "reports" / "vlm_metrics.json"
FIELDS = ["set", "image", "product", "defect_type", "label", "matches_truth", "oof_verdict",
          "object_matches_product", "defect_visible", "description", "source", "attempts", "problems"]
MAX_UNAVAILABLE = 0.10  # above this, the run is an outage, not a measurement: keep the previous results


def cases() -> list[dict]:
    rows = list(csv.DictReader(open(ROOT / "reports" / "oof_verdicts.csv", encoding="utf-8")))
    out = []
    for r in rows:
        if r["verdict"] == "ESCALATE":
            out.append(dict(set="escalated", image=f"data/mvtec/{r['image']}", product=r["category"],
                            defect_type=r["defect_type"], label=int(r["label"]), matches_truth=1,
                            oof_verdict=r["verdict"]))
    confident = [r for r in rows if r["category"] == "screw" and r["verdict"] != "ESCALATE"]
    for label in (0, 1):
        pool = [r for r in confident if int(r["label"]) == label]
        for r in pool[:: max(1, len(pool) // 10)][:10]:  # evenly spaced, deterministic
            out.append(dict(set="control", image=f"data/mvtec/{r['image']}", product="screw",
                            defect_type=r["defect_type"], label=label, matches_truth=1, oof_verdict=r["verdict"]))
    for p in sorted((ROOT / "data" / "foreign").glob("*.png")):
        for product in ("bottle", "screw"):
            out.append(dict(set="foreign", image=f"data/foreign/{p.name}", product=product,
                            defect_type=p.stem.rsplit("_", 1)[0], label=0, matches_truth=0, oof_verdict=""))
    return out


def run(vlm, workers: int = 4, todo: list[dict] | None = None, out_csv: Path = OUT_CSV) -> list[dict]:
    todo = cases() if todo is None else todo

    def one(c):
        r = cross_check(ROOT / c["image"], c["product"], vlm)
        return {**c, "object_matches_product": r.object_matches_product, "defect_visible": r.defect_visible,
                "description": r.description, "source": r.source, "attempts": r.attempts,
                "problems": " | ".join(r.problems)}

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, todo))
    failed = [r for r in results if r["source"] != "vlm"]
    if len(failed) > MAX_UNAVAILABLE * len(results):
        raise RuntimeError(f"{len(failed)}/{len(results)} VLM calls unavailable; previous results kept. "
                           f"First error: {failed[0]['problems'][:300]}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(results)
    return results


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["source"] == "vlm"]
    m = {"calls": len(rows), "unavailable": len(rows) - len(ok)}
    for s in ("escalated", "control"):
        sub = [r for r in ok if r["set"] == s]
        defective = [r for r in sub if int(r["label"]) == 1]
        good = [r for r in sub if int(r["label"]) == 0]
        m[s] = {
            "defect_seen_on_defective (recall)": _rate(sum(r["defect_visible"] is True for r in defective), len(defective)),
            "defect_claimed_on_good (false alarm)": _rate(sum(r["defect_visible"] is True for r in good), len(good)),
            "accuracy": _rate(sum((r["defect_visible"] is True) == (int(r["label"]) == 1) for r in sub), len(sub)),
        }
    wrong = [r for r in ok if r["set"] == "foreign"]
    right = [r for r in ok if r["set"] != "foreign"]
    m["object_check"] = {
        "wrong_object_caught": _rate(sum(r["object_matches_product"] is False for r in wrong), len(wrong)),
        "right_object_rejected (false alarm)": _rate(sum(r["object_matches_product"] is False for r in right), len(right)),
    }
    return m


def main() -> None:
    load_dotenv(ROOT / ".env")
    from agent.llm import get_vlm
    rows = run(get_vlm()) if "--reuse" not in sys.argv else [
        {**r, "object_matches_product": {"True": True, "False": False}.get(r["object_matches_product"]),
         "defect_visible": {"True": True, "False": False}.get(r["defect_visible"])}
        for r in csv.DictReader(open(OUT_CSV, encoding="utf-8"))]
    m = metrics(rows)
    OUT_JSON.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
