import pandas as pd
import pytest

import cv.evaluate as ev
from cv.evaluate import out_of_fold_verdicts, summarize, wilson_interval


def make_scores(n_good=10, n_bad=10, categories=("bottle",)):
    rows = []
    for cat in categories:
        for i in range(n_good):
            rows.append(dict(category=cat, defect_type="good", image=f"{cat}/test/good/{i:03d}.png",
                             label=0, mask="", score=0.10 + i * 0.01))
        for i in range(n_bad):
            dt = "crack" if i % 2 else "dent"
            rows.append(dict(category=cat, defect_type=dt, image=f"{cat}/test/{dt}/{i:03d}.png",
                             label=1, mask="m.png", score=0.15 + i * 0.05))
    df = pd.DataFrame(rows)
    df["score"] += df.index * 1e-6  # unique scores, so score sets identify rows
    return df


def test_wilson_known_value():
    lo, hi = wilson_interval(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3)
    assert hi == pytest.approx(1.0)


def test_wilson_empty_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_every_image_gets_exactly_one_out_of_fold_verdict():
    df = make_scores(categories=("bottle", "screw"))
    oof = out_of_fold_verdicts(df, k=5, seed=0)
    assert len(oof) == len(df)
    assert oof["image"].is_unique
    assert set(oof["verdict"]) <= {"APPROVE", "REJECT", "ESCALATE"}


def test_held_out_fold_never_used_for_its_own_calibration(monkeypatch):
    df = make_scores(n_good=15, n_bad=15)
    calls = []
    real = ev.calibrate_band

    def spy(scores, labels):
        calls.append(set(scores))
        return real(scores, labels)

    monkeypatch.setattr(ev, "calibrate_band", spy)
    oof = out_of_fold_verdicts(df, k=5, seed=0)

    assert len(calls) == 5
    for fold, group in oof.groupby("fold"):
        held_out = set(group["score"])
        used = [c for c in calls if not (c & held_out)]
        assert len(used) == 1, f"fold {fold} not cleanly held out"
        assert used[0] | held_out == set(df["score"])


def test_band_is_calibrated_per_category():
    df = make_scores(categories=("bottle", "screw"))
    df.loc[df.category == "screw", "score"] += 100  # wildly different scale
    both = out_of_fold_verdicts(df, k=5, seed=0).set_index("image")
    alone = out_of_fold_verdicts(df[df.category == "screw"], k=5, seed=0).set_index("image")
    pd.testing.assert_series_equal(both.loc[alone.index, "verdict"], alone["verdict"])


def test_summarize_counts():
    oof = pd.DataFrame([
        dict(category="bottle", defect_type="good", label=0, verdict="REJECT"),
        dict(category="bottle", defect_type="good", label=0, verdict="APPROVE"),   # false approve
        dict(category="bottle", defect_type="good", label=0, verdict="ESCALATE"),
        dict(category="bottle", defect_type="crack", label=1, verdict="APPROVE"),
        dict(category="bottle", defect_type="crack", label=1, verdict="REJECT"),   # missed defect
        dict(category="bottle", defect_type="dent", label=1, verdict="ESCALATE"),
    ])
    s = summarize(oof)["bottle"]
    assert s["n"] == 6
    assert s["escalation_rate"]["value"] == pytest.approx(2 / 6)
    assert s["approve_precision"]["value"] == pytest.approx(1 / 2)
    assert s["reject_precision"]["value"] == pytest.approx(1 / 2)
    assert s["false_approve_rate"]["value"] == pytest.approx(1 / 3)
    assert s["missed_defect_rate"]["value"] == pytest.approx(1 / 3)
    crack = s["by_defect_type"]["crack"]
    assert crack["n"] == 2 and crack["APPROVE"] == 1 and crack["REJECT"] == 1
