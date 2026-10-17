"""Evaluation page: cross-validated metrics and the error/dataset analysis figures."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app_pages._style import inject, lede

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "reports" / "figures"

inject()
st.title("Evaluation")
lede("Results on the MVTec AD test set. Every decision figure is from 5-fold cross-validation, so each image was "
     "judged by thresholds calibrated without it. Ranges in brackets are 95% confidence intervals.")

metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())


def fmt(r):
    if r["value"] is None:
        return "n/a"
    return f"{r['k']}/{r['n']} ({r['value']:.1%}) [{r['ci95'][0]:.0%}–{r['ci95'][1]:.0%}]"


def figure(name: str, caption: str) -> None:
    path = FIGS / name
    if path.exists():
        with st.container(border=True):
            st.image(str(path), use_container_width=True)
            st.caption(caption)


overview, errors, vlm_tab = st.tabs(["Overview", "Error analysis", "Vision-language cross-check"])

with overview:
    for cat, a in metrics["auroc"].items():
        p = metrics["policy_cv"][cat]
        n, escalated = p["n"], p["escalation_rate"]["k"]
        auto = n - escalated
        wrong = p["false_approve_rate"]["k"] + p["missed_defect_rate"]["k"]
        with st.container(border=True):
            st.markdown(f"**{cat.capitalize()}**  \n{n} test images")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Image AUROC", f"{a['image']['value']:.3f}",
                      help=f"95% CI {a['image']['ci95'][0]:.3f}–{a['image']['ci95'][1]:.3f}")
            c2.metric("Decided automatically", f"{auto / n:.0%}", help=f"{auto} of {n} claims")
            c3.metric("Automatic decisions correct", f"{(auto - wrong) / auto:.1%}" if auto else "n/a",
                      help=f"{auto - wrong} of {auto}")
            c4.metric("Sent to a specialist", f"{escalated / n:.0%}", help=f"{escalated} of {n} claims")
    figure("08_confusion_matrices.png",
           "True state against the cross-validated decision. For screws, 66% of claims are decided automatically and "
           "98.1% of those are right; the other 34% go to a specialist. Both errors are the images that set the "
           "threshold edges.")
    with st.expander("All metrics"):
        rows = []
        for cat, a in metrics["auroc"].items():
            p = metrics["policy_cv"][cat]
            rows.append({
                "product": cat, "test images": p["n"],
                "image AUROC": f"{a['image']['value']:.3f} [{a['image']['ci95'][0]:.3f}–{a['image']['ci95'][1]:.3f}]",
                "pixel AUROC": f"{a['pixel']:.3f}",
                "sent to a specialist": fmt(p["escalation_rate"]),
                "approve precision": fmt(p["approve_precision"]),
                "reject precision": fmt(p["reject_precision"]),
                "good product approved": fmt(p["false_approve_rate"]),
                "defect rejected": fmt(p["missed_defect_rate"]),
            })
        st.dataframe(pd.DataFrame(rows).set_index("product").T, use_container_width=True)

with errors:
    st.markdown("Why claims are sent to a specialist, and where the model's attention goes.")
    for name, caption in [
        ("01_score_distributions.png", "Bottle classes separate cleanly. Screw classes overlap, and the overlap "
                                       "becomes the review band."),
        ("02_verdicts_by_defect_type.png", "Screw reviews concentrate in thread_side (13 of 23) and in good images "
                                           "(20 of 41). Thread_top almost never needs review."),
        ("03_defect_size_vs_score.png", "Smaller defects score lower and are sent for review (Spearman ρ = 0.48, "
                                        "Mann-Whitney p < 0.0001)."),
        ("04_screw_lighting_orientation.png", "Every good screw test image is darker than every defective one. A "
                                              "relighting experiment showed this does not explain the reviews, and "
                                              "orientation does not predict the score."),
        ("05_defect_locations.png", "Where defects occur in each defect type (mean annotation mask)."),
        ("06_screw_error_gallery.png", "The two cross-validation errors and the hardest reviews. On some hard cases "
                                       "the strongest colour sits away from the defect."),
        ("07_screw_background_contamination.png", "Lint and dust on the backdrop approved claims for the wrong reason. "
                                                  "The score now counts only the segmented product."),
    ]:
        figure(name, caption)

with vlm_tab:
    vlm_path = ROOT / "reports" / "vlm_metrics.json"
    if not vlm_path.exists():
        st.info("Run python -m cv.vlm_eval to produce the cross-check evaluation.")
    else:
        v = json.loads(vlm_path.read_text())
        st.markdown("The vision-language model (GLM-5.3-Flash) gives an advisory second opinion on claims sent to a "
                    "specialist. It never changes a decision.")
        e, o = v["escalated"], v["object_check"]
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            r = e["defect_seen_on_defective (recall)"]
            c1.metric("Defects seen on reviewed claims", f"{r['k']} of {r['n']}")
            r = e["defect_claimed_on_good (false alarm)"]
            c2.metric("False alarms on good products", f"{r['k']} of {r['n']}")
            r = o["wrong_object_caught"]
            c3.metric("Wrong items recognised", f"{r['k']} of {r['n']}")
        st.caption("Re-running the same photos changed the defect answer on 9 of 54 reviewed claims; the product check "
                   "gave the same answer every time. Treat the defect opinion as a hint.")
        vf = lambda r: f"{r['k']}/{r['n']} ({r['rate']:.0%}) [{r['ci95'][0]:.0%}–{r['ci95'][1]:.0%}]"  # noqa: E731
        with st.expander("All cross-check metrics"):
            st.dataframe(pd.DataFrame({s: {k: vf(r) for k, r in v[s].items()} for s in ("escalated", "control")}),
                         use_container_width=True)
            st.dataframe(pd.DataFrame({"rate": {k: vf(r) for k, r in v["object_check"].items()}}),
                         use_container_width=True)
