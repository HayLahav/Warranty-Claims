"""Error-analysis and dataset-analysis figures -> reports/figures/*.png

Inputs: reports/oof_verdicts.csv, reports/dataset_features.csv (cv.analyze_dataset),
        reports/raw/maps_<category>.npz, config/thresholds.json, data/mvtec.
Every threshold-dependent figure uses OUT-OF-FOLD verdicts.
Usage: python -m cv.error_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated with the dataviz palette validator (light surface).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASS = {"good": "#2a78d6", "defective": "#eb6834"}
VERDICT = {"APPROVE": "#1baf7a", "REJECT": "#6250d6", "ESCALATE": "#eda100"}
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

DATA, REPORTS = Path("data/mvtec"), Path("reports")
FIGS = REPORTS / "figures"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110,
    })


def _bands():
    return json.loads(Path("config/thresholds.json").read_text())


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def score_distributions(oof: pd.DataFrame):
    bands = _bands()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (cat, g) in zip(axes, oof.groupby("category")):
        bins = np.linspace(g.score.min(), g.score.max(), 36)
        for label, name in [(0, "good"), (1, "defective")]:
            vals = g.loc[g.label == label, "score"]
            ax.hist(vals, bins=bins, color=CLASS[name], alpha=0.35, histtype="stepfilled", lw=0)
            ax.hist(vals, bins=bins, color=CLASS[name], histtype="step", lw=1.6,
                    label=f"{name} (n={len(vals)})")
        b = bands[cat]
        if b["hi"] > b["lo"]:
            ax.axvspan(b["lo"], b["hi"], color=VERDICT["ESCALATE"], alpha=0.15, lw=0)
            ax.text((b["lo"] + b["hi"]) / 2, ax.get_ylim()[1] * 0.97, "escalation\nband", ha="center", va="top",
                    color=INK_2, fontsize=9)
        for x in {b["lo"], b["hi"]}:
            ax.axvline(x, color=INK_2, lw=1, ls="--")
        ax.set_title(f"{cat}: anomaly score by true class")
        ax.set_xlabel("anomaly score (product region)")
        ax.set_ylabel("images")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return _save(fig, "01_score_distributions.png")


def verdicts_by_type(oof: pd.DataFrame):
    order = ["APPROVE", "ESCALATE", "REJECT"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [4, 6], "wspace": 0.75})
    for ax, (cat, g) in zip(axes, oof.groupby("category")):
        tab = pd.crosstab(g.defect_type, g.verdict).reindex(columns=order, fill_value=0)
        tab = tab.loc[["good"] + sorted(t for t in tab.index if t != "good")]
        share = tab.div(tab.sum(axis=1), axis=0)
        left = np.zeros(len(tab))
        y = np.arange(len(tab))[::-1]
        for v in order:
            ax.barh(y, share[v], left=left, color=VERDICT[v], height=0.62, edgecolor=SURFACE, linewidth=2, label=v)
            for yi, l, s, n in zip(y, left, share[v], tab[v]):
                if s >= 0.12:
                    ax.text(l + s / 2, yi, str(n), ha="center", va="center", fontsize=9, color=INK)
            left += share[v]
        ax.set_yticks(y, [f"{t}  (n={tab.loc[t].sum()})" for t in tab.index])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{cat}: out-of-fold verdict by defect type")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.text(0.01, -0.14, "Correct: good→REJECT, defect→APPROVE.  ESCALATE = sent to a human.  Numbers are image counts.",
             color=INK_2, fontsize=9)
    return _save(fig, "02_verdicts_by_defect_type.png")


def defect_size_vs_score(feats: pd.DataFrame):
    bands = _bands()
    d = feats[(feats.split == "test") & (feats.label == 1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"wspace": 0.3})
    for ax, (cat, g) in zip(axes, d.groupby("category")):
        for v in ["APPROVE", "ESCALATE", "REJECT"]:
            s = g[g.verdict == v]
            if len(s):
                ax.scatter(s.gt_area * 100, s.score, s=34, color=VERDICT[v], edgecolor=SURFACE, linewidth=1.5,
                           label=f"{v} (n={len(s)})", zorder=3)
        b = bands[cat]
        for x in {b["lo"], b["hi"]}:
            ax.axhline(x, color=INK_2, lw=1, ls="--")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_xlabel("true defect area (ground-truth mask, % of image, log scale)")
        ax.set_ylabel("anomaly score")
        ax.set_title(f"{cat}: defective images, size vs score")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return _save(fig, "03_defect_size_vs_score.png")


def screw_lighting_orientation(feats: pd.DataFrame):
    s = feats[feats.category == "screw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    groups = [("train good", s[s.split == "train"]), ("test good", s[(s.split == "test") & (s.label == 0)]),
              ("test defective", s[(s.split == "test") & (s.label == 1)])]
    rng = np.random.default_rng(0)
    for i, (name, g) in enumerate(groups):
        color = CLASS["defective"] if "defective" in name else CLASS["good"]
        ax.scatter(g.bg_brightness, i + rng.uniform(-0.18, 0.18, len(g)), s=10, color=color,
                   alpha=0.35 if name == "train good" else 0.8, edgecolor="none")
        ax.plot([g.bg_brightness.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
    tr = groups[0][1].bg_brightness
    ax.axvspan(tr.min(), tr.max(), color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.set_yticks(range(3), [f"{n} (n={len(g)})" for n, g in groups])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("background brightness (median grey level)")
    ax.set_title("screw: lighting shift — test 'good' is darker than training")

    ax = axes[1]
    ta = s[s.split == "train"].angle.to_numpy()
    gap = lambda a: np.abs((ta - a + 180) % 360 - 180).min()  # noqa: E731
    good = s[(s.split == "test") & (s.label == 0)].assign(angle_gap=lambda x: x.angle.map(gap))
    for v in ["APPROVE", "ESCALATE", "REJECT"]:
        g = good[good.verdict == v]
        if len(g):
            ax.scatter(g.angle, g.score, s=34, color=VERDICT[v], edgecolor=SURFACE, linewidth=1.5, label=f"{v} (n={len(g)})")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel("screw orientation (degrees, head direction)")
    ax.set_ylabel("anomaly score")
    ax.set_title("screw good images: orientation does not predict score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return _save(fig, "04_screw_lighting_orientation.png")


def defect_locations(feats: pd.DataFrame, size: int = 96):
    from PIL import Image
    d = feats[(feats.split == "test") & (feats.label == 1)]
    types = list(d.groupby(["category", "defect_type"]).groups)
    fig, axes = plt.subplots(1, len(types), figsize=(1.5 * len(types), 2.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    oof = pd.read_csv(REPORTS / "oof_verdicts.csv", keep_default_na=False).set_index("image")
    for ax, (cat, dt) in zip(axes, types):
        masks = [np.asarray(Image.open(DATA / oof.loc[img, "mask"]).convert("L").resize((size, size))) / 255
                 for img in d[(d.category == cat) & (d.defect_type == dt)].image]
        ax.imshow(np.mean(masks, axis=0), cmap=cmap, vmin=0, vmax=0.5)
        ax.set_title(f"{cat}\n{dt.replace('_', ' ')}", fontsize=8, fontweight="normal", loc="center")
        ax.axis("off")
    fig.suptitle("Where defects occur: share of images with a defect at each pixel", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=1.08)
    return _save(fig, "05_defect_locations.png")


def error_gallery(oof: pd.DataFrame, n_escalated: int = 4):
    """Out-of-fold errors plus the hardest escalations: photo | anomaly heatmap | ground truth."""
    from PIL import Image

    from claims.inspector import heat_alpha
    from claims.policy import Band
    s = oof[oof.category == "screw"]
    wrong = s[((s.label == 0) & (s.verdict == "APPROVE")) | ((s.label == 1) & (s.verdict == "REJECT"))]
    esc_good = s[(s.label == 0) & (s.verdict == "ESCALATE")].nlargest(n_escalated // 2, "score")
    esc_bad = s[(s.label == 1) & (s.verdict == "ESCALATE")].nsmallest(n_escalated // 2, "score")
    rows = pd.concat([wrong, esc_good, esc_bad])
    maps = np.load(REPORTS / "raw/maps_screw.npz")
    band = _bands()["screw"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 2.35 * len(rows)))
    for r, row in zip(axes, rows.itertuples()):
        photo = Image.open(DATA / row.image).convert("RGB").resize((256, 256))
        r[0].imshow(photo)
        norm, alpha = heat_alpha(maps[row.image].astype(np.float32), Band(**band))
        r[1].imshow(photo)
        r[1].imshow(norm, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        gt = np.zeros((256, 256)) if not row.mask else np.asarray(Image.open(DATA / row.mask).convert("L").resize((256, 256)))
        r[2].imshow(gt, cmap="gray", vmin=0, vmax=255)
        truth = "good" if row.label == 0 else row.defect_type
        r[0].set_title(f"{row.image.split('/', 2)[-1]}\ntruth: {truth}", fontsize=8, fontweight="normal", loc="left")
        r[1].set_title(f"score {row.score:.1f} → {row.verdict}", fontsize=8, fontweight="normal", loc="left")
        r[2].set_title("ground-truth mask", fontsize=8, fontweight="normal", loc="left")
        for a in r:
            a.axis("off")
    fig.suptitle("screw: out-of-fold errors (top 2) and hardest escalations", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "06_screw_error_gallery.png")


ERROR = "#c2410c"
CELL_MEANING = {  # (truth, verdict) -> label printed in the cell, so meaning never relies on colour
    ("defective", "APPROVE"): "✓ correct", ("defective", "ESCALATE"): "→ human", ("defective", "REJECT"): "✗ wrongly refused",
    ("good", "APPROVE"): "✗ wrongly paid", ("good", "ESCALATE"): "→ human", ("good", "REJECT"): "✓ correct",
}


def confusion_matrices(oof: pd.DataFrame):
    """Truth (rows) x out-of-fold verdict (columns), one panel per product. Fill = share of the true class."""
    rows, cols = ["defective", "good"], ["APPROVE", "ESCALATE", "REJECT"]
    cats = sorted(oof.category.unique())
    fig, axes = plt.subplots(1, len(cats), figsize=(5.6 * len(cats), 3.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES[:6])
    for ax, cat in zip(np.atleast_1d(axes), cats):
        g = oof[oof.category == cat]
        truth = g.label.map({0: "good", 1: "defective"})
        counts = pd.crosstab(truth, g.verdict).reindex(index=rows, columns=cols, fill_value=0)
        share = counts.div(counts.sum(axis=1), axis=0)
        ax.imshow(share.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        for i, t in enumerate(rows):
            for j, v in enumerate(cols):
                n, p = counts.iat[i, j], share.iat[i, j]
                ink = SURFACE if p > 0.6 else INK
                ax.text(j, i - 0.17, f"{n}", ha="center", va="center", fontsize=17, fontweight="bold", color=ink)
                ax.text(j, i + 0.12, f"{p:.0%} of {t}", ha="center", va="center", fontsize=8, color=ink)
                ax.text(j, i + 0.31, CELL_MEANING[(t, v)], ha="center", va="center", fontsize=7.8, color=ink,
                        fontweight="bold" if CELL_MEANING[(t, v)].startswith("✗") else "normal")
                if CELL_MEANING[(t, v)].startswith("✗"):
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=ERROR,
                                               lw=2.4 if n else 1.0, ls="-" if n else ":"))
        ax.set_xticks(range(3), cols)
        for tick, v in zip(ax.get_xticklabels(), cols):
            tick.set_color(INK_2)
        ax.xaxis.tick_top()
        ax.set_yticks(range(2), [f"{t}\n(n={int(counts.loc[t].sum())})" for t in rows])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        auto = g[g.verdict != "ESCALATE"]
        right = ((auto.label == 1) & (auto.verdict == "APPROVE")) | ((auto.label == 0) & (auto.verdict == "REJECT"))
        ax.set_title(f"{cat}: true state × out-of-fold verdict", pad=26)
        ax.text(-0.5, 1.75, f"auto-decided {len(auto)}/{len(g)} ({len(auto) / len(g):.0%})  ·  "
                            f"automatic decisions correct {int(right.sum())}/{len(auto)} ({right.mean():.1%})",
                fontsize=9, color=INK_2, va="center")
    fig.text(0.01, -0.04, "Fill = share of the true class (row). Red outline = automatic error. "
                          "→ human = escalated to a specialist. Each image judged by thresholds calibrated without it.",
             fontsize=8.5, color=INK_2)
    fig.tight_layout()
    return _save(fig, "08_confusion_matrices.png")


def main():
    _style()
    oof = pd.read_csv(REPORTS / "oof_verdicts.csv", keep_default_na=False)
    feats = pd.read_csv(REPORTS / "dataset_features.csv", keep_default_na=False, na_values=[""])
    for fn, arg in [(score_distributions, oof), (verdicts_by_type, oof), (defect_size_vs_score, feats),
                    (screw_lighting_orientation, feats), (defect_locations, feats), (error_gallery, oof),
                    (confusion_matrices, oof)]:
        print("wrote", fn(arg))


if __name__ == "__main__":
    main()
