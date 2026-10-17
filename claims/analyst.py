"""Claims Analyst: a chat agent that answers questions about the warranty-claims evaluation results.

Same rule as the claims pipeline: the LLM never produces numbers. Every figure comes from a deterministic
tool over the committed result files (reports/oof_verdicts.csv, reports/metrics.json, config/thresholds.json,
reports/dataset_features.csv, reports/vlm_*.{csv,json}); the LLM chooses tools and phrases the answer.
A live eval (tests/eval/test_claims_analyst_live.py) checks that every number in an answer appears in a
tool result.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

ROOT = Path(__file__).parent.parent
VERDICTS = ["APPROVE", "ESCALATE", "REJECT"]

SYSTEM_PROMPT = """You are the Claims Analyst for an automated warranty-claims system.
The system inspects a product photo (PatchCore anomaly detection on the segmented product), then a fail-closed
policy APPROVEs (defect found), REJECTs (no defect) or ESCALATEs to a human specialist. It was evaluated on the
MVTec AD test set for two products, bottle and screw. All verdicts in the data are 5-fold out-of-fold: each image
was judged by thresholds calibrated without it.

You answer questions about these results: verdicts, wrong decisions, escalations, defect types, thresholds,
individual test images, and the VLM cross-check evaluation.

Rules:
1. Every number you state must appear in a tool result from this conversation. Never estimate, extrapolate or
   compute new statistics. If no tool provides it, say the data does not contain it.
2. Before stating any number, call a tool in this turn; do not answer from memory. Call several tools if needed.
   A list tool shows only some rows: use the totals it reports, never generalise from the rows shown.
3. "good" = no defect (correct verdict REJECT); a defect type = defective (correct verdict APPROVE).
4. If the question is unrelated to the claims results, reply in one sentence that you only answer questions
   about the warranty-claims evaluation.
5. Be concise. Use a small markdown table for lists of more than three items."""


class ClaimsData:
    """Read-only view of the committed evaluation outputs."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)

    @cached_property
    def oof(self) -> pd.DataFrame:
        df = pd.read_csv(self.root / "reports" / "oof_verdicts.csv", keep_default_na=False)
        df["truth"] = df["label"].map({0: "good", 1: "defective"})
        feats = self.root / "reports" / "dataset_features.csv"
        if feats.exists():
            f = pd.read_csv(feats, keep_default_na=False, na_values=[""])
            df = df.merge(f[f.split == "test"][["image", "gt_area"]], on="image", how="left")
        return df

    @cached_property
    def metrics(self) -> dict:
        return json.loads((self.root / "reports" / "metrics.json").read_text())

    @cached_property
    def thresholds(self) -> dict:
        return json.loads((self.root / "config" / "thresholds.json").read_text())

    @cached_property
    def vlm_metrics(self) -> dict | None:
        p = self.root / "reports" / "vlm_metrics.json"
        return json.loads(p.read_text()) if p.exists() else None

    @cached_property
    def vlm_rows(self) -> dict[str, dict]:
        p = self.root / "reports" / "vlm_crosscheck.csv"
        if not p.exists():
            return {}
        rows = pd.read_csv(p, keep_default_na=False)
        rows = rows[rows["set"] != "foreign"]
        return {r["image"].removeprefix("data/mvtec/"): r for r in rows.to_dict("records")}

    @property
    def products(self) -> list[str]:
        return sorted(self.oof["category"].unique())

    def check_product(self, product: str | None) -> str | None:
        if product is not None and product not in self.products:
            return f"Unknown product {product!r}. Available: {', '.join(self.products)}."
        return None


def _fmt_rate(r: dict) -> str:
    if r.get("value") is None:
        return "n/a"
    return f"{r['k']}/{r['n']} ({r['value']:.1%}, 95% CI {r['ci95'][0]:.0%}-{r['ci95'][1]:.0%})"


def _table(df: pd.DataFrame) -> str:
    return df.to_string(index=False) if len(df) else "(no rows)"


def make_tools(data: ClaimsData) -> list:
    @tool
    def results_summary(product: str | None = None) -> str:
        """Headline results per product: number of test images, image AUROC, pixel AUROC, escalation rate,
        approve/reject precision, good products wrongly approved, defects wrongly rejected. product: bottle,
        screw, or omit for all."""
        if err := data.check_product(product):
            return err
        out = []
        for p in [product] if product else data.products:
            m, a = data.metrics["policy_cv"][p], data.metrics["auroc"][p]
            out.append(
                f"{p}: {m['n']} test images\n"
                f"  image AUROC {a['image']['value']:.3f} (95% CI {a['image']['ci95'][0]:.3f}-{a['image']['ci95'][1]:.3f})"
                f", pixel AUROC {a['pixel']:.3f}\n"
                f"  escalated to a human: {_fmt_rate(m['escalation_rate'])}\n"
                f"  approve precision: {_fmt_rate(m['approve_precision'])}\n"
                f"  reject precision: {_fmt_rate(m['reject_precision'])}\n"
                f"  good products wrongly approved: {_fmt_rate(m['false_approve_rate'])}\n"
                f"  defects wrongly rejected: {_fmt_rate(m['missed_defect_rate'])}")
        return "\n".join(out) + f"\n(verdicts are {data.metrics['k_folds']}-fold out-of-fold)"

    @tool
    def confusion_matrix(product: str) -> str:
        """True state (good / defective) x out-of-fold verdict (APPROVE / ESCALATE / REJECT) counts for one
        product, plus how many claims were decided automatically and how many of those were correct."""
        if err := data.check_product(product):
            return err
        g = data.oof[data.oof.category == product]
        cm = pd.crosstab(g.truth, g.verdict).reindex(index=["defective", "good"], columns=VERDICTS, fill_value=0)
        auto = g[g.verdict != "ESCALATE"]
        right = ((auto.truth == "defective") & (auto.verdict == "APPROVE")) | \
                ((auto.truth == "good") & (auto.verdict == "REJECT"))
        return (f"{product} confusion matrix (rows = truth, columns = verdict):\n{cm.to_string()}\n"
                f"decided automatically: {len(auto)}/{len(g)}; automatic decisions correct: {int(right.sum())}/{len(auto)}")

    @tool
    def verdicts_by_defect_type(product: str) -> str:
        """Per defect type of one product: number of images, APPROVE/ESCALATE/REJECT counts, and the median
        ground-truth defect size (% of the image). 'good' means no defect."""
        if err := data.check_product(product):
            return err
        g = data.oof[data.oof.category == product]
        t = pd.crosstab(g.defect_type, g.verdict).reindex(columns=VERDICTS, fill_value=0)
        t.insert(0, "images", t.sum(axis=1))
        t["pct_of_this_type_escalated"] = (100 * t["ESCALATE"] / t["images"]).round(1)
        if "gt_area" in g:
            t["median_defect_area_%"] = (g.groupby("defect_type").gt_area.median() * 100).round(2)
        t = t.sort_values("pct_of_this_type_escalated", ascending=False)
        note = ("Sorted by pct_of_this_type_escalated (share of THIS defect type's images that were escalated, "
                "highest first):\n")
        return note + t.reset_index().to_string(index=False)

    @tool
    def find_claims(product: str | None = None, defect_type: str | None = None, verdict: str | None = None,
                    truth: str | None = None, sort: str = "score_asc", limit: int = 10) -> str:
        """List test images matching filters. verdict: APPROVE/ESCALATE/REJECT. truth: good/defective.
        sort: score_asc or score_desc. Returns image, truth, defect type, anomaly score, verdict, the fold's
        reject/approve thresholds, and the total number of matches."""
        if err := data.check_product(product):
            return err
        g = data.oof
        if product:
            g = g[g.category == product]
        if defect_type:
            g = g[g.defect_type == defect_type]
        if verdict:
            g = g[g.verdict == verdict.upper()]
        if truth:
            g = g[g.truth == truth.lower()]
        g = g.sort_values("score", ascending=(sort != "score_desc"))
        limit = max(1, min(int(limit), 50))
        cols = g[["image", "truth", "defect_type", "score", "verdict", "band_lo", "band_hi"]].head(limit).round(2)
        breakdown = ", ".join(f"{k} {v}" for k, v in g.truth.value_counts().sort_index().items())
        by_type = ", ".join(f"{k} {v}" for k, v in g.defect_type.value_counts().items())
        order = "highest" if sort == "score_desc" else "lowest"
        return (f"{len(g)} matching images (truth: {breakdown or 'none'}; by type: {by_type or 'none'}); "
                f"showing the {len(cols)} {order}-scoring:\n{_table(cols)}")

    @tool
    def wrong_decisions(product: str | None = None) -> str:
        """Automatic decisions that were wrong (good product APPROVED, or defective product REJECTED), with the
        image's score and the thresholds of the fold that judged it."""
        if err := data.check_product(product):
            return err
        g = data.oof if product is None else data.oof[data.oof.category == product]
        bad = g[((g.truth == "good") & (g.verdict == "APPROVE")) | ((g.truth == "defective") & (g.verdict == "REJECT"))]
        cols = bad[["image", "truth", "score", "verdict", "band_lo", "band_hi"]].round(2)
        why = []
        for r in bad.itertuples():  # the explanation is computed, so the model only has to repeat it
            if r.verdict == "APPROVE":
                why.append(f"- {r.image}: good product scored {r.score:.2f}, at or above this fold's approve threshold "
                           f"{r.band_hi:.2f}, so it was approved automatically. The threshold was calibrated without "
                           f"this image, and no good image in the other folds scored that high.")
            else:
                why.append(f"- {r.image}: defective product scored {r.score:.2f}, below this fold's reject threshold "
                           f"{r.band_lo:.2f}, so it was rejected automatically. The threshold was calibrated without "
                           f"this image, and no defective image in the other folds scored that low.")
        note = ("These images were decided automatically, outside the review band (not escalated).\n" if len(bad)
                else "")
        return (f"{len(bad)} wrong automatic decisions out of {len(g)} images:\n{_table(cols)}\n{note}"
                + "\n".join(why))

    @tool
    def claim_details(image: str) -> str:
        """Everything recorded about one test image, e.g. 'screw/test/good/005.png' or 'good/005': truth, score,
        verdict, fold thresholds, defect size, and the VLM cross-check opinion if it was evaluated."""
        key = image.strip().removeprefix("data/mvtec/")
        hits = data.oof[data.oof.image.str.contains(key, regex=False)]
        if hits.empty:
            return f"No test image matches {image!r}."
        if len(hits) > 1:
            return f"{len(hits)} images match {image!r}; be more specific, e.g. {hits.image.iloc[0]!r}."
        r = hits.iloc[0]
        lines = [f"image: {r.image}", f"product: {r.category}", f"truth: {r.truth} ({r.defect_type})",
                 f"anomaly score: {r.score:.2f}", f"verdict: {r.verdict} (out-of-fold, fold {r.fold})",
                 f"fold thresholds: reject below {r.band_lo:.2f}, approve at or above {r.band_hi:.2f}"]
        if "gt_area" in r and pd.notna(r.gt_area) and r.label == 1:
            lines.append(f"ground-truth defect area: {r.gt_area * 100:.2f}% of the image")
        v = data.vlm_rows.get(r.image)
        if v:
            lines.append(f"VLM cross-check ({v['set']} set): defect_visible={v['defect_visible']}, "
                         f"object_matches_product={v['object_matches_product']}, description: {v['description']}")
        return "\n".join(lines)

    @tool
    def decision_thresholds() -> str:
        """Deployed thresholds per product: reject below lo, approve at or above hi, escalate in between; the
        maximum flagged area before a photo is treated as the wrong object; the plausible product coverage range."""
        out = []
        for p, b in data.thresholds.items():
            line = f"{p}: reject below {b['lo']:.2f}, approve at or above {b['hi']:.2f}"
            if b.get("max_area") is not None:
                line += f", wrong-object guard above {b['max_area'] * 100:.1f}% flagged area"
            if b.get("object_range"):
                line += f", product coverage must be {b['object_range'][0] * 100:.1f}-{b['object_range'][1] * 100:.1f}%"
            out.append(line)
        return "\n".join(out)

    @tool
    def vlm_crosscheck_results() -> str:
        """Offline evaluation of the advisory VLM cross-check (GLM-5.3-Flash) on escalated claims, a control set,
        and wrong-object photos. The VLM never changes a verdict."""
        m = data.vlm_metrics
        if not m:
            return "No VLM evaluation results found."
        out = [f"calls: {m['calls']}, unavailable: {m['unavailable']}"]
        for s in ("escalated", "control"):
            out += [f"{s}: {k} {_fmt_rate(dict(v, value=v['rate']))}" for k, v in m[s].items()]
        out += [f"object check: {k} {_fmt_rate(dict(v, value=v['rate']))}" for k, v in m["object_check"].items()]
        return "\n".join(out)

    return [results_summary, confusion_matrix, verdicts_by_defect_type, find_claims, wrong_decisions,
            claim_details, decision_thresholds, vlm_crosscheck_results]


_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[str]:
    """Numbers in text, normalised: thousands separators removed; 33.80 and 33.8 are the same number."""
    out = set()
    for n in _NUM.findall(text):
        n = n.replace(",", "").rstrip(".")
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def ungrounded(answer: str, sources: str) -> set[str]:
    """Numbers in the answer that appear in none of the sources (tool results and user messages)."""
    return numbers(answer) - numbers(sources)


CORRECTION = ("Your answer contains numbers that appear in no tool result: {nums}. Call the tools and answer "
              "again, quoting only numbers that the tools returned.")


@dataclass
class AnalystReply:
    text: str
    verified: bool                                   # every number appears in a tool result or user message
    attempts: int
    tool_calls: list = field(default_factory=list)   # (name, args) made while answering this question
    ungrounded: set = field(default_factory=set)


class ClaimsAnalyst:
    """ReAct agent plus a deterministic grounding check: an answer with numbers found in no tool result of the
    conversation is retried once with a correction; if it still fails, it is returned with a visible warning."""

    def __init__(self, llm, data: ClaimsData | None = None, checkpointer=None, max_attempts: int = 2):
        from langgraph.checkpoint.memory import MemorySaver
        self.agent = build_analyst(llm, data, checkpointer or MemorySaver())
        self.max_attempts = max_attempts

    def ask(self, question: str, thread_id: str = "default", recursion_limit: int = 16) -> AnalystReply:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
        start = len(self.agent.get_state(config).values.get("messages", []))
        message, calls, stray, answer = question, [], set(), ""
        for attempt in range(1, self.max_attempts + 1):
            msgs = self.agent.invoke({"messages": [("user", message)]}, config=config)["messages"]
            answer = msgs[-1].content if isinstance(msgs[-1].content, str) else str(msgs[-1].content)
            # sources: tool results and the user's own questions, never our correction messages (they quote the
            # very numbers being rejected, which would make a repeated invention look grounded)
            sources = "\n".join(m.content for m in msgs if isinstance(m.content, str) and (
                m.type == "tool" or (m.type == "human" and not m.content.startswith(CORRECTION[:40]))))
            calls = [(c["name"], c["args"]) for m in msgs[start:] for c in (getattr(m, "tool_calls", None) or [])]
            stray = ungrounded(answer, sources)
            if not stray:
                return AnalystReply(answer, True, attempt, calls)
            message = CORRECTION.format(nums=", ".join(sorted(stray)))
        warning = (f"⚠ Unverified: this answer contains numbers not found in the data ({', '.join(sorted(stray))}). "
                   "Check them before relying on it.\n\n")
        return AnalystReply(warning + answer, False, self.max_attempts, calls, stray)


def build_analyst(llm, data: ClaimsData | None = None, checkpointer=None):
    from langchain.agents import create_agent
    return create_agent(llm, make_tools(data or ClaimsData()), system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
