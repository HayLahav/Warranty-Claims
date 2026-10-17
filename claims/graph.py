"""Warranty-claims agent graph.

    route ──claim──────────► inspect ─► decide ─┬─────────────────► respond ─► END
      │                                          └─ESCALATE─► cross_check ─┘
      ├──unknown───────────────────────► decide(ESCALATE) ─► respond
      ├──analytics_question► handoff ─► END
      └──out_of_scope──────► decline ─► END

LLMs: route, respond, cross_check (VLM).  Deterministic: inspect (PatchCore), decide (policy).
cross_check is advisory: it runs only on escalated claims that have an inspected photo, writes a
note for the human specialist, and never changes the decision or reaches the customer reply.
"""
from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from claims.cross_check import cross_check
from claims.inspector import Inspector, check_image
from claims.policy import Band, Decision, Inspection, Verdict, decide
from claims.responder import write_reply
from claims.router import UNKNOWN, route_message

HANDOFF_REPLY = ("That sounds like a question about the claims system's results. "
                 "Please use the Claims Analyst tab for that.")
DECLINE_REPLY = ("I can only help with product warranty claims. "
                 "Please describe the problem with your product and attach a photo.")


class ClaimState(TypedDict, total=False):
    claim_text: str
    image_path: str | None
    product: str
    route: str
    route_reason: str
    image_ok: bool
    inspection: dict | None
    inspection_error: str | None
    decision: dict
    reply: str
    reply_source: str
    cross_check: dict | None
    trace: Annotated[list[str], operator.add]


def build_graph(*, llm, inspector: Inspector, bands: Mapping[str, Band], vlm=None, checkpointer=None):
    def route_node(state: ClaimState) -> dict:
        r = route_message(state["claim_text"], llm)
        return {"route": r.route, "route_reason": r.reason,
                "trace": [f"router[{r.source}, attempts={r.attempts}]: {r.route} — {r.reason}"]}

    def inspect_node(state: ClaimState) -> dict:
        image_path = state.get("image_path")
        if not check_image(image_path):
            return {"image_ok": False, "inspection": None, "inspection_error": None,
                    "trace": ["inspector: skipped — missing or unreadable image"]}
        product = state.get("product", "")
        if product not in bands:
            return {"image_ok": True, "inspection": None, "inspection_error": None,
                    "trace": [f"inspector: skipped — no model for product {product!r}"]}
        try:
            insp = inspector.inspect(Path(image_path), product)
        except Exception as exc:
            return {"image_ok": True, "inspection": None, "inspection_error": f"{type(exc).__name__}: {exc}",
                    "trace": [f"inspector: FAILED — {type(exc).__name__}: {exc}"]}
        band = bands[product]
        return {"image_ok": True, "inspection": asdict(insp), "inspection_error": None,
                "trace": [f"inspector: score={insp.score:.4f} (band lo={band.lo:.4f}, hi={band.hi:.4f})"]}

    def decide_node(state: ClaimState) -> dict:
        if state.get("route") == UNKNOWN:
            d = Decision(Verdict.ESCALATE, ("router_failed",))
        else:
            insp = state.get("inspection")
            d = decide(image_ok=state.get("image_ok", False), product=state.get("product", ""),
                       inspection=Inspection(**insp) if insp else None,
                       inspection_error=state.get("inspection_error"), bands=bands)
        return {"decision": {"verdict": d.verdict.value, "reasons": list(d.reasons)},
                "trace": [f"policy: {d.verdict.value} {list(d.reasons)}"]}

    def cross_check_node(state: ClaimState) -> dict:
        c = cross_check(state["image_path"], state["product"], vlm)
        note = (f"cross_check[{c.source}, attempts={c.attempts}]: object_matches_product={c.object_matches_product}, "
                f"defect_visible={c.defect_visible} — {c.description}" if c.source == "vlm"
                else f"cross_check: unavailable — {list(c.problems)}")
        return {"cross_check": asdict(c), "trace": [note + " (advisory; verdict unchanged)"]}

    def after_decide(state: ClaimState) -> str:
        wants = (vlm is not None and state["decision"]["verdict"] == Verdict.ESCALATE.value
                 and state.get("inspection") is not None)  # inspected photo of a supported product
        return "cross_check" if wants else "respond"

    def respond_node(state: ClaimState) -> dict:
        dec = state["decision"]
        decision = Decision(Verdict(dec["verdict"]), tuple(dec["reasons"]))
        reply = write_reply(decision, state["claim_text"], state.get("product", ""), llm)
        note = f"responder[{reply.source}, attempts={reply.attempts}]"
        if reply.problems:
            note += f" post-check issues: {list(reply.problems)}"
        return {"reply": reply.text, "reply_source": reply.source, "trace": [note]}

    def handoff_node(state: ClaimState) -> dict:
        return {"reply": HANDOFF_REPLY, "reply_source": "fixed", "trace": ["handoff: claims analyst"]}

    def decline_node(state: ClaimState) -> dict:
        return {"reply": DECLINE_REPLY, "reply_source": "fixed", "trace": ["decline: out of scope"]}

    g = StateGraph(ClaimState)
    for name, fn in [("route", route_node), ("inspect", inspect_node), ("decide", decide_node),
                     ("cross_check", cross_check_node), ("respond", respond_node), ("handoff", handoff_node), ("decline", decline_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["route"], {
        "claim": "inspect", UNKNOWN: "decide", "analytics_question": "handoff", "out_of_scope": "decline"})
    g.add_edge("inspect", "decide")
    g.add_conditional_edges("decide", after_decide, {"cross_check": "cross_check", "respond": "respond"})
    g.add_edge("cross_check", "respond")
    for terminal in ("respond", "handoff", "decline"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer)
