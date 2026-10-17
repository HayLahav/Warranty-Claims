"""Shared presentation helpers. Colours come from .streamlit/config.toml; this module only adds the verdict
panel (the one custom element) and a few spacing rules."""
from __future__ import annotations

import html

import streamlit as st

# verdict -> (label, badge colour for st.badge, text colour, tint, edge)
STATUS = {
    "APPROVE": ("Approved", "green", "#1d6b4f", "#eef7f2", "#2f8a66"),
    "REJECT": ("Not approved", "violet", "#4b3f9e", "#f2f1fb", "#6a5fc4"),
    "ESCALATE": ("Under review", "orange", "#7a4d00", "#fdf6e6", "#c98a12"),
    "NEEDS_INFO": ("More information needed", "blue", "#2e5a87", "#eef3f8", "#4c78a8"),
}

_CSS = """
<style>
.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1200px; }
h1 { letter-spacing: -0.012em; margin-bottom: 0.1rem; }
.page-lede { color: #556270; font-size: 0.98rem; max-width: 72ch; margin: 0 0 1.4rem; line-height: 1.5; }
.vpanel { background: var(--tint); border: 1px solid #dde2e8; border-left: 5px solid var(--edge);
          border-radius: 8px; padding: 1.15rem 1.4rem 1.2rem; margin: 0.2rem 0 1.4rem; }
.vpanel-top { display: flex; flex-wrap: wrap; gap: 0.4rem 2.2rem; align-items: baseline; justify-content: space-between; }
.vpanel-verdict { font-size: 1.75rem; font-weight: 600; color: var(--ink); line-height: 1.15; }
.vpanel-headline { color: #334155; margin-top: 0.25rem; }
.vpanel-facts { display: flex; gap: 2rem; color: #556270; font-size: 0.86rem; }
.vpanel-facts b { display: block; color: #1c2733; font-size: 1.15rem; font-weight: 600; }
.band { margin-top: 1rem; }
.band-track { position: relative; height: 10px; border-radius: 5px; display: flex; overflow: hidden; }
.band-track > div { height: 100%; }
.band-marker { position: absolute; top: -5px; width: 3px; height: 20px; background: #1c2733; border-radius: 2px;
               box-shadow: 0 0 0 2px #ffffff; }
.band-labels { display: flex; justify-content: space-between; font-size: 0.78rem; color: #556270; margin-top: 0.4rem; }
.band-wrap { position: relative; }
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def lede(text: str) -> None:
    st.markdown(f'<p class="page-lede">{html.escape(text)}</p>', unsafe_allow_html=True)


def verdict_badge(verdict: str) -> None:
    label, color, *_ = STATUS[verdict]
    st.badge(label, color=color)


def verdict_panel(verdict: str, headline: str, score: float | None = None, band=None, area: float | None = None,
                  product: str = "") -> None:
    """Status callout: verdict, one-line reason, and (if inspected) the score drawn on the decision band."""
    label, _, ink, tint, edge = STATUS[verdict]
    facts = ""
    if score is not None:
        facts += f"<div>Anomaly score<b>{score:.1f}</b></div>"
    if area is not None:
        facts += f"<div>Flagged area<b>{area:.1%}</b></div>"
    gauge = ""
    if score is not None and band is not None:
        lo, hi = band.lo, band.hi
        span_lo, span_hi = min(score, lo) - 6, max(score, hi) + 6
        pct = lambda v: (v - span_lo) / (span_hi - span_lo) * 100  # noqa: E731
        marker = min(max(pct(score), 0.5), 99.5)
        middle = f"Specialist review {lo:.1f}–{hi:.1f}" if hi > lo else ""
        gauge = f"""
<div class="band"><div class="band-wrap">
  <div class="band-track">
    <div style="width:{pct(lo):.2f}%;background:#d9d5f2"></div>
    <div style="width:{pct(hi) - pct(lo):.2f}%;background:#f6e2b3"></div>
    <div style="flex:1;background:#cfe8dc"></div>
  </div>
  <div class="band-marker" style="left:calc({marker:.2f}% - 1.5px)"></div>
</div>
<div class="band-labels"><span>Reject below {lo:.1f}</span><span>{middle}</span><span>Approve from {hi:.1f}</span></div>
</div>"""
    title = f"{label}" + (f" <span style='font-weight:400;color:#556270;font-size:1rem'>{html.escape(product)} claim</span>"
                          if product else "")
    st.markdown(f"""
<div class="vpanel" style="--tint:{tint};--edge:{edge};--ink:{ink}">
  <div class="vpanel-top">
    <div><div class="vpanel-verdict">{title}</div><div class="vpanel-headline">{html.escape(headline)}</div></div>
    <div class="vpanel-facts">{facts}</div>
  </div>{gauge}
</div>""", unsafe_allow_html=True)
