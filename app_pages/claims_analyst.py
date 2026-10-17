"""Claims Analyst page: chat about the warranty-claims evaluation results (claims/analyst.py)."""
from __future__ import annotations

import uuid

import streamlit as st

from app_pages._style import inject, lede

SAMPLES = [
    "How many screw claims were escalated to a human?",
    "Show me the confusion matrix for screws.",
    "Which automatic decisions were wrong, and why?",
    "Which screw defect type is escalated most often?",
    "What are the decision thresholds?",
    "How reliable is the VLM cross-check?",
    "Tell me about screw/test/thread_side/000.png",
    "Of the escalated screw claims, how many were actually good?",
]

inject()
st.title("Claims analyst")
lede("Ask about the claims system's test results for bottles and screws. "
     "The language model chooses which data to look up and writes the answer; every number comes from the "
     "result files, and any number it cannot trace back to them is flagged.")


@st.cache_resource(show_spinner="Starting the analyst…")
def _analyst():
    from agent.llm import get_llm
    from claims.analyst import ClaimsAnalyst
    return ClaimsAnalyst(get_llm())


if "analyst_thread" not in st.session_state:
    st.session_state.analyst_thread = uuid.uuid4().hex
    st.session_state.analyst_chat = []

with st.sidebar:
    st.subheader("Example questions")
    picked = None
    for q in SAMPLES:
        if st.button(q, use_container_width=True):
            picked = q
    if st.button("New conversation", type="secondary", use_container_width=True):
        st.session_state.analyst_thread = uuid.uuid4().hex
        st.session_state.analyst_chat = []
        st.rerun()

AVATAR = {"user": ":material/person:", "assistant": ":material/query_stats:"}
for turn in st.session_state.analyst_chat:
    with st.chat_message(turn["role"], avatar=AVATAR[turn["role"]]):
        st.markdown(turn["text"])
        if turn.get("tools"):
            with st.expander(f"Tools used ({len(turn['tools'])})"):
                for name, args in turn["tools"]:
                    st.markdown(f"- `{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})`")

question = st.chat_input("Ask about the claims results…") or picked
if question:
    st.session_state.analyst_chat.append({"role": "user", "text": question})
    with st.chat_message("user", avatar=AVATAR["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        with st.spinner("Looking it up…"):
            try:
                r = _analyst().ask(question, thread_id=st.session_state.analyst_thread)
                text, tools = r.text, r.tool_calls
                note = "" if r.verified else "unverified"
            except Exception as exc:  # network / provider failure: say so, never guess
                text, tools, note = f"The analyst is unavailable right now ({type(exc).__name__}). Please try again.", [], "error"
        st.markdown(text)
        if tools:
            with st.expander(f"Tools used ({len(tools)})"):
                for name, args in tools:
                    st.markdown(f"- `{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})`")
        if not note:
            st.caption("Every number above was traced to the result files.")
    st.session_state.analyst_chat.append({"role": "assistant", "text": text, "tools": tools})
