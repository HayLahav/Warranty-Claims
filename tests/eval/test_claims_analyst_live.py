"""Live eval for the Claims Analyst (needs NEBIUS_API_KEY; a few cents).

Runs the same guarded ClaimsAnalyst the app uses. Per question:
  1. a tool was called and the answer passed the grounding check (every number appears in a tool result);
  2. the answer contains the expected facts, derived from the committed reports (not hard-coded).
Grounding alone is not enough: a model can quote correct numbers and draw the wrong conclusion from them
(it once called thread_top "most escalated" while quoting thread_top's 1/23), so facts are checked too.
Run with:  pytest -m live tests/eval/test_claims_analyst_live.py
"""
import os
import re
import uuid

import pytest
from dotenv import load_dotenv

from claims.analyst import ClaimsAnalyst, ClaimsData, numbers

load_dotenv(".env")
pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def analyst():
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")
    from agent.llm import get_llm
    return ClaimsAnalyst(get_llm())


@pytest.fixture(scope="module")
def data():
    return ClaimsData()


def expected_cases(data):
    m = data.metrics["policy_cv"]["screw"]
    oof = data.oof
    wrong = oof[((oof.truth == "good") & (oof.verdict == "APPROVE")) |
                ((oof.truth == "defective") & (oof.verdict == "REJECT"))]
    s = oof[(oof.category == "screw") & (oof.truth == "defective")]
    most_escalated = s.groupby("defect_type").verdict.apply(lambda v: (v == "ESCALATE").mean()).idxmax()
    return [
        ("How many screw claims were escalated to a human?", [str(m["escalation_rate"]["k"])]),
        ("What is the approve precision for screws?",
         [str(m["approve_precision"]["k"]), str(m["approve_precision"]["n"])]),
        ("Which automatic decisions were wrong?", [p.split("/test/")[1] for p in wrong.image]),
        ("Which screw defect type is escalated most often?", [most_escalated]),
        # regression: the model once quoted thread_side's escalation rate as its share of all screw images
        ("How many of the escalated screw claims were thread_side defects?",
         [str(int((s[s.defect_type == "thread_side"].verdict == "ESCALATE").sum()))]),
    ]


@pytest.mark.parametrize("i", range(5))
def test_answers_are_grounded_and_contain_expected_facts(analyst, data, i):
    question, facts = expected_cases(data)[i]
    r = analyst.ask(question, thread_id=uuid.uuid4().hex)
    assert r.tool_calls, f"no tool was called for {question!r}"
    assert r.verified, f"{question!r}: ungrounded numbers {r.ungrounded}:\n{r.text}"
    missing = [f for f in facts if f not in r.text]
    assert not missing, f"{question!r}: missing {missing} (attempts={r.attempts}):\n{r.text}"


def test_follow_up_uses_conversation_context_and_totals(analyst, data):
    thread = uuid.uuid4().hex
    first = analyst.ask("Show me the screw claims that were escalated.", thread)
    # the tool shows 10 rows (all good screws); the answer must not generalise them to all 54
    assert not re.search(r"\ball\b[^.\n]*\bgood\b", first.text, re.I), first.text
    answer = analyst.ask("Of those, how many were actually good screws?", thread).text
    good_escalated = int(((data.oof.category == "screw") & (data.oof.truth == "good")
                          & (data.oof.verdict == "ESCALATE")).sum())
    assert str(good_escalated) in answer, answer


def test_unrelated_question_is_declined(analyst):
    r = analyst.ask("Who won the last World Cup?", thread_id=uuid.uuid4().hex)
    assert not numbers(r.text) - {"1"}, r.text          # no invented facts
    assert re.search(r"claims|warranty", r.text, re.I), r.text


def test_error_explanation_does_not_put_auto_decided_images_in_the_review_band(analyst):
    # regression: the model once said both errors "sat in the escalation band"; they were decided automatically
    r = analyst.ask("Which automatic decisions were wrong, and why?", thread_id=uuid.uuid4().hex)
    assert not re.search(r"\b(sat|fell|were|was|lay|lie)\s+(with)?in\s+the\s+(escalation|review|uncertainty)\s+band",
                         r.text, re.I), r.text
