"""Live LLM eval for the responder (needs NEBIUS_API_KEY; costs a few cents).

Measures a failure the deterministic post-check cannot see: the reply
attributing the customer's own description ("crack on the rim") to our
inspection, which only reports whether a defect exists and its area.
Before prompt rule 5 this happened in 3/10 replies; after, 0/10.
Run with:  pytest -m live tests/eval/test_responder_live.py
"""
import os
import re

import pytest
from dotenv import load_dotenv

from claims.policy import Decision, Verdict
from claims.responder import check_reply, write_reply

load_dotenv(".env")
pytestmark = pytest.mark.live

RUNS = 5
INSPECTION = r"inspect|review|confirm|found|identif|detect|examin"
ATTRIBUTED = r"you('ve)? (reported|mentioned|described|said|noted)|your (report|description|message)|as you (said|described)"
CASES = [
    (Decision(Verdict.APPROVE, ("defect_detected", "defect_area=19.6%")),
     "My new bottle arrived with a big crack on the rim. I want a replacement.", "bottle", r"\brim\b|crack"),
    (Decision(Verdict.APPROVE, ("defect_detected", "defect_area=4.1%")),
     "The screw thread is stripped near the tip, can I get a new one?", "screw", r"strip|tip|thread"),
    (Decision(Verdict.REJECT, ("no_defect_found",)),
     "The screw has a scratched head, refund please.", "screw", r"scratch"),
]


def inspection_hallucinations(reply: str, detail: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", "\n".join(reply.splitlines()[1:]))
    return [s for s in sentences if re.search(INSPECTION, s, re.I) and re.search(detail, s, re.I)
            and not re.search(ATTRIBUTED, s, re.I)]


@pytest.fixture(scope="module")
def llm():
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")
    from agent.llm import get_llm
    return get_llm()


@pytest.mark.parametrize("decision,text,product,detail", CASES)
def test_replies_are_consistent_and_do_not_invent_inspection_findings(llm, decision, text, product, detail):
    failures = []
    for _ in range(RUNS):
        r = write_reply(decision, text, product, llm)
        assert check_reply(r.text, decision.verdict) == []
        if r.source != "llm":
            failures.append(f"fell back to template: {r.problems}")
        failures += inspection_hallucinations(r.text, detail)
    assert not failures, failures
