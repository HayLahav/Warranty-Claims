"""Claims Analyst tools run on the committed evaluation outputs; assertions are consistency checks against
reports/metrics.json so they survive recalibration."""
import re

import pytest
from langchain_core.messages import AIMessage

from claims.analyst import ClaimsData, build_analyst, make_tools
from tests.stubs import tool_script_llm


@pytest.fixture(scope="module")
def data():
    return ClaimsData()


@pytest.fixture(scope="module")
def tools(data):
    return {t.name: t for t in make_tools(data)}


def ints(text):
    return [int(x) for x in re.findall(r"\b\d+\b", text)]


@pytest.mark.parametrize("product", ["bottle", "screw"])
def test_confusion_matrix_matches_metrics(tools, data, product):
    out = tools["confusion_matrix"].invoke({"product": product})
    m = data.metrics["policy_cv"][product]
    auto = m["n"] - m["escalation_rate"]["k"]
    assert f"decided automatically: {auto}/{m['n']}" in out
    wrong = m["false_approve_rate"]["k"] + m["missed_defect_rate"]["k"]
    assert f"automatic decisions correct: {auto - wrong}/{auto}" in out


def test_summary_quotes_metrics_counts(tools, data):
    out = tools["results_summary"].invoke({"product": "screw"})
    m = data.metrics["policy_cv"]["screw"]
    assert f"{m['escalation_rate']['k']}/{m['n']}" in out
    assert f"{m['approve_precision']['k']}/{m['approve_precision']['n']}" in out


def test_wrong_decisions_count_matches_metrics(tools, data):
    out = tools["wrong_decisions"].invoke({})
    expected = sum(data.metrics["policy_cv"][p][k]["k"] for p in data.products
                   for k in ("false_approve_rate", "missed_defect_rate"))
    assert out.startswith(f"{expected} wrong automatic decisions")


def test_defect_type_table_sums_to_test_set(tools, data):
    out = tools["verdicts_by_defect_type"].invoke({"product": "screw"})
    rows = [ints(line)[0] for line in out.splitlines()[2:]]          # skip the note and the column header
    assert sum(rows) == data.metrics["policy_cv"]["screw"]["n"]


def test_claim_details_requires_an_unambiguous_image(tools):
    assert "be more specific" in tools["claim_details"].invoke({"image": "good/005"})   # bottle and screw
    out = tools["claim_details"].invoke({"image": "screw/test/good/005.png"})
    assert "truth: good" in out and "verdict:" in out
    assert "No test image" in tools["claim_details"].invoke({"image": "hazelnut/1"})


def test_unknown_product_is_reported_not_guessed(tools):
    assert tools["confusion_matrix"].invoke({"product": "car"}).startswith("Unknown product")


def test_find_claims_filters_and_caps_rows(tools):
    out = tools["find_claims"].invoke({"product": "screw", "verdict": "escalate", "truth": "good", "limit": 500})
    total = ints(out.splitlines()[0])[0]
    shown = len(out.splitlines()) - 2
    assert shown == min(total, 50)
    assert "APPROVE" not in out and "REJECT" not in out.split("\n", 2)[2]


def test_agent_calls_a_tool_and_answers_from_it():
    llm = tool_script_llm(
        AIMessage(content="", tool_calls=[{"name": "confusion_matrix", "args": {"product": "screw"}, "id": "c1"}]),
        AIMessage(content="See the matrix above."))
    out = build_analyst(llm).invoke({"messages": [("user", "How did screw claims go?")]})
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert len(tool_msgs) == 1 and "decided automatically" in tool_msgs[0].content
    assert out["messages"][-1].content == "See the matrix above."


# ── grounding: every number in an answer must appear in a tool result ─────────────────────────────
def test_ungrounded_numbers_are_detected():
    from claims.analyst import ungrounded
    tools = "screw: escalated to a human: 54/160 (33.8%)"
    assert ungrounded("54 of 160 claims (33.8%) were escalated.", tools) == set()
    assert ungrounded("The escalation rate for screw is 12.5%.", tools) == {"12.5"}
    assert ungrounded("1,000 records", "1000 rows") == set()            # thousands separators
    assert ungrounded("33.80%", tools) == set()                          # trailing zeros


def test_find_claims_reports_the_breakdown_of_all_matches_not_just_shown_rows(tools):
    out = tools["find_claims"].invoke({"product": "screw", "verdict": "ESCALATE", "limit": 3})
    head = out.splitlines()[0]
    assert "good" in head and "defective" in head and "showing the 3" in head


def test_defect_type_table_includes_and_sorts_by_escalation_rate(tools):
    out = tools["verdicts_by_defect_type"].invoke({"product": "screw"})
    assert "pct_of_this_type_escalated" in out.splitlines()[1]
    rates = [float(line.split()[-2]) for line in out.splitlines()[2:]]
    assert rates == sorted(rates, reverse=True)


def test_analyst_retries_an_answer_with_invented_numbers():
    from claims.analyst import ClaimsAnalyst
    llm = tool_script_llm(
        AIMessage(content="The escalation rate for screw is 12.5%."),                       # no tool, invented
        AIMessage(content="", tool_calls=[{"name": "results_summary", "args": {"product": "screw"}, "id": "c1"}]),
        AIMessage(content="54/160 screw claims (33.8%) were escalated."))
    r = ClaimsAnalyst(llm).ask("How many screw claims were escalated?")
    assert (r.verified, r.attempts) == (True, 2)
    assert r.text == "54/160 screw claims (33.8%) were escalated."
    assert r.tool_calls == [("results_summary", {"product": "screw"})]


def test_analyst_flags_answers_that_stay_ungrounded():
    from claims.analyst import ClaimsAnalyst
    llm = tool_script_llm(AIMessage(content="About 12.5% were escalated."),
                          AIMessage(content="I believe it is 12.5%."))
    r = ClaimsAnalyst(llm).ask("How many screw claims were escalated?")
    assert not r.verified and r.ungrounded == {"12.5"}
    assert r.text.startswith("⚠") and "12.5" in r.text


def test_answers_without_numbers_need_no_tool():
    from claims.analyst import ClaimsAnalyst
    r = ClaimsAnalyst(tool_script_llm(AIMessage(content="I only answer questions about the claims evaluation."))).ask(
        "Who won the World Cup?")
    assert r.verified and r.attempts == 1


def test_wrong_decisions_explain_each_error_deterministically(tools, data):
    out = tools["wrong_decisions"].invoke({"product": "screw"})
    assert "decided automatically, outside the review band" in out
    for r in data.oof[(data.oof.category == "screw") & (data.oof.truth == "good") & (data.oof.verdict == "APPROVE")].itertuples():
        assert f"good product scored {r.score:.2f}, at or above this fold's approve threshold {r.band_hi:.2f}" in out
    for r in data.oof[(data.oof.category == "screw") & (data.oof.truth == "defective") & (data.oof.verdict == "REJECT")].itertuples():
        assert f"defective product scored {r.score:.2f}, below this fold's reject threshold {r.band_lo:.2f}" in out
