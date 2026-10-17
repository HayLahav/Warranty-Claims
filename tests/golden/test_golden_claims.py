"""Golden claims through the full graph with the REAL PatchCore models and deployed bands.

The LLM is a deterministic stub (route=claim, reply echoes the required status
line): the LLM cannot influence verdicts, and live LLM behaviour is measured
separately in tests/eval/test_responder_live.py.

Caveat: the deployed band is calibrated on the full test set, so labelled
cases are in-sample for the threshold. These tests verify wiring and the
adversarial guards, not generalisation (see cv/evaluate.py for that).
Run with:  pytest -m golden
"""
import json
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from claims.graph import build_graph
from claims.inspector import PatchCoreInspector, load_bands

CASES_FILE = Path(__file__).with_name("claims.jsonl")
CASES = [json.loads(line) for line in CASES_FILE.read_text().splitlines() if line.strip()]
pytestmark = pytest.mark.golden


class EchoStatusLLM:
    """Router -> claim. Responder -> the exact status line it was told to use, plus a neutral body."""

    def invoke(self, messages):
        system = messages[0].content
        if "route messages" in system:
            return AIMessage(content='{"route": "claim", "reason": "golden"}')
        status = re.search(r"must be exactly: (.+)", system).group(1).strip()
        return AIMessage(content=f"{status}\nThank you for contacting us. Here is the outcome and the next step.")


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    if not Path("models").is_dir() or not Path("data/mvtec").is_dir():
        pytest.skip("models/ or data/mvtec/ missing")
    bands = load_bands()
    inspector = PatchCoreInspector(Path("models"), bands, tmp_path_factory.mktemp("heatmaps"))
    return build_graph(llm=EchoStatusLLM(), inspector=inspector, bands=bands)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden_claim(graph, case, tmp_path):
    image = case["image"]
    if image == "<corrupt>":
        image = str(tmp_path / "corrupt.png")
        Path(image).write_bytes(b"\x89PNG\r\n\x1a\n not really a png")
    elif image and not Path(image).exists():
        pytest.skip(f"{image} missing")

    out = graph.invoke({"claim_text": case["text"], "image_path": image, "product": case["product"]})

    verdict = out["decision"]["verdict"]
    assert verdict in case["expected"], f"{case['id']}: got {verdict} {out['decision']['reasons']}, trace={out['trace']}"
    if "reason" in case:
        assert out["decision"]["reasons"][0] == case["reason"]
    assert out["reply"].splitlines()[0].endswith(
        {"APPROVE": "APPROVED", "REJECT": "NOT APPROVED", "ESCALATE": "UNDER REVIEW",
         "NEEDS_INFO": "MORE INFORMATION NEEDED"}[verdict])
