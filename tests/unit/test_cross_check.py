import numpy as np
import pytest
from PIL import Image

from claims.cross_check import cross_check, parse
from tests.stubs import StubLLM

GOOD = '{"object_matches_product": true, "defect_visible": true, "description": "Bent tip near the point."}'


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "photo.png"
    Image.fromarray(np.full((2000, 1500), 120, np.uint8)).save(p)  # large greyscale, like MVTec screw
    return p


def test_valid_answer_is_used(photo):
    r = cross_check(photo, "screw", StubLLM(GOOD))
    assert (r.object_matches_product, r.defect_visible, r.source, r.attempts) == (True, True, "vlm", 1)
    assert r.description == "Bent tip near the point."


def test_vlm_sees_only_the_photo_and_product_never_customer_text(photo):
    vlm = StubLLM(GOOD)
    cross_check(photo, "screw", vlm)
    system, human = vlm.prompts[0]
    parts = human.content
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[0]["text"] == "Stated product: screw"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_is_downscaled_and_converted(photo):
    import base64, io
    vlm = StubLLM(GOOD)
    cross_check(photo, "screw", vlm)
    data = vlm.prompts[0][1].content[1]["image_url"]["url"].split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    assert max(img.size) == 1024 and img.mode == "RGB"


@pytest.mark.parametrize("bad", [
    "I think it is damaged.",
    '{"object_matches_product": "yes", "defect_visible": true, "description": "x"}',
    '{"object_matches_product": true, "defect_visible": null, "description": "x"}',
    '{"object_matches_product": true, "defect_visible": false, "description": ""}',
    '[true, false]',
])
def test_malformed_answer_is_retried_then_reported_unavailable(photo, bad):
    r = cross_check(photo, "screw", StubLLM(bad))
    assert (r.source, r.attempts, r.defect_visible, r.object_matches_product) == ("unavailable", 2, None, None)
    assert len(r.problems) == 2


def test_retry_recovers(photo):
    r = cross_check(photo, "screw", StubLLM("not json", GOOD))
    assert (r.source, r.attempts) == ("vlm", 2)


def test_api_error_and_unreadable_image_are_unavailable(tmp_path, photo):
    assert cross_check(photo, "screw", StubLLM(ConnectionError("down"))).source == "unavailable"
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    vlm = StubLLM(GOOD)
    r = cross_check(bad, "screw", vlm)
    assert r.source == "unavailable" and vlm.prompts == []


def test_long_description_is_truncated():
    _, _, desc = parse('{"object_matches_product": true, "defect_visible": false, "description": "%s"}' % ("w " * 100))
    assert len(desc.split()) == 40


def test_fenced_and_think_wrapped_json_parses(photo):
    r = cross_check(photo, "screw", StubLLM("<think>hmm</think>\n```json\n" + GOOD + "\n```"))
    assert r.source == "vlm"
