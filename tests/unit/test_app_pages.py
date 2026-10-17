"""Every Streamlit page must at least parse and render its initial state (no models or LLM needed)."""
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
PAGES = [ROOT / "streamlit_app.py", *sorted((ROOT / "app_pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_compiles(page):
    py_compile.compile(str(page), doraise=True)


def test_claims_page_renders_and_lists_calibrated_products():
    from streamlit.testing.v1 import AppTest
    import json
    # Start from the real entry point: run directly, app_pages/ would be first on sys.path and
    # `import claims` would load the page itself instead of the claims package.
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()  # Claims is the default page
    assert not at.exception
    at.sidebar.radio[0].set_value("Upload").run()
    assert not at.exception
    products = list(json.loads((ROOT / "config" / "thresholds.json").read_text()))
    assert at.sidebar.selectbox[0].options == products


@pytest.mark.skipif(not (ROOT / "data" / "mvtec").exists() or not list((ROOT / "reports" / "raw").glob("maps_*.npz")),
                    reason="needs MVTec data and anomaly maps")
@pytest.mark.parametrize("scale", ["fixed", "per-image"])
def test_gallery_renders_all_filters(scale):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    at.switch_page("app_pages/gallery.py").run()
    assert not at.exception
    radios = {r.label: r for r in at.sidebar.radio}
    radios["Colour scale"].set_value(scale).run()
    radios = {r.label: r for r in at.sidebar.radio}
    radios["Product"].set_value("screw").run()
    expected = {"Misses the defect (defects only)": "17 / 160",   # 119 - 102 screw defects (pointing game)
                "On the background": "15 / 160"}                  # heat_zones: 8 defective + 7 good
    for option, shown in expected.items():
        radios = {r.label: r for r in at.sidebar.radio}
        radios["Map peak (yellow ring)"].set_value(option).run()
        assert not at.exception
        assert at.metric[0].value == shown, option
    # background-peak cards each get a zoomed inset of the hotspot: 15 cards -> 15 tiles + 15 insets
    assert len(at.get("image")) == 30


def test_claims_analyst_page_renders_without_calling_the_llm():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    at.switch_page("app_pages/claims_analyst.py").run()
    assert not at.exception
    labels = [b.label for b in at.sidebar.button]
    assert "How many screw claims were escalated to a human?" in labels and "New conversation" in labels
