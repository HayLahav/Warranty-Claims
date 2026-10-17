import io
import sys


def test_router_log_line_survives_a_narrow_console_encoding(monkeypatch):
    # Windows: when stdout is a pipe or file, Python encodes as cp1252; LLM text such as a non-breaking
    # hyphen (U+2011) used to raise UnicodeEncodeError and crash the whole question.
    from agent.graph import _console
    buf = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buf, encoding="cp1252"))
    _console("[Router] STRUCTURED \u2014 a data\u2011driven question")
    sys.stdout.flush()
    assert buf.getvalue().startswith(b"[Router] STRUCTURED")
