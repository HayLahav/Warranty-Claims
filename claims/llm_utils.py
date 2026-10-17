"""Helpers for parsing chat-model output across providers."""
from __future__ import annotations

import json
import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def message_text(response) -> str:
    """Content of an AIMessage (or a plain string), with reasoning blocks removed."""
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return _THINK.sub("", str(content)).strip()


def extract_json(text: str) -> dict:
    """First JSON object in text; tolerates code fences and surrounding prose."""
    text = _FENCE.sub("", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("model output JSON is not an object")
    return obj
