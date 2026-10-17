"""VLM cross-check: an advisory second opinion for the human specialist on escalated claims.

It never changes the verdict. The VLM sees only the photo and the product on the order,
never the customer's message, so injected instructions or the customer's own description
cannot steer it. Malformed output is retried once, then reported as unavailable.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from claims.llm_utils import extract_json, message_text

MAX_SIDE = 1024
MAX_WORDS = 40

SYSTEM_PROMPT = """You are a quality-control inspector looking at a single product photo.
Answer only about what is visible in the photo. Reply with one JSON object and nothing else:
{"object_matches_product": true|false, "defect_visible": true|false, "description": "<at most 25 words>"}
- object_matches_product: the photo shows the stated product type.
- defect_visible: you can see damage, contamination, scratches, deformation or a manufacturing flaw.
- description: what you see; if a defect is visible, say where it is."""


@dataclass(frozen=True)
class CrossCheck:
    object_matches_product: bool | None
    defect_visible: bool | None
    description: str
    source: str                # "vlm" or "unavailable"
    attempts: int
    problems: tuple[str, ...]


def encode_image(path: Path, max_side: int = MAX_SIDE) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse(text: str) -> tuple[bool, bool, str]:
    obj = extract_json(text)
    match, defect, desc = obj.get("object_matches_product"), obj.get("defect_visible"), obj.get("description")
    if not isinstance(match, bool) or not isinstance(defect, bool):
        raise ValueError("object_matches_product and defect_visible must be booleans")
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError("description must be a non-empty string")
    return match, defect, " ".join(desc.split()[:MAX_WORDS])


def cross_check(image_path: str | Path, product: str, vlm, max_attempts: int = 2) -> CrossCheck:
    problems: list[str] = []
    try:
        url = encode_image(Path(image_path))
    except Exception as exc:
        return CrossCheck(None, None, "", "unavailable", 0, (f"image error: {type(exc).__name__}: {exc}",))
    messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=[{"type": "text", "text": f"Stated product: {product}"},
                                      {"type": "image_url", "image_url": {"url": url}}])]
    for attempt in range(1, max_attempts + 1):
        try:
            match, defect, desc = parse(message_text(vlm.invoke(messages)))
            return CrossCheck(match, defect, desc, "vlm", attempt, tuple(problems))
        except Exception as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
    return CrossCheck(None, None, "", "unavailable", max_attempts, tuple(problems))
