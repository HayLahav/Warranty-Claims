"""
Centralised LLM factory.

Priority:
  1. NEBIUS_API_KEY  → Nebius Token Factory (OpenAI-compatible)
       text agents (router, responder): Nemotron 3.5 Lightning  (NEBIUS_MODEL)
       vision cross-check:              GLM-5.3-Flash          (NEBIUS_VLM_MODEL)
  2. OPENAI_API_KEY  → standard OpenAI  (gpt-4o-mini by default)

All agent modules call get_llm() so switching providers never requires
touching individual files.
"""

import os
from langchain_openai import ChatOpenAI

# ── Nebius defaults ────────────────────────────────────────────────────────────
NEBIUS_BASE_URL    = "https://api.studio.nebius.com/v1/"
NEBIUS_MODEL       = "nvidia/Nemotron-3_5-Lightning"
NEBIUS_VLM_MODEL   = "zai-org/GLM-5.3-Flash"
# Lightning reasons by default (~700 hidden tokens, ~5 s per reply); with thinking
# off it answers in <1 s. Only the chat-template switch works: "/no_think" and
# reasoning_effort did not change token counts (measured 2026-10-14).
NO_THINKING_MODELS = {"nvidia/Nemotron-3_5-Lightning"}

# ── OpenAI fallback default ────────────────────────────────────────────────────
OPENAI_MODEL       = "gpt-4o-mini"


def get_llm(model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    """
    Return a LangChain ChatOpenAI instance pointed at the configured provider.

    Parameters
    ----------
    model       : Override the model name.  Pass None to use the provider default.
    temperature : Sampling temperature (default 0 = deterministic).
    """
    nebius_key = os.getenv("NEBIUS_API_KEY")

    if nebius_key:
        model = model or os.getenv("NEBIUS_MODEL", NEBIUS_MODEL)
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if model in NO_THINKING_MODELS else None
        return ChatOpenAI(
            model=model,
            api_key=nebius_key,
            base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
            temperature=temperature,
            extra_body=extra,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=model or OPENAI_MODEL,
            api_key=openai_key,
            temperature=temperature,
        )

    raise EnvironmentError(
        "No LLM provider configured. "
        "Set NEBIUS_API_KEY (for Nebius/Llama) or OPENAI_API_KEY (for OpenAI)."
    )


def get_vlm(temperature: float = 0) -> ChatOpenAI:
    """Vision-language model for the image cross-check (Nebius only)."""
    if not os.getenv("NEBIUS_API_KEY"):
        raise EnvironmentError("The vision cross-check needs NEBIUS_API_KEY.")
    return get_llm(model=os.getenv("NEBIUS_VLM_MODEL", NEBIUS_VLM_MODEL), temperature=temperature)


def active_provider() -> str:
    """Return a human-readable string describing the active LLM provider."""
    if os.getenv("NEBIUS_API_KEY"):
        model = os.getenv("NEBIUS_MODEL", NEBIUS_MODEL)
        return f"Nebius ({model})"
    if os.getenv("OPENAI_API_KEY"):
        return f"OpenAI ({OPENAI_MODEL})"
    return "unconfigured"
