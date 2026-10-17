from agent import llm


def test_text_agents_use_lightning(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    assert llm.get_llm().model_name == "nvidia/Nemotron-3_5-Lightning"


def test_vision_cross_check_uses_glm_flash(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.delenv("NEBIUS_VLM_MODEL", raising=False)
    vlm = llm.get_vlm()
    assert vlm.model_name == "zai-org/GLM-5.3-Flash"
    assert vlm.openai_api_base == llm.NEBIUS_BASE_URL


def test_env_overrides_win(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "a/text")
    monkeypatch.setenv("NEBIUS_VLM_MODEL", "b/vision")
    assert (llm.get_llm().model_name, llm.get_vlm().model_name) == ("a/text", "b/vision")


def test_vlm_requires_nebius(monkeypatch):
    import pytest
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        llm.get_vlm()


def test_thinking_disabled_only_for_lightning(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    assert llm.get_llm().extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm.get_vlm().extra_body is None
