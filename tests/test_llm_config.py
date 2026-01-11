from __future__ import annotations

from pathlib import Path

from local_file_agent.llm import ModelConfig, get_model_params, select_endpoint


CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_config"
    / "base.yaml.example"
)


def test_endpoint_from_config(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_AGENT_MODEL_CONFIG", str(CONFIG_PATH))
    cfg = ModelConfig(CONFIG_PATH)
    endpoints = cfg.get_endpoints("deepseek-ai/deepseek-v3.2")
    assert endpoints, "Expected endpoints from config"
    assert endpoints[0].base_url == "https://integrate.api.nvidia.com/v1"
    assert endpoints[0].api_key == "YOUR_NVIDIA_API_KEY"


def test_select_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_AGENT_MODEL_CONFIG", str(CONFIG_PATH))
    endpoint = select_endpoint("deepseek-ai/deepseek-v3.2")
    assert endpoint is not None
    assert endpoint.base_url
    assert endpoint.api_key


def test_get_model_params(monkeypatch) -> None:
    """Test that model-specific params are correctly loaded from config."""
    monkeypatch.setenv("LOCAL_AGENT_MODEL_CONFIG", str(CONFIG_PATH))
    # Model with params configured
    params = get_model_params("deepseek-ai/deepseek-v3.2")
    assert "temperature" in params
    assert "top_p" in params
    assert isinstance(params["temperature"], (int, float))
    assert isinstance(params["top_p"], (int, float))

    # Non-existent model
    params = get_model_params("non-existent-model")
    assert params == {}
