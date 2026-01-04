from __future__ import annotations

import os
from pathlib import Path

from local_file_agent.llm import ModelConfig, select_endpoint


def test_endpoint_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_3_PRO_BASE_URL_1", "https://example.com/api")
    monkeypatch.setenv("GEMINI_3_PRO_API_KEY_1", "test-key")
    monkeypatch.setenv("GEMINI_3_PRO_WEIGHT_1", "2")

    config_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "model_config"
        / "base.yaml"
    )
    cfg = ModelConfig(config_path)
    endpoints = cfg.get_endpoints("gemini-3-pro-preview-new")
    assert endpoints, "Expected endpoints from env config"
    assert endpoints[0].base_url == "https://example.com/api"
    assert endpoints[0].api_key == "test-key"
    assert endpoints[0].weight == 2


def test_select_endpoint() -> None:
    endpoint = select_endpoint("gemini-3-pro-preview-new")
    assert endpoint is not None
    assert endpoint.base_url
    assert endpoint.api_key
