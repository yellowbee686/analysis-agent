"""Test NVIDIA API models configured in base.yaml / base.yaml.example.

This module provides both pytest-compatible tests and a standalone script
for testing NVIDIA API model connectivity and parameter passing.

Usage:
    # Run as pytest (fast, uses mocking by default)
    pytest tests/test_nvidia_models.py -v
    
    # Run standalone with real API calls
    python tests/test_nvidia_models.py
    
    # Test specific model
    python tests/test_nvidia_models.py deepseek-ai/deepseek-v3.2
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from local_file_agent.llm import (
    build_openai_clients,
    get_model_params,
    select_endpoint,
)


EXAMPLE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_config"
    / "base.yaml.example"
)
PRIVATE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_config"
    / "base.yaml"
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Models to test from base.yaml / base.yaml.example
NVIDIA_MODELS = [
    "deepseek-ai/deepseek-v3.2",  # Thinking: extra_body.chat_template_kwargs.thinking=true
    "minimaxai/minimax-m2.1",      # Has built-in thinking (<think> tags in content)
    "z-ai/glm4.7",                 # May have thinking
    "openai/gpt-oss-120b",         # Has reasoning_effort param
]

# Models with built-in thinking/reasoning capability
REASONING_MODELS = [
    "deepseek-ai/deepseek-v3.2",  # Returns reasoning_content when enabled
    "minimaxai/minimax-m2.1",
    "z-ai/glm4.7",
    "openai/gpt-oss-120b",
]

TEST_MESSAGE = "Hello! Please respond with 'OK' and nothing else."


def _has_real_nvidia_key(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = data.get("models", {})
    for model in NVIDIA_MODELS:
        endpoints = models.get(model, {}).get("endpoints", [])
        for endpoint in endpoints:
            api_key = endpoint.get("api_key")
            if api_key and api_key != "YOUR_NVIDIA_API_KEY":
                return True
    return False


HAS_PRIVATE_NVIDIA_KEY = _has_real_nvidia_key(PRIVATE_CONFIG_PATH)


# ============================================================================
# Pytest tests (fast, don't require API)
# ============================================================================

def test_nvidia_model_endpoints_configured(monkeypatch) -> None:
    """Test that NVIDIA models have endpoints configured."""
    monkeypatch.setenv("LOCAL_AGENT_MODEL_CONFIG", str(EXAMPLE_CONFIG_PATH))
    for model in NVIDIA_MODELS:
        endpoint = select_endpoint(model)
        assert endpoint is not None, f"No endpoint for {model}"
        assert endpoint.base_url == "https://integrate.api.nvidia.com/v1"
        assert endpoint.api_key == "YOUR_NVIDIA_API_KEY"


def test_nvidia_model_params_configured(monkeypatch) -> None:
    """Test that NVIDIA models have custom params configured."""
    monkeypatch.setenv("LOCAL_AGENT_MODEL_CONFIG", str(EXAMPLE_CONFIG_PATH))
    for model in NVIDIA_MODELS:
        params = get_model_params(model)
        assert "temperature" in params, f"No temperature for {model}"
        assert "top_p" in params, f"No top_p for {model}"
        assert 0 <= params["temperature"] <= 2, f"Invalid temperature for {model}"
        assert 0 <= params["top_p"] <= 1, f"Invalid top_p for {model}"


# ============================================================================
# Integration tests (require API key, slower)
# ============================================================================

@pytest.mark.skipif(
    not HAS_PRIVATE_NVIDIA_KEY,
    reason="Private base.yaml with a real NVIDIA API key not found",
)
@pytest.mark.parametrize("model_name", NVIDIA_MODELS[:2])  # Only test fast models
def test_nvidia_model_api_call(model_name: str, monkeypatch) -> None:
    """Test actual API call to NVIDIA models (requires API key)."""
    monkeypatch.setenv("LOCAL_AGENT_MODEL_CONFIG", str(PRIVATE_CONFIG_PATH))
    endpoint = select_endpoint(model_name)
    assert endpoint is not None
    
    client, _ = build_openai_clients(endpoint)
    params = get_model_params(model_name)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": TEST_MESSAGE}],
        max_tokens=50,
        **params,
    )
    
    assert response.choices
    assert response.choices[0].message.content


# ============================================================================
# Standalone test runner
# ============================================================================

def run_model_test(model_name: str, timeout: float = 60) -> tuple[bool, str]:
    """Test if a model can generate a basic response (standalone helper).
    
    Returns:
        Tuple of (success, message)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info("=" * 60)
    
    endpoint = select_endpoint(model_name)
    if not endpoint:
        return False, f"No endpoint found for model '{model_name}'"
    
    logger.info(f"Endpoint: {endpoint.base_url}")
    logger.info(f"API Key: {endpoint.api_key[:8]}...{endpoint.api_key[-4:]}")
    
    # Get model-specific params
    params = get_model_params(model_name)
    logger.info(f"Model params: {params}")
    
    try:
        from openai import OpenAI
        
        # Build client with custom timeout
        client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=timeout,
        )
        
        # Test API call with model-specific params
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": TEST_MESSAGE}],
            max_tokens=50,
            **params,
        )
        
        content = response.choices[0].message.content
        logger.info(f"Response: {content}")
        logger.info(f"Usage: {response.usage}")
        
        return True, content or "Empty response"
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False, str(e)


def main():
    """Run model tests as standalone script."""
    from dotenv import load_dotenv
    load_dotenv()

    config_path = Path(
        os.environ.get(
            "LOCAL_AGENT_MODEL_CONFIG",
            "models/model_config/base.yaml",
        )
    ).expanduser()
    if not config_path.exists():
        config_path = config_path.with_name("base.yaml.example")

    os.environ["LOCAL_AGENT_MODEL_CONFIG"] = str(config_path)
    if not _has_real_nvidia_key(config_path):
        logger.error(
            "No real NVIDIA API key found in %s. Update base.yaml with your key.",
            config_path,
        )
        sys.exit(1)
    
    # Allow testing specific model via CLI arg
    models_to_test = NVIDIA_MODELS
    if len(sys.argv) > 1:
        models_to_test = sys.argv[1:]
        logger.info(f"Testing specific models: {models_to_test}")
    
    results: dict[str, tuple[bool, str]] = {}
    
    # Test each model
    for model in models_to_test:
        success, msg = run_model_test(model, timeout=90)
        results[model] = (success, msg)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model, (success, msg) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {model}")
        if not success:
            print(f"       Error: {msg[:100]}...")
    
    # Return exit code based on results
    failed = sum(1 for s, _ in results.values() if not s)
    sys.exit(failed)


if __name__ == "__main__":
    main()
