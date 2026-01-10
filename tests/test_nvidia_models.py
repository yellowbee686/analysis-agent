"""Test NVIDIA API models configured in base.yaml.

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

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from local_file_agent.llm import (
    build_openai_clients,
    get_model_params,
    select_endpoint,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Models to test from base.yaml
NVIDIA_MODELS = [
    "deepseek-ai/deepseek-v3.2",  # Thinking enabled via extra_body
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


# ============================================================================
# Pytest tests (fast, don't require API)
# ============================================================================

def test_nvidia_model_endpoints_configured() -> None:
    """Test that NVIDIA models have endpoints configured."""
    for model in NVIDIA_MODELS:
        endpoint = select_endpoint(model)
        assert endpoint is not None, f"No endpoint for {model}"
        assert endpoint.base_url == "https://integrate.api.nvidia.com/v1"


def test_nvidia_model_params_configured() -> None:
    """Test that NVIDIA models have custom params configured."""
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
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set"
)
@pytest.mark.parametrize("model_name", NVIDIA_MODELS[:2])  # Only test fast models
def test_nvidia_model_api_call(model_name: str) -> None:
    """Test actual API call to NVIDIA models (requires API key)."""
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
    
    # Check if NVIDIA API key is set
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        logger.error("NVIDIA_API_KEY environment variable is not set!")
        logger.info("Please set it with: export NVIDIA_API_KEY=your_key_here")
        sys.exit(1)
    
    logger.info(f"NVIDIA_API_KEY is set: {nvidia_key[:8]}...{nvidia_key[-4:]}")
    
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
