from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI


load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str
    weight: int = 1
    api_version: str | None = None


class ModelConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config = self._load()

    def _load(self) -> dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_endpoints(self, model_name: str) -> list[Endpoint]:
        """Get all valid endpoints for a model.

        Returns:
            List of Endpoint objects with resolved environment variables.
        """
        models = self._config.get("models", {})
        if model_name not in models:
            logger.debug(
                "Model '%s' not found in config. Available models: %s",
                model_name,
                list(models.keys()),
            )
            return []

        endpoints = []
        endpoint_configs = models[model_name].get("endpoints", [])
        logger.debug(
            "Processing %d endpoint config(s) for model '%s'",
            len(endpoint_configs),
            model_name,
        )

        for i, endpoint in enumerate(endpoint_configs):
            base_url = endpoint.get("base_url")
            base_url_env = endpoint.get("base_url_env")
            if base_url_env:
                base_url = os.environ.get(base_url_env, base_url)

            api_key = endpoint.get("api_key")
            api_key_env = endpoint.get("api_key_env")
            if api_key_env:
                api_key = os.environ.get(api_key_env, api_key or "")

            weight = endpoint.get("weight", 1)
            weight_env = endpoint.get("weight_env")
            if weight_env:
                weight_val = os.environ.get(weight_env)
                if weight_val is not None:
                    try:
                        weight = int(weight_val)
                    except ValueError:
                        weight = 1

            api_version = endpoint.get("api_version")
            api_version_env = endpoint.get("api_version_env")
            if api_version_env:
                api_version = os.environ.get(api_version_env, api_version)

            if not base_url or not api_key:
                logger.debug(
                    "Skipping endpoint[%d] for model '%s': base_url=%s, api_key=%s",
                    i,
                    model_name,
                    "set" if base_url else "missing",
                    "set" if api_key else "missing",
                )
                continue

            # Skip endpoints with weight=0 (disabled)
            if weight <= 0:
                logger.debug(
                    "Skipping endpoint[%d] for model '%s': weight=%d (disabled)",
                    i,
                    model_name,
                    weight,
                )
                continue

            endpoints.append(
                Endpoint(
                    base_url=base_url,
                    api_key=api_key,
                    weight=int(weight),
                    api_version=api_version,
                )
            )
            logger.debug(
                "Added endpoint[%d] for model '%s': base_url=%s, weight=%d",
                i,
                model_name,
                base_url,
                weight,
            )

        return endpoints

    def list_models(self) -> list[str]:
        models = self._config.get("models", {})
        return sorted(models.keys())


_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "models"
    / "model_config"
    / "base.yaml"
)
_MODEL_CONFIG = ModelConfig(_DEFAULT_CONFIG_PATH)


def select_endpoint(model_name: str) -> Optional[Endpoint]:
    """Select an endpoint for the given model using weighted random selection."""
    logger.debug("Selecting endpoint for model: %s", model_name)
    endpoints = _MODEL_CONFIG.get_endpoints(model_name)

    if not endpoints:
        logger.warning(
            "No valid endpoints found for model '%s'. "
            "Check that the required environment variables are set.",
            model_name,
        )
        return None

    logger.debug("Found %d endpoint(s) for model '%s'", len(endpoints), model_name)

    if len(endpoints) == 1:
        selected = endpoints[0]
    else:
        weights = [endpoint.weight for endpoint in endpoints]
        idx = random.choices(range(len(endpoints)), weights=weights, k=1)[0]
        selected = endpoints[idx]

    logger.info(
        "Selected endpoint for model '%s': base_url=%s",
        model_name,
        selected.base_url,
    )
    return selected


def _should_use_azure(base_url: str) -> bool:
    """Determine if Azure client should be used based on the base URL.

    Returns False for known OpenAI-compatible APIs (NVIDIA, etc.).
    """
    openai_compatible_hosts = [
        "integrate.api.nvidia.com",  # NVIDIA API
        "api.openai.com",  # OpenAI
    ]
    for host in openai_compatible_hosts:
        if host in base_url:
            return False
    return True


def build_openai_clients(
    endpoint: Endpoint,
    use_azure: bool | None = None,
) -> tuple[AzureOpenAI | OpenAI, AsyncAzureOpenAI | AsyncOpenAI]:
    """Build OpenAI clients for the given endpoint.

    Args:
        endpoint: The endpoint configuration.
        use_azure: If True, use AzureOpenAI client; if False, use standard OpenAI client.
                   If None, auto-detect based on base_url.

    Returns:
        Tuple of (sync_client, async_client).
    """
    if use_azure is None:
        use_azure = _should_use_azure(endpoint.base_url)

    logger.debug(
        "Building %s clients for endpoint: base_url=%s, api_key=%s...%s",
        "AzureOpenAI" if use_azure else "OpenAI",
        endpoint.base_url,
        endpoint.api_key[:5] if endpoint.api_key else "None",
        endpoint.api_key[-3:] if endpoint.api_key and len(endpoint.api_key) > 8 else "",
    )

    # Set a reasonable timeout to avoid hanging requests
    timeout = float(os.environ.get("LOCAL_AGENT_REQUEST_TIMEOUT", "120"))

    if use_azure:
        api_version = endpoint.api_version or os.environ.get(
            "LOCAL_AGENT_AZURE_API_VERSION", "2024-12-01-preview"
        )
        logger.debug("Using Azure API version: %s", api_version)

        # Use base_url instead of azure_endpoint because the configured URL
        # is already a full path (not just the Azure resource endpoint).
        # azure_endpoint would append /openai/... which breaks the URL.
        client = AzureOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            api_version=api_version,
            timeout=timeout,
        )
        async_client = AsyncAzureOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            api_version=api_version,
            timeout=timeout,
        )
    else:
        client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=timeout,
        )
        async_client = AsyncOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=timeout,
        )

    logger.debug("Clients created successfully with timeout=%s", timeout)
    return client, async_client


def list_models() -> list[str]:
    return _MODEL_CONFIG.list_models()
