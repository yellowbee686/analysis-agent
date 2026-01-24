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
    use_azure: bool | None = None


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
            List of Endpoint objects from config.
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
            api_key = endpoint.get("api_key")
            weight = endpoint.get("weight", 1)
            try:
                weight = int(weight)
            except (TypeError, ValueError):
                weight = 1
            api_version = endpoint.get("api_version")
            use_azure = endpoint.get("use_azure")

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
                    weight=weight,
                    api_version=api_version,
                    use_azure=use_azure,
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

    def get_model_params(self, model_name: str) -> dict:
        """Get model-specific parameters (temperature, top_p, etc.).

        Returns:
            Dictionary of model parameters. Empty dict if not configured.
        """
        models = self._config.get("models", {})
        if model_name not in models:
            return {}
        return models[model_name].get("params", {})

    def get_context_window(self, model_name: str) -> int | None:
        """Get context window size for a model.

        Returns:
            Context window size in tokens, or None if not configured.
        """
        models = self._config.get("models", {})
        if model_name not in models:
            return None
        return models[model_name].get("context_window")

    def list_models(self) -> list[str]:
        models = self._config.get("models", {})
        return sorted(models.keys())

def _resolve_config_path() -> Path:
    env_path = os.environ.get("LOCAL_AGENT_MODEL_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    root_dir = Path(__file__).resolve().parent.parent.parent
    base_path = root_dir / "models" / "model_config" / "base.yaml"
    if base_path.exists():
        return base_path
    return base_path.with_name("base.yaml.example")


_MODEL_CONFIG: ModelConfig | None = None
_MODEL_CONFIG_PATH: Path | None = None


def _get_model_config() -> ModelConfig:
    global _MODEL_CONFIG, _MODEL_CONFIG_PATH
    config_path = _resolve_config_path()
    if _MODEL_CONFIG is None or _MODEL_CONFIG_PATH != config_path:
        _MODEL_CONFIG = ModelConfig(config_path)
        _MODEL_CONFIG_PATH = config_path
    return _MODEL_CONFIG


def select_endpoint(model_name: str) -> Optional[Endpoint]:
    """Select an endpoint for the given model using weighted random selection."""
    logger.debug("Selecting endpoint for model: %s", model_name)
    endpoints = _get_model_config().get_endpoints(model_name)

    if not endpoints:
        logger.warning(
            "No valid endpoints found for model '%s'. "
            "Check that the model config file has valid endpoints.",
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
    return _get_model_config().list_models()


def get_model_params(model_name: str) -> dict:
    """Get model-specific parameters from config.

    Returns:
        Dictionary of model parameters (temperature, top_p, etc.).
    """
    return _get_model_config().get_model_params(model_name)


def get_context_window(model_name: str) -> int | None:
    """Get context window size for a model from config.

    This value can be used by agents to size their token limits. If not
    configured, the model's default limit is used.

    Returns:
        Context window size in tokens, or None if not configured.
    """
    return _get_model_config().get_context_window(model_name)


def _normalize_message_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                continue
            parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "content" in value:
            return str(value["content"])
    return ""


def extract_message_text(message: object) -> str:
    if message is None:
        return ""
    if isinstance(message, dict):
        content = _normalize_message_text(message.get("content"))
        if content:
            return content
        for key in ("reasoning_content", "reasoning"):
            text = _normalize_message_text(message.get(key))
            if text:
                return text
        return ""

    content = _normalize_message_text(getattr(message, "content", None))
    if content:
        return content
    for attr in ("reasoning_content", "reasoning"):
        text = _normalize_message_text(getattr(message, attr, None))
        if text:
            return text
    return ""


def extract_delta_text(delta: object) -> str:
    if delta is None:
        return ""
    if isinstance(delta, dict):
        content = _normalize_message_text(delta.get("content"))
        if content:
            return content
        for key in ("reasoning_content", "reasoning"):
            text = _normalize_message_text(delta.get(key))
            if text:
                return text
        return ""

    content = _normalize_message_text(getattr(delta, "content", None))
    if content:
        return content
    for attr in ("reasoning_content", "reasoning"):
        text = _normalize_message_text(getattr(delta, attr, None))
        if text:
            return text
    return ""


def extract_response_text(response: object) -> str:
    if response is None:
        return ""
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        text = extract_message_text(message)
        if text:
            return text
    try:
        dumped = response.model_dump()
    except Exception:
        return ""
    choices = dumped.get("choices") if isinstance(dumped, dict) else None
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    return extract_message_text(message)
