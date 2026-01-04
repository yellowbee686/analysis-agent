from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI


load_dotenv()


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
        models = self._config.get("models", {})
        if model_name not in models:
            return []
        endpoints = []
        for endpoint in models[model_name].get("endpoints", []):
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
                continue
            endpoints.append(
                Endpoint(
                    base_url=base_url,
                    api_key=api_key,
                    weight=int(weight),
                    api_version=api_version,
                )
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
    endpoints = _MODEL_CONFIG.get_endpoints(model_name)
    if not endpoints:
        return None
    if len(endpoints) == 1:
        return endpoints[0]
    weights = [endpoint.weight for endpoint in endpoints]
    idx = random.choices(range(len(endpoints)), weights=weights, k=1)[0]
    return endpoints[idx]


def build_openai_clients(endpoint: Endpoint) -> tuple[AzureOpenAI, AsyncAzureOpenAI]:
    api_version = endpoint.api_version or os.environ.get(
        "LOCAL_AGENT_AZURE_API_VERSION", "2024-03-01-preview"
    )
    client = AzureOpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        api_version=api_version,
    )
    async_client = AsyncAzureOpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        api_version=api_version,
    )
    return client, async_client


def list_models() -> list[str]:
    return _MODEL_CONFIG.list_models()
