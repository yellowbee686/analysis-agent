from __future__ import annotations

import logging

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType, ModelType

from local_file_agent.config import AppConfig
from local_file_agent.llm import build_openai_clients, select_endpoint
from local_file_agent.mcp import get_mcp_tools, is_mcp_connected
from local_file_agent.tools import LocalDocTools

logger = logging.getLogger(__name__)


SYSTEM_MESSAGE = """
You are a local file analysis assistant. Use the available local retrieval
tools before answering questions that depend on documents.
When answering, cite sources using [source: file_path#heading]. If nothing is
found, say so clearly.

You also have access to CBETA Buddhist Scripture tools (if enabled) for
searching and retrieving Buddhist texts. Use these tools when the user asks
about Buddhist scriptures, sutras, or related topics.
""".strip()


def _resolve_model_platform(name: str | None) -> ModelPlatformType:
    if not name:
        return ModelPlatformType.DEFAULT
    try:
        return ModelPlatformType.from_name(name)
    except ValueError:
        return ModelPlatformType.DEFAULT


def _resolve_model_type(name: str | None) -> ModelType | str:
    if not name:
        return ModelType.DEFAULT
    try:
        return ModelType.from_name(name)
    except ValueError:
        return name


def _sanitize_schema(schema: dict) -> dict:
    if isinstance(schema, list):
        return [_sanitize_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    schema = {key: _sanitize_schema(value) for key, value in schema.items()}

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        schema["type"] = non_null[0] if non_null else schema_type[0]

    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list) and schema[key]:
            candidates = [
                item
                for item in schema[key]
                if isinstance(item, dict)
                and item.get("type") not in (None, "null")
            ]
            schema = candidates[0] if candidates else schema[key][0]
            return _sanitize_schema(schema)

    return schema


def _wrap_tool(tool: FunctionTool) -> FunctionTool:
    original = tool.get_openai_tool_schema

    def _sanitized_schema():
        return _sanitize_schema(original())

    tool.get_openai_tool_schema = _sanitized_schema  # type: ignore[assignment]
    return tool


def _should_use_stream(model_name: str, config: AppConfig) -> bool:
    """Determine if streaming should be enabled for the given model.

    Returns False if model name matches any pattern in non_stream_patterns.
    """
    if not config.stream:
        return False

    patterns = [
        p.strip().lower()
        for p in config.non_stream_patterns.split(",")
        if p.strip()
    ]
    model_lower = model_name.lower()
    for pattern in patterns:
        if pattern in model_lower:
            logger.info(
                "Streaming disabled for model '%s' (matches pattern '%s')",
                model_name,
                pattern,
            )
            return False
    return True


def build_agent(tools: LocalDocTools, config: AppConfig) -> ChatAgent:
    logger.info(
        "Building agent with model_platform=%s, model_type=%s",
        config.model_platform,
        config.model_type,
    )
    model_platform = _resolve_model_platform(config.model_platform)
    model_type = _resolve_model_type(config.model_type)
    system_message = config.system_prompt or SYSTEM_MESSAGE
    model_type_name = (
        model_type.value if isinstance(model_type, ModelType) else model_type
    )

    # Determine streaming mode based on model name
    use_stream = _should_use_stream(str(model_type_name), config)
    logger.info("Stream mode for '%s': %s", model_type_name, use_stream)

    model_config_dict = {
        "max_tokens": config.max_tokens,
        "stream": use_stream,
    }
    logger.debug("Resolved model_type_name: %s", model_type_name)

    endpoint = select_endpoint(str(model_type_name))
    if not endpoint:
        logger.error("No endpoint found for model: %s", model_type_name)
    else:
        logger.info(
            "Using endpoint: base_url=%s, api_key=%s...%s",
            endpoint.base_url,
            endpoint.api_key[:5] if endpoint.api_key else "None",
            endpoint.api_key[-3:] if endpoint.api_key and len(endpoint.api_key) > 8 else "",
        )

    client = None
    async_client = None
    if endpoint:
        client, async_client = build_openai_clients(endpoint)

    logger.debug("Creating model via ModelFactory...")
    model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        api_key=endpoint.api_key if endpoint else None,
        url=endpoint.base_url if endpoint else None,
        client=client,
        async_client=async_client,
        model_config_dict=model_config_dict,
    )
    logger.debug("Model created: %s", type(model).__name__)

    sanitize_tools = (
        model_platform == ModelPlatformType.OPENAI_COMPATIBLE_MODEL
    )
    tool_list: list[FunctionTool] = [
        FunctionTool(tools.retrieve_local_docs),
        FunctionTool(tools.corpus_stats),
        FunctionTool(tools.list_docs),
    ]

    # Add MCP tools if enabled and connected
    if config.mcp_enabled and is_mcp_connected():
        mcp_tools = get_mcp_tools()
        logger.info("Adding %d MCP tools to agent", len(mcp_tools))
        tool_list.extend(mcp_tools)
    elif config.mcp_enabled:
        logger.warning("MCP is enabled but not connected, no MCP tools added")

    if sanitize_tools:
        tool_list = [_wrap_tool(tool) for tool in tool_list]

    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=tool_list,
    )
    agent.reset()
    logger.info("Agent built successfully")
    return agent
