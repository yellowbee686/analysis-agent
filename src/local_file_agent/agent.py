from __future__ import annotations

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType, ModelType

from local_file_agent.config import AppConfig
from local_file_agent.llm import build_openai_clients, select_endpoint
from local_file_agent.tools import LocalDocTools


SYSTEM_MESSAGE = """
You are a local file analysis assistant. Use the available local retrieval
tools before answering questions that depend on documents.
When answering, cite sources using [source: file_path#heading]. If nothing is
found, say so clearly.
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


def build_agent(tools: LocalDocTools, config: AppConfig) -> ChatAgent:
    model_platform = _resolve_model_platform(config.model_platform)
    model_type = _resolve_model_type(config.model_type)
    system_message = config.system_prompt or SYSTEM_MESSAGE
    model_config_dict = {
        "max_tokens": config.max_tokens,
        "stream": config.stream,
    }
    model_type_name = (
        model_type.value if isinstance(model_type, ModelType) else model_type
    )
    endpoint = select_endpoint(str(model_type_name))
    client = None
    async_client = None
    if endpoint:
        client, async_client = build_openai_clients(endpoint)
    model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        api_key=endpoint.api_key if endpoint else None,
        url=endpoint.base_url if endpoint else None,
        client=client,
        async_client=async_client,
        model_config_dict=model_config_dict,
    )
    sanitize_tools = (
        model_platform == ModelPlatformType.OPENAI_COMPATIBLE_MODEL
    )
    tool_list = [
        FunctionTool(tools.retrieve_local_docs),
        FunctionTool(tools.corpus_stats),
        FunctionTool(tools.list_docs),
    ]
    if sanitize_tools:
        tool_list = [_wrap_tool(tool) for tool in tool_list]
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=tool_list,
    )
    agent.reset()
    return agent
