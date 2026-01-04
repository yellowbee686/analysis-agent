from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_file_agent.agent import build_agent  # noqa: E402
from local_file_agent.config import load_config  # noqa: E402
from local_file_agent.indexer import LocalIndex  # noqa: E402
from local_file_agent.llm import list_models  # noqa: E402
from camel.agents.chat_agent import StreamingChatAgentResponse  # noqa: E402
from local_file_agent.tools import LocalDocTools  # noqa: E402
from camel.types import ModelPlatformType  # noqa: E402


@st.cache_resource(show_spinner="Indexing markdown files...")
def build_index(
    data_dir: str, chunk_max_chars: int, snippet_chars: int
) -> LocalIndex:
    index = LocalIndex(
        Path(data_dir),
        chunk_max_chars=chunk_max_chars,
        snippet_chars=snippet_chars,
    )
    index.build()
    return index


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "agent" not in st.session_state:
        st.session_state["agent"] = None
    if "agent_data_dir" not in st.session_state:
        st.session_state["agent_data_dir"] = ""
    if "agent_chunk_max" not in st.session_state:
        st.session_state["agent_chunk_max"] = 0
    if "agent_model_platform" not in st.session_state:
        st.session_state["agent_model_platform"] = ""
    if "agent_model_type" not in st.session_state:
        st.session_state["agent_model_type"] = ""
    if "agent_max_tokens" not in st.session_state:
        st.session_state["agent_max_tokens"] = 0
    if "agent_stream" not in st.session_state:
        st.session_state["agent_stream"] = True


def main() -> None:
    st.set_page_config(page_title="Local File Agent", layout="wide")
    config = load_config()
    ensure_session_state()

    st.sidebar.header("Index Settings")
    data_dir = st.sidebar.text_input(
        "Data directory", value=str(config.data_dir)
    )
    chunk_max_chars = st.sidebar.number_input(
        "Chunk max chars",
        min_value=200,
        max_value=5000,
        value=int(config.chunk_max_chars),
        step=100,
    )
    if st.sidebar.button("Rebuild index"):
        build_index.clear()

    st.sidebar.header("Model Settings")
    platform_values = [p.value for p in ModelPlatformType]
    default_platform = (
        config.model_platform
        if config.model_platform in platform_values
        else ModelPlatformType.DEFAULT.value
    )
    selected_platform = st.sidebar.selectbox(
        "Model platform",
        options=platform_values,
        index=platform_values.index(default_platform),
    )
    model_options = list_models()
    model_options_with_custom = ["(custom)"] + model_options
    selected_model = st.sidebar.selectbox(
        "Model type",
        options=model_options_with_custom,
        index=(
            model_options_with_custom.index(config.model_type)
            if config.model_type in model_options_with_custom
            else 0
        ),
    )
    custom_model_type = ""
    if selected_model == "(custom)":
        custom_model_type = st.sidebar.text_input(
            "Custom model type",
            value=config.model_type or "",
        ).strip()

    model_type_value = (
        custom_model_type if selected_model == "(custom)" else selected_model
    )
    if not model_type_value:
        model_type_value = config.model_type or ""

    enable_stream = st.sidebar.toggle(
        "Stream output", value=bool(config.stream)
    )

    index = build_index(
        data_dir, int(chunk_max_chars), int(config.snippet_chars)
    )
    stats = index.stats()
    st.sidebar.caption(
        f"Files: {stats['file_count']} | "
        f"Chunks: {stats['chunk_count']} | "
        f"Chars: {stats['total_chars']}"
    )

    if (
        st.session_state["agent"] is None
        or st.session_state["agent_data_dir"] != data_dir
        or st.session_state["agent_chunk_max"] != int(chunk_max_chars)
        or st.session_state["agent_model_platform"] != selected_platform
        or st.session_state["agent_model_type"] != model_type_value
        or st.session_state["agent_max_tokens"] != int(config.max_tokens)
        or st.session_state["agent_stream"] != bool(enable_stream)
    ):
        config = replace(
            config,
            model_platform=selected_platform,
            model_type=model_type_value,
            stream=bool(enable_stream),
        )
        tools = LocalDocTools(index)
        st.session_state["agent"] = build_agent(tools, config)
        st.session_state["agent_data_dir"] = data_dir
        st.session_state["agent_chunk_max"] = int(chunk_max_chars)
        st.session_state["agent_model_platform"] = selected_platform
        st.session_state["agent_model_type"] = model_type_value
        st.session_state["agent_max_tokens"] = int(config.max_tokens)
        st.session_state["agent_stream"] = bool(enable_stream)
        st.session_state["messages"] = []

    st.title("Local File Analysis Agent")

    if st.sidebar.button("Reset conversation"):
        st.session_state["messages"] = []
        if st.session_state["agent"] is not None:
            st.session_state["agent"].reset()

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("reasoning"):
                with st.expander("Think summary"):
                    st.markdown(message["reasoning"])
            tool_calls = message.get("tool_calls") or []
            for call in tool_calls:
                tool_name = call.get("tool_name", "tool")
                with st.expander(f"Tool call: {tool_name}"):
                    st.markdown("Args")
                    st.json(call.get("args", {}))
                    st.markdown("Result")
                    st.json(call.get("result", {}))

    user_input = st.chat_input("Ask about the local documents...")
    if user_input:
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        agent = st.session_state["agent"]
        with st.chat_message("assistant"):
            content_placeholder = st.empty()
            reasoning_placeholder = st.empty()
            tools_placeholder = st.empty()

            def render_tools(calls: list[dict]) -> None:
                with tools_placeholder.container():
                    for call in calls:
                        tool_name = call.get("tool_name", "tool")
                        with st.expander(f"Tool call: {tool_name}"):
                            st.markdown("Args")
                            st.json(call.get("args", {}))
                            st.markdown("Result")
                            st.json(call.get("result", {}))

            try:
                response = agent.step(user_input)
                assistant_text = ""
                reasoning = ""
                tool_calls = []
                seen_tool_calls = set()

                if isinstance(response, StreamingChatAgentResponse):
                    for partial in response:
                        if partial.msg:
                            assistant_text = partial.msg.content or ""
                            content_placeholder.markdown(assistant_text)
                            if partial.msg.reasoning_content:
                                reasoning = partial.msg.reasoning_content
                                with reasoning_placeholder.container():
                                    st.markdown("**Think summary**")
                                    st.markdown(reasoning)
                        for record in partial.info.get("tool_calls", []) or []:
                            if hasattr(record, "as_dict"):
                                record_dict = record.as_dict()
                            elif hasattr(record, "model_dump"):
                                record_dict = record.model_dump()
                            else:
                                record_dict = record
                            key = (
                                record_dict.get("tool_call_id")
                                or f"{record_dict.get('tool_name')}:{record_dict.get('args')}"
                            )
                            if key in seen_tool_calls:
                                continue
                            seen_tool_calls.add(key)
                            tool_calls.append(record_dict)
                            render_tools(tool_calls)
                else:
                    assistant_text = (
                        response.msg.content
                        if response.msg
                        else "(no response)"
                    )
                    reasoning = (
                        response.msg.reasoning_content
                        if response.msg and response.msg.reasoning_content
                        else ""
                    )
                    tool_calls = []
                    for record in response.info.get("tool_calls", []) or []:
                        if hasattr(record, "as_dict"):
                            tool_calls.append(record.as_dict())
                        elif hasattr(record, "model_dump"):
                            tool_calls.append(record.model_dump())
                        else:
                            tool_calls.append(record)
                    content_placeholder.markdown(assistant_text)
                    if reasoning:
                        with reasoning_placeholder.container():
                            st.markdown("**Think summary**")
                            st.markdown(reasoning)
                    render_tools(tool_calls)
            except Exception as exc:
                assistant_text = f"Error: {exc}"
                reasoning = ""
                tool_calls = []
                content_placeholder.markdown(assistant_text)
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": assistant_text,
                "reasoning": reasoning,
                "tool_calls": tool_calls,
            }
        )


if __name__ == "__main__":
    main()
