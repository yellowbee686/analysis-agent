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
from local_file_agent.history import (  # noqa: E402
    append_message as append_history_message,
    ensure_history_dir,
    init_session,
    list_sessions,
    load_messages,
    new_session_path,
)
from camel.agents.chat_agent import StreamingChatAgentResponse  # noqa: E402
from local_file_agent.tools import LocalDocTools  # noqa: E402
from camel.types import ModelPlatformType  # noqa: E402


def build_index(
    *,
    data_dir: Path,
    index_dir: Path,
    chunk_max_chars: int,
    snippet_chars: int,
    force_rebuild: bool,
) -> LocalIndex:
    index_key = (
        str(data_dir),
        str(index_dir),
        chunk_max_chars,
        snippet_chars,
    )
    cached_index = st.session_state.get("index")
    cached_key = st.session_state.get("index_key")
    if cached_index is None or cached_key != index_key or force_rebuild:
        index = LocalIndex(
            data_dir,
            chunk_max_chars=chunk_max_chars,
            snippet_chars=snippet_chars,
            index_dir=index_dir,
        )
        index.build(force_rebuild=force_rebuild)
        st.session_state["index"] = index
        st.session_state["index_key"] = index_key
    return st.session_state["index"]


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
    if "history_path" not in st.session_state:
        st.session_state["history_path"] = ""
    if "index" not in st.session_state:
        st.session_state["index"] = None
    if "index_key" not in st.session_state:
        st.session_state["index_key"] = None


def start_history_session(
    history_dir: Path,
    *,
    data_dir: str,
    chunk_max_chars: int,
    model_platform: str,
    model_type: str,
    max_tokens: int,
    stream: bool,
) -> Path:
    path = new_session_path(history_dir)
    meta = {
        "data_dir": data_dir,
        "chunk_max_chars": chunk_max_chars,
        "model_platform": model_platform,
        "model_type": model_type,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    init_session(path, meta)
    return path


def main() -> None:
    st.set_page_config(page_title="Local File Agent", layout="wide")
    config = load_config()
    ensure_session_state()
    history_dir = ensure_history_dir()

    st.sidebar.header("Index")
    rebuild_index = st.sidebar.button("Rebuild index")

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

    st.sidebar.header("History")
    sessions = list_sessions(history_dir)
    history_options = ["(new session)"] + [session.label for session in sessions]
    selected_history = st.sidebar.selectbox(
        "Saved sessions", options=history_options
    )
    if st.sidebar.button("Load history"):
        if selected_history != "(new session)" and sessions:
            session_index = history_options.index(selected_history) - 1
            session_path = sessions[session_index].path
            st.session_state["messages"] = load_messages(session_path)
            st.session_state["history_path"] = str(session_path)
            if st.session_state["agent"] is not None:
                st.session_state["agent"].reset()

    index = build_index(
        data_dir=config.data_dir,
        index_dir=config.index_dir,
        chunk_max_chars=int(config.chunk_max_chars),
        snippet_chars=int(config.snippet_chars),
        force_rebuild=rebuild_index,
    )
    stats = index.stats()
    st.sidebar.caption(
        f"Files: {stats['file_count']} | "
        f"Chunks: {stats['chunk_count']} | "
        f"Chars: {stats['total_chars']}"
    )

    if (
        st.session_state["agent"] is None
        or st.session_state["agent_data_dir"] != str(config.data_dir)
        or st.session_state["agent_chunk_max"]
        != int(config.chunk_max_chars)
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
        st.session_state["agent_data_dir"] = str(config.data_dir)
        st.session_state["agent_chunk_max"] = int(config.chunk_max_chars)
        st.session_state["agent_model_platform"] = selected_platform
        st.session_state["agent_model_type"] = model_type_value
        st.session_state["agent_max_tokens"] = int(config.max_tokens)
        st.session_state["agent_stream"] = bool(enable_stream)
        st.session_state["messages"] = []
        history_path = start_history_session(
            history_dir,
            data_dir=str(config.data_dir),
            chunk_max_chars=int(config.chunk_max_chars),
            model_platform=selected_platform,
            model_type=model_type_value,
            max_tokens=int(config.max_tokens),
            stream=bool(enable_stream),
        )
        st.session_state["history_path"] = str(history_path)

    if not st.session_state["history_path"]:
        history_path = start_history_session(
            history_dir,
            data_dir=str(config.data_dir),
            chunk_max_chars=int(config.chunk_max_chars),
            model_platform=selected_platform,
            model_type=model_type_value,
            max_tokens=int(config.max_tokens),
            stream=bool(enable_stream),
        )
        st.session_state["history_path"] = str(history_path)

    history_path_value = st.session_state.get("history_path", "")
    if history_path_value and st.session_state["messages"]:
        history_path = Path(history_path_value)
        if not load_messages(history_path):
            for message in st.session_state["messages"]:
                append_history_message(
                    history_path,
                    message.get("role", "assistant"),
                    message.get("content", ""),
                    reasoning=message.get("reasoning", ""),
                    tool_calls=message.get("tool_calls", []) or [],
                )

    st.title("Local File Analysis Agent")

    if st.sidebar.button("Reset conversation"):
        st.session_state["messages"] = []
        if st.session_state["agent"] is not None:
            st.session_state["agent"].reset()
        history_path = start_history_session(
            history_dir,
            data_dir=str(config.data_dir),
            chunk_max_chars=int(config.chunk_max_chars),
            model_platform=selected_platform,
            model_type=model_type_value,
            max_tokens=int(config.max_tokens),
            stream=bool(enable_stream),
        )
        st.session_state["history_path"] = str(history_path)

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
        history_path_value = st.session_state.get("history_path", "")
        if history_path_value:
            append_history_message(
                Path(history_path_value), "user", user_input
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
        history_path_value = st.session_state.get("history_path", "")
        if history_path_value:
            append_history_message(
                Path(history_path_value),
                "assistant",
                assistant_text,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )


if __name__ == "__main__":
    main()
