from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path
import sys

import streamlit as st

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from local_file_agent.agent import build_agent  # noqa: E402
from local_file_agent.config import load_config  # noqa: E402
from local_file_agent.indexer import LocalIndex  # noqa: E402
from local_file_agent.llm import list_models  # noqa: E402
from local_file_agent.mcp import connect_mcp, is_mcp_connected  # noqa: E402
from local_file_agent.history import (  # noqa: E402
    append_message as append_history_message,
    cleanup_empty_sessions,
    delete_session,
    ensure_history_dir,
    init_session,
    list_sessions,
    load_messages,
    new_session_path,
)
from camel.agents.chat_agent import StreamingChatAgentResponse  # noqa: E402
from camel.messages import BaseMessage  # noqa: E402
from local_file_agent.tools import LocalDocTools  # noqa: E402
from camel.types import ModelPlatformType, OpenAIBackendRole  # noqa: E402


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
    if "mcp_connected" not in st.session_state:
        st.session_state["mcp_connected"] = False


def init_mcp_connection(config) -> bool:
    """Initialize MCP connection if enabled.

    Returns True if MCP is connected or not enabled.
    """
    if not config.mcp_enabled:
        return True

    if st.session_state.get("mcp_connected"):
        return True

    if config.mcp_config_path is None:
        logger.warning("MCP enabled but no config path provided")
        return False

    try:
        # Run async connection in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        toolkit = loop.run_until_complete(connect_mcp(config.mcp_config_path))
        if toolkit is not None:
            st.session_state["mcp_connected"] = True
            logger.info("MCP connection established")
            return True
        else:
            logger.warning("Failed to connect to MCP servers")
            return False
    except Exception as e:
        logger.error("Error initializing MCP connection: %s", e)
        return False


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


def _get_query_param(name: str) -> str:
    if hasattr(st, "query_params"):
        value = st.query_params.get(name)
    else:
        value = st.experimental_get_query_params().get(name)
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def _clear_query_params() -> None:
    if hasattr(st, "query_params"):
        st.query_params.clear()
    else:
        st.experimental_set_query_params()


def _start_new_conversation(
    history_dir: Path,
    *,
    data_dir: str,
    chunk_max_chars: int,
    model_platform: str,
    model_type: str,
    max_tokens: int,
    stream: bool,
) -> None:
    st.session_state["messages"] = []
    agent = st.session_state.get("agent")
    if agent is not None:
        agent.reset()
    history_path = start_history_session(
        history_dir,
        data_dir=data_dir,
        chunk_max_chars=chunk_max_chars,
        model_platform=model_platform,
        model_type=model_type,
        max_tokens=max_tokens,
        stream=stream,
    )
    st.session_state["history_path"] = str(history_path)


def _select_history_session(path: Path) -> None:
    st.session_state["history_path"] = str(path)
    st.session_state["messages"] = load_messages(path)
    agent = st.session_state.get("agent")
    if agent is not None:
        agent.reset()
        try:
            for message in st.session_state["messages"]:
                role = message.get("role", "assistant")
                content = message.get("content", "")
                if role == "user":
                    msg = BaseMessage.make_user_message("User", content)
                    agent.update_memory(msg, OpenAIBackendRole.USER)
                else:
                    msg = BaseMessage.make_assistant_message(
                        "Assistant", content
                    )
                    agent.update_memory(msg, OpenAIBackendRole.ASSISTANT)
        except Exception:
            agent.reset()


def _render_history_list(
    sessions: list,
    *,
    selected_id: str,
) -> None:
    """Render history list using native Streamlit components with popover menu."""
    if not sessions:
        st.sidebar.caption("No history yet.")
        return

    with st.sidebar:
        for session in sessions:
            label = session.label or "(no questions yet)"
            session_id = session.path.name
            is_selected = session.path.name == selected_id

            col1, col2 = st.columns([5, 1])
            with col1:
                # Use button for selection
                button_type = "primary" if is_selected else "secondary"
                if st.button(
                    label,
                    key=f"history_open_{session_id}",
                    use_container_width=True,
                    type=button_type,
                    help=label,
                ):
                    _select_history_session(session.path)
                    st.rerun()
            with col2:
                # Context menu using popover
                with st.popover("⋮", help="More options"):
                    if st.button(
                        "🗑️ Delete",
                        key=f"history_delete_{session_id}",
                        use_container_width=True,
                    ):
                        st.session_state["pending_delete"] = session_id
                        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Local File Agent", layout="wide")
    config = load_config()
    ensure_session_state()
    history_dir = ensure_history_dir()

    # Initialize MCP connection if enabled
    if config.mcp_enabled and not st.session_state.get("mcp_connected"):
        with st.spinner("Connecting to MCP servers..."):
            mcp_ok = init_mcp_connection(config)
            if mcp_ok and is_mcp_connected():
                st.toast("MCP servers connected!", icon="✅")
            elif config.mcp_enabled:
                st.toast("MCP connection failed, tools unavailable", icon="⚠️")

    # Clean up empty sessions (no user messages) at startup
    if "startup_cleanup_done" not in st.session_state:
        cleanup_empty_sessions(history_dir)
        st.session_state["startup_cleanup_done"] = True
        # If current session was cleaned up, clear the history_path
        current_path = st.session_state.get("history_path", "")
        if current_path and not Path(current_path).exists():
            st.session_state["history_path"] = ""
            st.session_state["messages"] = []

    st.sidebar.header("Index")
    rebuild_index = st.sidebar.button("Rebuild index")

    st.sidebar.header("Model Settings")
    platform_values = [p.value for p in ModelPlatformType]
    selected_platform = (
        config.model_platform
        if config.model_platform in platform_values
        else ModelPlatformType.DEFAULT.value
    )
    model_options = list_models()
    if not model_options:
        st.sidebar.caption("No models configured.")
        model_type_value = config.model_type or ""
    else:
        default_model = st.session_state.get("selected_model_type")
        if default_model not in model_options:
            default_model = (
                config.model_type
                if config.model_type in model_options
                else model_options[0]
            )
        st.session_state["selected_model_type"] = default_model
        model_type_value = st.sidebar.selectbox(
            "Model type",
            options=model_options,
            key="selected_model_type",
        )

    enable_stream = True

    index = build_index(
        data_dir=config.data_dir,
        index_dir=config.index_dir,
        chunk_max_chars=int(config.chunk_max_chars),
        snippet_chars=int(config.snippet_chars),
        force_rebuild=rebuild_index,
    )
    stats = index.stats()

    # Check if we need to rebuild the agent
    need_rebuild = (
        st.session_state["agent"] is None
        or st.session_state["agent_data_dir"] != str(config.data_dir)
        or st.session_state["agent_chunk_max"] != int(config.chunk_max_chars)
        or st.session_state["agent_model_platform"] != selected_platform
        or st.session_state["agent_model_type"] != model_type_value
        or st.session_state["agent_max_tokens"] != int(config.max_tokens)
        or st.session_state["agent_stream"] != bool(enable_stream)
    )

    # Check if this is just a model switch (don't need new conversation)
    is_model_switch_only = (
        st.session_state["agent"] is not None
        and st.session_state["agent_data_dir"] == str(config.data_dir)
        and st.session_state["agent_chunk_max"] == int(config.chunk_max_chars)
        and (
            st.session_state["agent_model_platform"] != selected_platform
            or st.session_state["agent_model_type"] != model_type_value
        )
    )

    if need_rebuild:
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

        # Only reset conversation if this is NOT just a model switch
        if not is_model_switch_only:
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
        else:
            # Model switch: restore agent memory from current messages
            agent = st.session_state["agent"]
            agent.reset()
            try:
                for message in st.session_state["messages"]:
                    role = message.get("role", "assistant")
                    content = message.get("content", "")
                    if role == "user":
                        msg = BaseMessage.make_user_message("User", content)
                        agent.update_memory(msg, OpenAIBackendRole.USER)
                    else:
                        msg = BaseMessage.make_assistant_message(
                            "Assistant", content
                        )
                        agent.update_memory(msg, OpenAIBackendRole.ASSISTANT)
            except Exception:
                agent.reset()

    st.sidebar.header("History")
    new_chat_clicked = st.sidebar.button(
        "New conversation",
        use_container_width=True,
    )
    if new_chat_clicked:
        _start_new_conversation(
            history_dir,
            data_dir=str(config.data_dir),
            chunk_max_chars=int(config.chunk_max_chars),
            model_platform=selected_platform,
            model_type=model_type_value,
            max_tokens=int(config.max_tokens),
            stream=bool(enable_stream),
        )

    sessions = list_sessions(history_dir)
    session_lookup = {session.path.name: session.path for session in sessions}
    history_action = _get_query_param("history_action")
    session_id = _get_query_param("session")
    if history_action and session_id:
        target = session_lookup.get(session_id)
        if history_action == "delete" and target:
            delete_session(target)
            # Refresh sessions list after deletion
            sessions = list_sessions(history_dir)
            session_lookup = {
                session.path.name: session.path for session in sessions
            }
            # If deleted the current session, switch to most recent or create new
            if st.session_state.get("history_path") == str(target):
                if sessions:
                    # Select the most recent session
                    _select_history_session(sessions[0].path)
                else:
                    # No sessions left, create a new one
                    _start_new_conversation(
                        history_dir,
                        data_dir=str(config.data_dir),
                        chunk_max_chars=int(config.chunk_max_chars),
                        model_platform=selected_platform,
                        model_type=model_type_value,
                        max_tokens=int(config.max_tokens),
                        stream=bool(enable_stream),
                    )
                    sessions = list_sessions(history_dir)
                    session_lookup = {
                        session.path.name: session.path for session in sessions
                    }
            _clear_query_params()
            st.rerun()
        elif history_action == "open" and target:
            _select_history_session(target)
            _clear_query_params()
            st.rerun()

    if not st.session_state.get("history_path"):
        if sessions:
            _select_history_session(sessions[0].path)
        else:
            _start_new_conversation(
                history_dir,
                data_dir=str(config.data_dir),
                chunk_max_chars=int(config.chunk_max_chars),
                model_platform=selected_platform,
                model_type=model_type_value,
                max_tokens=int(config.max_tokens),
                stream=bool(enable_stream),
            )
            sessions = list_sessions(history_dir)
            session_lookup = {
                session.path.name: session.path for session in sessions
            }

    selected_id = ""
    history_path_value = st.session_state.get("history_path", "")
    if history_path_value:
        selected_id = Path(history_path_value).name

    # Handle pending delete from history list
    pending_delete = st.session_state.pop("pending_delete", None)
    if pending_delete:
        target = session_lookup.get(pending_delete)
        if target:
            delete_session(target)
            # Refresh sessions list after deletion
            sessions = list_sessions(history_dir)
            session_lookup = {
                session.path.name: session.path for session in sessions
            }
            # If deleted the current session, switch to most recent or create new
            if st.session_state.get("history_path") == str(target):
                if sessions:
                    _select_history_session(sessions[0].path)
                else:
                    _start_new_conversation(
                        history_dir,
                        data_dir=str(config.data_dir),
                        chunk_max_chars=int(config.chunk_max_chars),
                        model_platform=selected_platform,
                        model_type=model_type_value,
                        max_tokens=int(config.max_tokens),
                        stream=bool(enable_stream),
                    )
                    sessions = list_sessions(history_dir)
                    session_lookup = {
                        session.path.name: session.path for session in sessions
                    }
            # Update selected_id after deletion
            history_path_value = st.session_state.get("history_path", "")
            selected_id = Path(history_path_value).name if history_path_value else ""
            st.rerun()

    _render_history_list(
        sessions,
        selected_id=selected_id,
    )

    # Display MCP status
    if config.mcp_enabled:
        mcp_status = "✅ Connected" if is_mcp_connected() else "❌ Disconnected"
        st.sidebar.caption(f"MCP: {mcp_status}")

    st.sidebar.caption(
        f"Files: {stats['file_count']} | "
        f"Chunks: {stats['chunk_count']} | "
        f"Chars: {stats['total_chars']}"
    )

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
                logger.info("Sending user input to agent: %s...", user_input[:50])
                response = agent.step(user_input)
                logger.info("Agent response received, type: %s", type(response).__name__)
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
