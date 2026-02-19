"""
React Code Agent for local document analysis.

This module provides a React-style agent that uses LLM-generated code
to interact with local documents through an execution environment.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from local_file_agent.config import AppConfig
from local_file_agent.indexer import LocalIndex
from local_file_agent.llm import (
    build_openai_clients,
    extract_delta_text,
    extract_response_text,
    get_model_params,
    select_endpoint,
)
from local_file_agent.code_agent.prompt_engine import PromptEngine, LOOP_UPPER_BOUND
from local_file_agent.code_agent.react_env import LocalEnv

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from agent step."""
    content: str
    reasoning: str = ""
    tool_calls: list[dict] | None = None
    is_final: bool = False


# ---------------------------------------------------------------------------
# Code extraction helpers — tolerant of various LLM output formats
# ---------------------------------------------------------------------------

# DeepSeek / Qwen delimiter-style tool-call wrappers:
#   <|tool_call_argument_begin|> CONTENT <|tool_call_argument_end|>
#   or <|tool_call_argument_begin|> CONTENT <|tool_call_end|>
_RE_DELIM_ARG = re.compile(
    r"<\|tool_call_argument_begin\|>\s*(.*?)\s*"
    r"<\|(?:tool_call_argument_end|tool_call_end)\|>",
    re.DOTALL,
)

# Generic XML-like tool-call wrappers:
#   <minimax:tool_call>...</minimax:tool_call>
#   <tool_call>...</tool_call>
_RE_XML_TOOL = re.compile(
    r"<(?:[\w.-]+:)?tool_call[^>]*>(.*?)</(?:[\w.-]+:)?tool_call[^>]*>",
    re.DOTALL,
)

_TOOL_CALL_PATTERNS: list[re.Pattern[str]] = [_RE_DELIM_ARG, _RE_XML_TOOL]


def _unwrap_tool_call(text: str) -> str | None:
    """Extract inner payload from tool-call wrapper tags.

    Supports delimiter-style (``<|..._begin|>...<|..._end|>``) and
    XML-style (``<xxx:tool_call>...</xxx:tool_call>``) wrappers.

    Returns the unwrapped content, or *None* if no wrapper matched.
    """
    for pat in _TOOL_CALL_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


def _extract_json_code(text: str) -> str:
    """Try to parse *text* as / containing JSON and pull out a ``code`` field.

    Handles common shapes emitted by models:
    - ``{"code": "..."}``
    - ``{"arguments": {"code": "..."}}``
    - ``{"arguments": "{\\"code\\": \\"...\\"}" }``  (double-encoded)
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""

    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return ""

    if not isinstance(data, dict):
        return ""

    # Direct {"code": "..."}
    if "code" in data:
        return str(data["code"]).strip()

    # Nested {"arguments": {"code": "..."}}  or double-encoded
    args = data.get("arguments")
    if isinstance(args, dict) and "code" in args:
        return str(args["code"]).strip()
    if isinstance(args, str):
        try:
            args_obj = json.loads(args)
            if isinstance(args_obj, dict) and "code" in args_obj:
                return str(args_obj["code"]).strip()
        except (json.JSONDecodeError, ValueError):
            pass

    return ""


def _extract_closed_md(text: str) -> str:
    """Extract code from a **closed** markdown fence (````python...```` or ````...````)."""
    for tag in ("python", "py", ""):
        pat = rf"```{tag}\s*\n(.*?)```"
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def _extract_unclosed_md(text: str) -> str:
    """Extract code from an **unclosed** markdown fence (no closing ``````).

    This is a last-resort fallback: grab everything after the opening fence.
    """
    for tag in ("python", "py", ""):
        pat = rf"```{tag}\s*\n(.+)"
        m = re.search(pat, text, re.DOTALL)
        if m:
            code = m.group(1)
            # Strip trailing incomplete backticks (1–2 stray `)
            code = re.sub(r"`{1,2}\s*$", "", code)
            return code.strip()
    return ""


def _postprocess_code(code: str) -> str:
    """Fix common Unicode artefacts produced by some models."""
    code = code.replace("\u2192", "->")  # → → ->
    code = code.replace("\u2190", "=")   # ← → =
    code = code.replace("\u201c", '"')   # " → "
    code = code.replace("\u201d", '"')   # " → "
    code = code.replace("\u2018", "'")   # ' → '
    code = code.replace("\u2019", "'")   # ' → '
    return code


def extract_code_from_md(text: str) -> str:
    """Extract Python code from an LLM response.

    Tries multiple strategies in order of reliability:

    1. **Closed markdown fence** — standard ````python ... ````
    2. **Tool-call wrappers** — DeepSeek ``<|...|>`` / XML ``<xxx:tool_call>``
       tags, with optional JSON ``{"code": "..."}`` payload inside.
    3. **Bare JSON code field** — ``{"code": "..."}`` without a wrapper.
    4. **Unclosed markdown fence** — ````python`` without a closing fence.

    All extracted code is post-processed to replace common Unicode artefacts
    (e.g. ``→`` → ``->``).

    Args:
        text: Raw LLM response text.

    Returns:
        Extracted Python code, or empty string if nothing found.
    """
    if not text:
        return ""

    # 1. Closed markdown fence (most common & reliable)
    code = _extract_closed_md(text)
    if code:
        return _postprocess_code(code)

    # 2. Unwrap tool-call tags, then try JSON / markdown inside
    inner = _unwrap_tool_call(text)
    if inner:
        code = (
            _extract_json_code(inner)
            or _extract_closed_md(inner)
            or _extract_unclosed_md(inner)
        )
        if code:
            return _postprocess_code(code)

    # 3. Bare JSON {"code": "..."} anywhere in the text
    code = _extract_json_code(text)
    if code:
        return _postprocess_code(code)

    # 4. Unclosed markdown fence (last resort)
    code = _extract_unclosed_md(text)
    if code:
        return _postprocess_code(code)

    return ""


class ReactCodeAgent:
    """React-style code agent for local document analysis.
    
    Uses an agent loop where the LLM generates code that is executed
    against a LocalEnv, with results fed back for the next iteration.
    """
    
    def __init__(
        self,
        index: LocalIndex,
        config: AppConfig,
        working_directory: str | Path | None = None,
        mcp_toolkit: Any | None = None,
    ):
        """Initialize ReactCodeAgent.
        
        Args:
            index: LocalIndex for document search.
            config: Application configuration.
            working_directory: Working directory for file operations.
            mcp_toolkit: Optional MCPToolkit for MCP tools.
        """
        self.index = index
        self.config = config
        self.model_name = config.model_type or ""
        self._working_directory = working_directory
        self._mcp_toolkit = mcp_toolkit
        
        # Initialize environment with optional MCP toolkit
        self.env = LocalEnv(index, working_directory, mcp_toolkit=mcp_toolkit)
        
        # Initialize prompt engine
        self.prompt_engine = PromptEngine(self.model_name)
        
        # Conversation history
        self.messages: list[dict] = []
        
        # Build LLM client
        self._client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize OpenAI client for LLM calls."""
        endpoint = select_endpoint(self.model_name)
        if endpoint:
            client, _ = build_openai_clients(endpoint, use_azure=endpoint.use_azure)
            self._client = client
        else:
            logger.warning("No endpoint found for model: %s", self.model_name)
    
    def reset(self) -> None:
        """Reset agent state for new conversation."""
        self.messages = []
        self.env = LocalEnv(
            self.index,
            self._working_directory,
            mcp_toolkit=self._mcp_toolkit,
        )
        self.prompt_engine.loop_step = 0

    def load_history(self, history: list[dict]) -> None:
        """Load chat history into prompt messages."""
        self.reset()
        stats = self.index.stats()
        init_msg = self.prompt_engine.make_init_message(
            language="English",
            file_count=stats.get("file_count", 0),
            chunk_count=stats.get("chunk_count", 0),
        )
        self.messages.append(init_msg)
        for message in history:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = message.get("content", "")
            if content is None:
                continue
            self.messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )
        self.prompt_engine.loop_step = 0
    
    def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM and return response content.
        
        Args:
            messages: Conversation messages.
            
        Returns:
            LLM response content.
        """
        if self._client is None:
            raise RuntimeError("LLM client not initialized")
        
        model_params = get_model_params(self.model_name)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                **model_params,
            )
            return extract_response_text(response)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise
    
    def _call_llm_stream(
        self,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        """Call LLM with streaming and yield content chunks.
        
        Args:
            messages: Conversation messages.
            
        Yields:
            Content chunks from streaming response.
        """
        if self._client is None:
            raise RuntimeError("LLM client not initialized")
        
        model_params = get_model_params(self.model_name)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                stream=True,
                **model_params,
            )
            
            for chunk in response:
                if not chunk.choices:
                    continue
                text = extract_delta_text(chunk.choices[0].delta)
                if text:
                    yield text
                    
        except Exception as e:
            logger.error("LLM streaming call failed: %s", e)
            raise
    
    def _execute_code(self, code: str, query: str) -> tuple[bool, str]:
        """Execute LLM-generated code.
        
        Args:
            code: Python code to execute.
            query: Current user query.
            
        Returns:
            tuple of (success, result_or_error).
        """
        # Wrap code if run_env not defined
        if 'run_env' not in code:
            env_prefix = "def run_env(env: LocalEnv, query: str) -> tuple[str, str]:\n"
            indented_code = "\n".join(f"    {line}" for line in code.splitlines())
            code = env_prefix + indented_code
        
        exec_code = f"from local_file_agent.code_agent.react_env import *\n{code}"
        
        namespace = {}
        try:
            exec(exec_code, namespace)
            
            if 'run_env' not in namespace:
                return False, "Error: Code does not contain run_env function"
            
            run_env_func = namespace['run_env']
            _, result = run_env_func(self.env, query)
            
            if isinstance(result, dict):
                result = str(result)
            
            logger.info("Code execution successful, result length: %d", len(str(result)))
            return True, result
            
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.info("Code execution error: %s", tb_str)
            return False, tb_str
    
    def step(self, user_input: str) -> AgentResponse:
        """Run a single conversation step (non-streaming).
        
        Args:
            user_input: User's message.
            
        Returns:
            AgentResponse with content and metadata.
        """
        # Collect streaming output
        content_parts = []
        final_response = None
        
        for response in self.stream_step(user_input):
            if response.content:
                content_parts.append(response.content)
            final_response = response
        
        if final_response:
            final_response.content = "".join(content_parts)
            return final_response
        
        return AgentResponse(content="", is_final=True)
    
    def stream_step(
        self,
        user_input: str,
    ) -> Generator[AgentResponse, None, None]:
        """Run conversation step with streaming.
        
        Args:
            user_input: User's message.
            
        Yields:
            AgentResponse chunks during processing.
        """
        # Initialize messages if empty
        if not self.messages:
            stats = self.index.stats()
            init_msg = self.prompt_engine.make_init_message(
                language="English",
                file_count=stats.get("file_count", 0),
                chunk_count=stats.get("chunk_count", 0),
            )
            self.messages.append(init_msg)
        
        # Add user message
        user_msg = self.prompt_engine.make_user_message(user_input)
        self.messages.append(user_msg)
        
        # Agent loop
        self.prompt_engine.loop_step = 1
        
        while self.prompt_engine.loop_step <= LOOP_UPPER_BOUND:
            logger.info("Agent loop step %d", self.prompt_engine.loop_step)
            
            # Call LLM with streaming
            full_content = ""
            for chunk in self._call_llm_stream(self.messages):
                full_content += chunk
                yield AgentResponse(content=chunk, is_final=False)
            
            # Extract code from response
            code = extract_code_from_md(full_content)
            
            if not code:
                # No code = final answer
                logger.info("No code found, treating as final answer")
                self.messages.extend(
                    self.prompt_engine.make_assistant_message(full_content)
                )
                yield AgentResponse(content="", is_final=True)
                return
            
            # Execute code
            success, result = self._execute_code(code, user_input)
            
            # Build messages
            msgs = self.prompt_engine.make_assistant_message(full_content, result)
            self.messages.extend(msgs)
            
            # Yield execution result as reasoning/tool info
            result_preview = result[:500] + "..." if len(result) > 500 else result
            yield AgentResponse(
                content=f"\n\n---\n**Code Result (step {self.prompt_engine.loop_step}):**\n```\n{result_preview}\n```\n",
                reasoning=result,
                is_final=False,
            )
            
            self.prompt_engine.loop_step += 1
        
        # Max iterations reached
        logger.warning("Max iterations reached (%d)", LOOP_UPPER_BOUND)
        yield AgentResponse(
            content="\n\n*[Max iterations reached, returning partial results]*",
            is_final=True,
        )


def build_react_agent(
    index: LocalIndex,
    config: AppConfig,
    mcp_toolkit: Any | None = None,
) -> ReactCodeAgent:
    """Build a ReactCodeAgent instance.
    
    Args:
        index: LocalIndex for document search.
        config: Application configuration.
        mcp_toolkit: Optional MCPToolkit for MCP tools.
        
    Returns:
        Configured ReactCodeAgent instance.
    """
    from local_file_agent.mcp import get_session_storage_path
    
    working_dir = get_session_storage_path()
    if working_dir is None:
        working_dir = Path("cache/sessions/_default/mcp_responses")
        working_dir.mkdir(parents=True, exist_ok=True)
    
    return ReactCodeAgent(
        index=index,
        config=config,
        working_directory=working_dir,
        mcp_toolkit=mcp_toolkit,
    )
