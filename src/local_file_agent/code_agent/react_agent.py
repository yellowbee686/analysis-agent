"""
React Code Agent for local document analysis.

This module provides a React-style agent that uses LLM-generated code
to interact with local documents through an execution environment.
"""
from __future__ import annotations

import logging
import re
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from local_file_agent.config import AppConfig
from local_file_agent.indexer import LocalIndex
from local_file_agent.llm import build_openai_clients, get_model_params, select_endpoint
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


def extract_code_from_md(text: str) -> str:
    """Extract Python code from markdown code blocks.
    
    Args:
        text: Text that may contain markdown code blocks.
        
    Returns:
        Extracted code or empty string if no code found.
    """
    if not text:
        return ""
    
    # Match ```python ... ``` or ``` ... ```
    patterns = [
        r'```python\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
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
        self.env = LocalEnv(self.index, None)
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
            return response.choices[0].message.content or ""
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
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
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
