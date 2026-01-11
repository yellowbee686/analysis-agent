#!/usr/bin/env python3
"""Test non-streaming mode reasoning_content for DeepSeek V3.2.

This script tests whether reasoning_content is correctly returned
in non-streaming mode, both via raw OpenAI API and via CAMEL agent.

Usage:
    # Test raw OpenAI API
    uv run python scripts/test_non_stream_reasoning.py

    # Test with CAMEL agent
    uv run python scripts/test_non_stream_reasoning.py --agent
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path for config loading
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from dotenv import load_dotenv

load_dotenv()

from local_file_agent.llm import select_endpoint, get_model_params

MODEL_NAME = "deepseek-ai/deepseek-v3.2"
TEST_PROMPT = "What is 15 + 27? Think step by step."


def test_openai_non_stream() -> tuple[bool, str, str]:
    """Test non-streaming mode with raw OpenAI API.
    
    Returns:
        Tuple of (has_reasoning, reasoning_content, regular_content)
    """
    from openai import OpenAI
    
    print("\n" + "=" * 60)
    print("Testing: Raw OpenAI API (non-streaming)")
    print("=" * 60)
    
    endpoint = select_endpoint(MODEL_NAME)
    if not endpoint:
        print(f"ERROR: No endpoint found for {MODEL_NAME}")
        return False, "", ""
    
    model_params = get_model_params(MODEL_NAME)
    extra_body = model_params.get("extra_body", {})
    
    print(f"Endpoint: {endpoint.base_url}")
    print(f"API Key: {endpoint.api_key[:8]}...{endpoint.api_key[-4:]}")
    print(f"extra_body: {extra_body}")
    
    client = OpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            temperature=model_params.get("temperature", 1.0),
            top_p=model_params.get("top_p", 0.95),
            max_tokens=1024,
            extra_body=extra_body,
            stream=False,  # Non-streaming!
        )
        
        message = completion.choices[0].message
        
        # Get content
        content = message.content or ""
        
        # Get reasoning_content (may be in different places)
        reasoning = ""
        
        # Try direct attribute
        if hasattr(message, "reasoning_content"):
            reasoning = getattr(message, "reasoning_content", "") or ""
        
        # Try model_extra (for Pydantic models)
        if not reasoning and hasattr(message, "model_extra"):
            reasoning = message.model_extra.get("reasoning_content", "") or ""
        
        # Try dict access
        if not reasoning:
            try:
                msg_dict = message.model_dump() if hasattr(message, "model_dump") else dict(message)
                reasoning = msg_dict.get("reasoning_content", "") or ""
            except Exception:
                pass
        
        print(f"\n--- Results ---")
        print(f"Has reasoning_content: {bool(reasoning)}")
        print(f"Reasoning length: {len(reasoning)} chars")
        print(f"Content length: {len(content)} chars")
        
        if reasoning:
            print(f"\nReasoning (first 500 chars):")
            print(reasoning[:500])
            print("...")
        
        print(f"\nContent:")
        print(content)
        
        # Debug: print raw message structure
        print(f"\n--- Debug: Message structure ---")
        try:
            msg_dict = message.model_dump() if hasattr(message, "model_dump") else vars(message)
            for key, value in msg_dict.items():
                if key != "content":
                    val_str = str(value)[:100] if value else "None"
                    print(f"  {key}: {val_str}")
        except Exception as e:
            print(f"  Could not dump message: {e}")
        
        return bool(reasoning), reasoning, content
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, "", str(e)


def test_camel_agent_non_stream() -> tuple[bool, str, str]:
    """Test non-streaming mode with CAMEL agent.
    
    Returns:
        Tuple of (has_reasoning, reasoning_content, regular_content)
    """
    print("\n" + "=" * 60)
    print("Testing: CAMEL Agent (non-streaming)")
    print("=" * 60)
    
    # Set stream=false via environment
    os.environ["LOCAL_AGENT_STREAM"] = "false"
    os.environ["LOCAL_AGENT_MODEL_TYPE"] = MODEL_NAME
    
    from local_file_agent.config import load_config
    from local_file_agent.agent import build_agent
    from local_file_agent.indexer import LocalIndex
    from local_file_agent.tools import LocalDocTools
    
    config = load_config()
    print(f"Config stream: {config.stream}")
    print(f"Config model_type: {config.model_type}")
    
    # Create minimal index and tools
    data_dir = ROOT_DIR / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    index = LocalIndex(
        data_dir,
        chunk_max_chars=1200,
        snippet_chars=400,
        index_dir=ROOT_DIR / "cache" / "indices",
    )
    index.build(force_rebuild=False)
    
    tools = LocalDocTools(index)
    agent = build_agent(tools, config)
    
    try:
        print(f"\nSending prompt: {TEST_PROMPT}")
        response = agent.step(TEST_PROMPT)
        
        # Get content
        content = response.msg.content if response.msg else ""
        
        # Get reasoning_content
        reasoning = ""
        if response.msg and hasattr(response.msg, "reasoning_content"):
            reasoning = response.msg.reasoning_content or ""
        
        print(f"\n--- Results ---")
        print(f"Response type: {type(response).__name__}")
        print(f"Has reasoning_content: {bool(reasoning)}")
        print(f"Reasoning length: {len(reasoning)} chars")
        print(f"Content length: {len(content)} chars")
        
        if reasoning:
            print(f"\nReasoning (first 500 chars):")
            print(reasoning[:500])
            print("...")
        
        print(f"\nContent:")
        print(content)
        
        # Debug: print message structure
        print(f"\n--- Debug: Message attributes ---")
        if response.msg:
            for attr in dir(response.msg):
                if not attr.startswith("_"):
                    try:
                        val = getattr(response.msg, attr)
                        if not callable(val) and val:
                            val_str = str(val)[:100]
                            print(f"  {attr}: {val_str}")
                    except Exception:
                        pass
        
        return bool(reasoning), reasoning, content
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, "", str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Test non-streaming reasoning_content"
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Test with CAMEL agent (default: raw OpenAI API)",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Test both raw API and CAMEL agent",
    )
    args = parser.parse_args()
    
    results = {}
    
    if args.both or not args.agent:
        has_reasoning, reasoning, content = test_openai_non_stream()
        results["Raw OpenAI API"] = has_reasoning
    
    if args.both or args.agent:
        has_reasoning, reasoning, content = test_camel_agent_non_stream()
        results["CAMEL Agent"] = has_reasoning
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for desc, has_reasoning in results.items():
        status = "✅ Has reasoning_content" if has_reasoning else "❌ No reasoning_content"
        print(f"{status}: {desc}")
    
    # Return non-zero if no reasoning found
    if not any(results.values()):
        print("\n⚠️  No reasoning_content found in non-streaming mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
