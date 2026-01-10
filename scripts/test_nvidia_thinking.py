#!/usr/bin/env python3
"""Test NVIDIA DeepSeek V3.2 thinking/reasoning_content feature.

This script tests the exact API behavior to verify:
1. The correct extra_body parameter format for enabling thinking
2. Whether reasoning_content is returned in streaming responses

Usage:
    python scripts/test_nvidia_thinking.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path for config loading
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Load API key from config
from local_file_agent.llm import select_endpoint

MODEL_NAME = "deepseek-ai/deepseek-v3.2"
TEST_PROMPT = "What is 2+2? Think step by step."


def test_thinking_param(
    client: OpenAI,
    extra_body: dict,
    description: str,
) -> tuple[bool, str, str]:
    """Test a specific extra_body configuration.
    
    Returns:
        Tuple of (has_reasoning, reasoning_content, regular_content)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"extra_body: {extra_body}")
    print("=" * 60)
    
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            temperature=1,
            top_p=0.95,
            max_tokens=1024,
            extra_body=extra_body,
            stream=True,
        )
        
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            
            delta = chunk.choices[0].delta
            
            # Check for reasoning_content
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                print(f"[REASONING] {reasoning}", end="", flush=True)
            
            # Check for regular content
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
                print(f"[CONTENT] {content}", end="", flush=True)
        
        print()  # newline after streaming
        
        full_reasoning = "".join(reasoning_parts)
        full_content = "".join(content_parts)
        
        print(f"\n--- Results ---")
        print(f"Has reasoning_content: {bool(full_reasoning)}")
        print(f"Reasoning length: {len(full_reasoning)} chars")
        print(f"Content length: {len(full_content)} chars")
        
        if full_reasoning:
            print(f"\nReasoning preview (first 200 chars):")
            print(full_reasoning[:200])
        
        return bool(full_reasoning), full_reasoning, full_content
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False, "", str(e)


def main():
    # Get endpoint from config
    endpoint = select_endpoint(MODEL_NAME)
    if not endpoint:
        print(f"ERROR: No endpoint found for {MODEL_NAME}")
        print("Make sure models/model_config/base.yaml is configured correctly")
        sys.exit(1)
    
    print(f"Using endpoint: {endpoint.base_url}")
    print(f"API Key: {endpoint.api_key[:8]}...{endpoint.api_key[-4:]}")
    
    client = OpenAI(
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
    )
    
    # Test different extra_body configurations
    test_cases = [
        # Official NVIDIA format (from their docs)
        (
            {"chat_template_kwargs": {"thinking": True}},
            "Official NVIDIA format: thinking=True"
        ),
        # Current config format (potentially wrong)
        (
            {"chat_template_kwargs": {"enable_thinking": True}},
            "Current config format: enable_thinking=True"
        ),
        # No extra_body (baseline)
        (
            {},
            "No extra_body (baseline)"
        ),
    ]
    
    results = {}
    
    for extra_body, description in test_cases:
        has_reasoning, reasoning, content = test_thinking_param(
            client, extra_body, description
        )
        results[description] = has_reasoning
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for desc, has_reasoning in results.items():
        status = "✅ Has reasoning_content" if has_reasoning else "❌ No reasoning_content"
        print(f"{status}: {desc}")
    
    # Recommendation
    print("\n--- Recommendation ---")
    if results.get("Official NVIDIA format: thinking=True"):
        print("✅ Use 'thinking: true' in config (official format)")
    elif results.get("Current config format: enable_thinking=True"):
        print("✅ Current config format works")
    else:
        print("⚠️  Neither format produced reasoning_content")
        print("   The model or API might not support this feature currently")


if __name__ == "__main__":
    main()
