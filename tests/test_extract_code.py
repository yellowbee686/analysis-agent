"""Tests for extract_code_from_md and its helper functions."""
from __future__ import annotations

import textwrap

import pytest

from local_file_agent.code_agent.react_agent import (
    _extract_closed_md,
    _extract_json_code,
    _extract_unclosed_md,
    _postprocess_code,
    _unwrap_tool_call,
    extract_code_from_md,
)


# ── Standard closed markdown ──────────────────────────────────────────────


class TestClosedMarkdown:
    def test_python_fence(self):
        text = textwrap.dedent("""\
            Here is the plan:

            ```python
            def run_env(env, query):
                return query, "ok"
            ```

            Done.
        """)
        assert "def run_env" in extract_code_from_md(text)

    def test_bare_fence(self):
        text = "```\nprint('hello')\n```"
        assert extract_code_from_md(text) == "print('hello')"

    def test_py_fence(self):
        text = "```py\nprint('hello')\n```"
        assert extract_code_from_md(text) == "print('hello')"


# ── Unclosed markdown ─────────────────────────────────────────────────────


class TestUnclosedMarkdown:
    def test_no_closing_fence(self):
        text = textwrap.dedent("""\
            ```python
            def run_env(env, query):
                result = env.retrieve_docs(query)
                return query, str(result)
        """)
        code = extract_code_from_md(text)
        assert "def run_env" in code
        assert "retrieve_docs" in code

    def test_trailing_single_backtick(self):
        text = "```python\nprint('hi')\n`"
        assert extract_code_from_md(text) == "print('hi')"

    def test_trailing_double_backtick(self):
        text = "```python\nprint('hi')\n``"
        assert extract_code_from_md(text) == "print('hi')"


# ── Tool-call wrappers ────────────────────────────────────────────────────


class TestToolCallWrappers:
    def test_deepseek_delimiter_with_json_code(self):
        """DeepSeek-style: <|tool_call_argument_begin|>{"code":"..."}"""
        text = (
            '<|tool_calls_section_begin|> <|tool_call_begin|> '
            'functions.python:0 <|tool_call_argument_begin|> '
            '{"code":"def run_env(env, query):\\n'
            '    result = env.cbeta_search(\\"test\\")\\n'
            '    return (query, str(result))"}'
            ' <|tool_call_end|> <|tool_calls_section_end|>'
        )
        code = extract_code_from_md(text)
        assert "def run_env" in code
        assert "cbeta_search" in code

    def test_deepseek_delimiter_with_markdown_inside(self):
        """Tool-call wrapper containing markdown fence."""
        inner_md = "```python\nresult = env.search_exact('test')\n```"
        text = (
            f"<|tool_call_argument_begin|> {inner_md} <|tool_call_end|>"
        )
        code = extract_code_from_md(text)
        assert "search_exact" in code

    def test_xml_minimax_tool_call(self):
        """MiniMax-style: <minimax:tool_call>{"code":"..."}</minimax:tool_call>"""
        text = (
            '<minimax:tool_call>{"code":"def run_env(env, q):\\n'
            '    return q, \\"done\\""}</minimax:tool_call>'
        )
        code = extract_code_from_md(text)
        assert "def run_env" in code

    def test_generic_xml_tool_call(self):
        """<tool_call>{"code":"..."}</tool_call>"""
        text = '<tool_call>{"code":"x = 1"}</tool_call>'
        assert extract_code_from_md(text) == "x = 1"


# ── JSON code field ───────────────────────────────────────────────────────


class TestJsonCodeField:
    def test_direct_code_field(self):
        text = '{"code": "print(42)"}'
        assert _extract_json_code(text) == "print(42)"

    def test_nested_arguments(self):
        text = '{"name": "python", "arguments": {"code": "a = 1"}}'
        assert _extract_json_code(text) == "a = 1"

    def test_double_encoded_arguments(self):
        text = '{"name": "python", "arguments": "{\\"code\\": \\"b = 2\\"}"}'
        assert _extract_json_code(text) == "b = 2"

    def test_no_code_field(self):
        assert _extract_json_code('{"name": "test"}') == ""

    def test_invalid_json(self):
        assert _extract_json_code("not json at all") == ""

    def test_json_with_surrounding_text(self):
        text = 'function call: {"code": "x = 1"} end'
        assert _extract_json_code(text) == "x = 1"


# ── Unicode postprocessing ────────────────────────────────────────────────


class TestPostprocess:
    def test_unicode_arrow(self):
        code = "def f(x: int) \u2192 str:"
        assert _postprocess_code(code) == "def f(x: int) -> str:"

    def test_smart_quotes(self):
        code = 'name = \u201chello\u201d'
        assert _postprocess_code(code) == 'name = "hello"'

    def test_smart_single_quotes(self):
        code = "name = \u2018hello\u2019"
        assert _postprocess_code(code) == "name = 'hello'"

    def test_integration_arrow_in_tool_call(self):
        """Full pipeline: tool-call wrapper + JSON code + unicode arrow."""
        text = (
            '<|tool_call_argument_begin|> '
            '{"code":"def run_env(env, query) \\u2192 tuple[str, str]:\\n'
            '    return query, \\"ok\\""}'
            ' <|tool_call_end|>'
        )
        code = extract_code_from_md(text)
        assert "->" in code
        assert "\u2192" not in code


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self):
        assert extract_code_from_md("") == ""

    def test_none(self):
        # Our function checks `if not text`
        assert extract_code_from_md(None) == ""  # type: ignore[arg-type]

    def test_plain_text_no_code(self):
        assert extract_code_from_md("This is just a normal answer.") == ""

    def test_priority_closed_over_tool_call(self):
        """If both closed markdown and tool-call exist, prefer markdown."""
        text = (
            "```python\nfrom_markdown = True\n```\n"
            '<tool_call>{"code":"from_tool = True"}</tool_call>'
        )
        code = extract_code_from_md(text)
        assert "from_markdown" in code
        assert "from_tool" not in code

    def test_multiple_code_blocks_returns_first(self):
        text = "```python\nfirst = 1\n```\n\n```python\nsecond = 2\n```"
        code = extract_code_from_md(text)
        assert "first" in code

    def test_unclosed_md_inside_tool_call(self):
        """Tool-call wrapper containing unclosed markdown fence."""
        inner = "```python\nx = 42"
        text = f"<|tool_call_argument_begin|>{inner}<|tool_call_end|>"
        assert extract_code_from_md(text) == "x = 42"


# ── _unwrap_tool_call unit tests ──────────────────────────────────────────


class TestUnwrapToolCall:
    def test_returns_none_for_normal_text(self):
        assert _unwrap_tool_call("no wrappers here") is None

    def test_delimiter_style(self):
        text = "<|tool_call_argument_begin|>payload<|tool_call_argument_end|>"
        assert _unwrap_tool_call(text) == "payload"

    def test_delimiter_end_variant(self):
        text = "<|tool_call_argument_begin|>payload<|tool_call_end|>"
        assert _unwrap_tool_call(text) == "payload"

    def test_xml_with_prefix(self):
        text = "<abc:tool_call>inner</abc:tool_call>"
        assert _unwrap_tool_call(text) == "inner"

    def test_xml_bare(self):
        text = "<tool_call>inner</tool_call>"
        assert _unwrap_tool_call(text) == "inner"
