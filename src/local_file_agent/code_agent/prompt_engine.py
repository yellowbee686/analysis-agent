"""
Prompt engine for React Code Agent.

This module provides the PromptEngine class for constructing
system prompts and conversation messages for the agent loop.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LOOP_UPPER_BOUND = 30

CODE_AGENT_PROMPT_TEMPLATE = """# Local Document Analysis Agent

## Current Time
{current_time}

## Task Overview
Analyze local documents to answer user questions based on indexed content.

## Workflow
You complete tasks through a **code-driven analysis loop**:

1. **Understand** → Plan your analysis approach
2. **Write Code** → Implement `run_env` function in markdown code block
3. **Observe** → Review execution results
4. **Iterate or Complete** → Continue or reply directly without code

### Loop Rules
- Maximum iterations: {loop_upper_bound}
- Collect information incrementally to avoid complex code
- **When ready to answer**: Reply directly without any code block

### Code Format
Write Python code in a markdown code block. Implement the following interface:

```python
def run_env(env: LocalEnv, query: str) -> tuple[str, str]:
    '''
    Args:
        env: LocalEnv instance with document and file tools
        query: Current user query
        
    Returns:
        tuple[str, str]: (next_query, info)
            - next_query: Query for next iteration (can be same as input)
            - info: Information/result from this iteration
    '''
    # ... your code ...
    return (query, info)
```

## Core Methods

### Document Retrieval
- `env.retrieve_docs(query, top_k=5)`: BM25 semantic search
- `env.search_exact(pattern, max_results=20)`: Exact string matching
- `env.get_chunk_content(path, heading=None)`: Get full chunk content
- `env.list_chunks_in_file(path)`: List all chunks in a file
- `env.corpus_stats()`: Get corpus statistics
- `env.list_docs(limit=20)`: List indexed document paths

### File Operations
- `env.read_file_chunk(file_path, offset=0, length=8000)`: Read file chunk
- `env.search_in_file(file_path, pattern)`: Search within a file
- `env.get_file_info(file_path)`: Get file metadata
- `env.search_files_pattern(pattern, file_types, path)`: Search pattern in folder

### Academic Search (Semantic Scholar)
- `env.search_papers(query, limit=5)`: Search academic papers
- `env.get_paper_details(paper_id)`: Get paper metadata
- `env.get_author_info(author_id)`: Get author information

### CBETA Buddhist Scripture Tools (requires MCP connection)

**Search Tools:**
- `env.cbeta_search(query, rows=20)`: Full-text search in Buddhist scriptures
- `env.cbeta_search_all_in_one(query, around=10)`: Search with KWIC context
- `env.cbeta_extended_search(query)`: Advanced AND/OR/NOT/NEAR search
- `env.cbeta_search_title(query)`: Search scripture titles (經名)
- `env.cbeta_kwic_search(work, juan, query)`: KWIC search in specific fascicle

**Catalog/Metadata Tools:**
- `env.cbeta_search_catalog(query)`: Search catalog by keyword or volume
- `env.cbeta_search_by_translator(creator)`: Find works by translator (e.g., "玄奘")
- `env.cbeta_search_by_dynasty(dynasty)`: Find works by dynasty (e.g., "唐")

**Content Tools:**
- `env.cbeta_get_work_info(work)`: Get scripture metadata (T0001 → 長阿含經)
- `env.cbeta_get_toc(work)`: Get table of contents structure
- `env.cbeta_get_juan_html(work, juan)`: Get HTML content of a fascicle
- `env.cbeta_get_lines(linehead)`: Get specific lines by position
- `env.cbeta_goto(linehead, canon, work, vol, page, col, line)`: Navigate to specific position

**Enhanced CBETA Tools:**
- `env.cbeta_search_sc(query, rows=10)`: Search with simplified/traditional auto-conversion
- `env.cbeta_search_notes(query, rows=20)`: Search notes/annotations and collations
- `env.cbeta_facet_query(query, facet_type="canon")`: Aggregate by canon/category/dynasty/creator/work

## Code Requirements

### Environment
- Python 3.11+
- Pre-imported: `from local_file_agent.code_agent.react_env import *`
- Use only standard library

### Key Constraints
- Implement complete `run_env(env, query)` function
- Do NOT construct env objects
- Use env methods, not direct attribute access
- Call one tool per iteration, wait for results
- Use return for output, print is ineffective
- Do NOT catch exceptions, let them propagate

## Completion Strategy
- When you have enough information, **reply directly without code**
- Your direct reply (no code block) is the final answer to the user
- The `info` returned from `run_env` is for your reference only

## Output
- Language: {language}
- Final format: Markdown

## Context
- Indexed files: {file_count}
- Total chunks: {chunk_count}

---

# Environment API
```python
{env_api}
```
"""

PROGRESS_PROMPT_TEMPLATE = """
This is iteration {loop_step}.
"""


class PromptEngine:
    """Prompt construction engine for React Code Agent."""
    
    def __init__(self, model_name: str):
        """Initialize PromptEngine.
        
        Args:
            model_name: Name of the model being used.
        """
        self.model_name = model_name
        self.loop_step = 0
    
    def _get_file_content_as_string(self, file_path: str) -> str:
        """Read a file and return its content as a string."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("File not found: %s", file_path)
            return f"# File not found: {file_path}"
        except Exception as e:
            logger.error("Error reading file %s: %s", file_path, e)
            return f"# Error reading file: {e}"
    
    def _strip_method_implementations(self, source_code: str) -> str:
        """Strip method implementations, keeping only signatures and docstrings.
        
        This reduces token usage in prompts.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return source_code
        
        lines = source_code.splitlines(keepends=True)
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        
        ranges_to_delete: list[tuple[int, int]] = []
        
        def process_function(node: ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.body:
                return
            
            first_stmt = node.body[0]
            has_docstring = (
                isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Constant) and 
                isinstance(first_stmt.value.value, str)
            )
            
            if has_docstring:
                if len(node.body) > 1:
                    impl_start = node.body[1].lineno
                    impl_end = node.body[-1].end_lineno
                    ranges_to_delete.append((impl_start, impl_end))
            else:
                impl_start = node.body[0].lineno
                impl_end = node.body[-1].end_lineno
                ranges_to_delete.append((impl_start, impl_end))
        
        def visit_node(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                process_function(node)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        process_function(item)
            
            for child in ast.iter_child_nodes(node):
                visit_node(child)
        
        visit_node(tree)
        
        if not ranges_to_delete:
            return source_code
        
        ranges_to_delete = sorted(set(ranges_to_delete), reverse=True)
        
        for start_line, end_line in ranges_to_delete:
            start_idx = start_line - 1
            end_idx = end_line
            
            if start_idx >= len(lines) or start_idx < 0:
                continue
            
            first_impl_line = lines[start_idx]
            indent = len(first_impl_line) - len(first_impl_line.lstrip())
            pass_line = ' ' * indent + '...\n'
            lines[start_idx:end_idx] = [pass_line]
        
        return ''.join(lines)
    
    def _get_env_api_string(self) -> str:
        """Get LocalEnv API as string for prompt."""
        # Reference the react_env.py in the same code_agent directory
        env_path = Path(__file__).parent / "react_env.py"
        raw_content = self._get_file_content_as_string(str(env_path))
        return self._strip_method_implementations(raw_content)
    
    def _get_system_role(self) -> str:
        """Get system message role based on model type."""
        model_lower = self.model_name.lower()
        # GPT models use "developer" role
        if "gpt" in model_lower:
            return "developer"
        return "system"
    
    def make_init_message(
        self,
        language: str = "English",
        file_count: int = 0,
        chunk_count: int = 0,
    ) -> dict:
        """Create initial system message.
        
        Args:
            language: Output language preference.
            file_count: Number of indexed files.
            chunk_count: Number of indexed chunks.
            
        Returns:
            dict message with role and content.
        """
        from datetime import datetime
        
        env_api = self._get_env_api_string()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        init_prompt = CODE_AGENT_PROMPT_TEMPLATE.format(
            current_time=current_time,
            loop_upper_bound=LOOP_UPPER_BOUND,
            env_api=env_api,
            language=language,
            file_count=file_count,
            chunk_count=chunk_count,
        )
        
        role = self._get_system_role()
        return {
            "role": role,
            "content": init_prompt,
        }
    
    def make_user_message(self, query: str) -> dict:
        """Create user message with query.
        
        Args:
            query: User's question.
            
        Returns:
            dict message with role and content.
        """
        return {
            "role": "user",
            "content": query,
        }
    
    def make_assistant_message(
        self,
        content: str,
        code_result: str | None = None,
    ) -> list[dict]:
        """Build assistant and result messages.
        
        Args:
            content: Assistant's response content.
            code_result: Result of code execution (if any).
            
        Returns:
            list of messages to add to conversation.
        """
        messages = []
        
        # Assistant message
        messages.append({
            "role": "assistant",
            "content": content,
        })
        
        # If there's a code result, add as user message
        if code_result is not None:
            progress = PROGRESS_PROMPT_TEMPLATE.format(loop_step=self.loop_step)
            messages.append({
                "role": "user",
                "content": f"{progress}\n<code_response>\n{code_result}\n</code_response>",
            })
        
        return messages
