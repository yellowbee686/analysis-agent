# Code Agent Development Guide

> React-style Code Execution Agent for Local Document Analysis

---

## 1. Overview

### 1.1 Architecture

The Code Agent uses a React-style loop where an LLM generates Python code to interact with local documents and external services through an execution environment (`LocalEnv`).

```
┌─────────────────────────────────────────────────────────┐
│                     Code Agent Loop                      │
├─────────────────────────────────────────────────────────┤
│  1. LLM receives query + env API                         │
│  2. LLM generates `run_env(env, query)` function         │
│  3. Agent executes code, captures result                 │
│  4. Result fed back to LLM for next iteration            │
│  5. When ready, LLM replies directly (no code)           │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Module Structure

```
code_agent/
├── __init__.py         # Public exports
├── react_agent.py      # Main agent loop implementation
├── react_env.py        # LocalEnv: execution environment with tools
├── prompt_engine.py    # System prompt construction
└── DEV_DOC.md         # This documentation
```

---

## 2. Core Components

### 2.1 `LocalEnv` (react_env.py)

The execution environment that wraps all available tools. LLM-generated code calls methods on this object.

**Categories of tools:**

| Category | Prefix | Description |
|----------|--------|-------------|
| Document Retrieval | `retrieve_*`, `search_*` | BM25/exact search on indexed local docs |
| File Operations | `read_file_*`, `get_file_*` | Read/search within files |
| Academic Search | `search_papers`, `get_paper_*` | Semantic Scholar API |
| **CBETA Buddhist Scriptures** | `cbeta_*` | MCP-based CBETA API access |

### 2.2 `PromptEngine` (prompt_engine.py)

Constructs system prompts with:
- Current time and context
- Available env methods and their signatures
- Code format requirements
- Loop constraints

### 2.3 `ReactCodeAgent` (react_agent.py)

Main agent loop that:
1. Sends system prompt + user query to LLM
2. Extracts Python code from LLM response
3. Executes code in sandboxed environment
4. Feeds result back for next iteration
5. Detects when LLM replies without code (completion)

---

## 3. CBETA MCP Tools

The agent integrates with CBETA (Chinese Buddhist Electronic Text Association) through MCP (Model Context Protocol).

### 3.1 Tool Categories

#### Search Tools (Finding Content)

| Method | Description | Key Args |
|--------|-------------|----------|
| `cbeta_search(query)` | Basic full-text search | `query`, `rows`, `start` |
| `cbeta_search_all_in_one(query)` | Search with KWIC context | `query`, `around`, `facet` |
| `cbeta_extended_search(query)` | AND/OR/NOT/NEAR operators | `query` with operator syntax |
| `cbeta_search_title(query)` | Search scripture titles | `query` (min 3 chars) |
| `cbeta_kwic_search(work, juan, query)` | KWIC in specific fascicle | `work`, `juan`, `query` |

#### Catalog/Metadata Tools (Finding Scriptures)

| Method | Description | Key Args |
|--------|-------------|----------|
| `cbeta_search_catalog(query)` | Search by keyword or volume | `query` e.g., "阿含", "T01" |
| `cbeta_search_by_translator(creator)` | Find by translator | `creator` e.g., "玄奘" |
| `cbeta_search_by_dynasty(dynasty)` | Find by dynasty/period | `dynasty`, `time_start/end` |

#### Content Tools (Reading Scriptures)

| Method | Description | Key Args |
|--------|-------------|----------|
| `cbeta_get_work_info(work)` | Scripture metadata | `work` e.g., "T0001" |
| `cbeta_get_toc(work)` | Table of contents | `work` |
| `cbeta_get_juan_html(work, juan)` | Fascicle HTML content | `work`, `juan` |
| `cbeta_get_lines(linehead)` | Specific lines by position | `linehead`, `before/after` |
| `cbeta_goto(linehead)` | Navigate to position | `linehead` or structured params |

### 3.2 CBETA ID Formats

**Work ID (佛典編號):**
- `T0001` = Taishō 大正藏, Work #1 (長阿含經)
- `X0600` = Xuzang 卍續藏, Work #600
- `J0001` = Jiaxing 嘉興藏, Work #1
- `N0001` = Nandenchō 南傳大藏經, Work #1

**Linehead (行首位置):**
- Format: `{vol}n{work}_p{page}{col}{line}`
- Example: `T01n0001_p0001a04`
  - `T01` = Volume T01
  - `n0001` = Work 0001
  - `p0001` = Page 1
  - `a` = Column a (a/b/c)
  - `04` = Line 4

### 3.3 Common Usage Patterns

**Pattern 1: Find scripture by topic, read content**
```python
def run_env(env, query):
    # 1. Search for relevant scriptures
    result = env.cbeta_search("四聖諦", rows=5)
    works = [r["work"] for r in result.get("result", {}).get("results", [])]
    
    # 2. Get details of first match
    if works:
        info = env.cbeta_get_work_info(works[0])
        return query, json.dumps(info, ensure_ascii=False)
    return query, "No results found"
```

**Pattern 2: Find by translator, get table of contents**
```python
def run_env(env, query):
    # 1. Find works by Xuanzang
    result = env.cbeta_search_by_translator(creator="玄奘")
    works = result.get("result", {}).get("results", [])[:3]
    
    # 2. Get TOC for first work
    if works:
        toc = env.cbeta_get_toc(works[0]["work"])
        return query, json.dumps(toc, ensure_ascii=False)
    return query, "No works found"
```

**Pattern 3: Read specific passage with context**
```python
def run_env(env, query):
    # Get lines around a specific position
    result = env.cbeta_get_lines(
        linehead="T01n0001_p0066c25",
        before=5,
        after=10
    )
    return query, json.dumps(result, ensure_ascii=False)
```

---

## 4. Adding New Tools

### 4.1 To LocalEnv

1. Add method to `LocalEnv` class in `react_env.py`
2. Include comprehensive docstring with:
   - One-line description
   - Args documentation
   - Returns documentation
   - Example usage
3. The docstring is shown to LLM, so be clear and include examples

### 4.2 To CBETA (via MCP)

1. Add tool in `mcp_servers/CbetaMCP/tools/cebta/<category>/`
2. Add wrapper method in `LocalEnv` using `_run_mcp_tool()`
3. Update prompt in `prompt_engine.py`
4. Update this DEV_DOC.md

---

## 5. Configuration

### 5.1 MCP Connection

CBETA tools require MCP server running. Configure in `config/mcp_config.json`:

```json
{
  "mcpServers": {
    "cbeta": {
      "url": "http://localhost:8001/mcp/",
      "type": "streamable_http"
    }
  }
}
```

Start MCP server: `./scripts/start_mcp_server.sh`

### 5.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_AGENT_MCP_ENABLED` | `false` | Enable MCP integration |

---

## 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| CBETA tools return "MCP not connected" | Start MCP server, set `LOCAL_AGENT_MCP_ENABLED=true` |
| CBETA search returns 0 results | Check query (use Traditional Chinese), try shorter terms |
| Tool not found in env | Check method name matches MCP tool name exactly |
| Code execution timeout | Simplify code, reduce iterations, add timeouts |

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026-01-11 | Enhanced CBETA tools: added 12 new methods covering search, catalog, and content retrieval |
| 1.0.0 | Initial | Basic LocalEnv with document retrieval and 3 CBETA tools |
