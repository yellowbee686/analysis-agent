# AGENTS.md

## 0. Runtime Environment (Mandatory)

- Always use the project virtual environment at `/Users/bytedance/code/analysis-agent/.venv`.
- Never use system `python`, `pip`, or global site-packages for this project.
- Recommended commands:

```bash
uv sync
.venv/bin/python -m pytest
.venv/bin/streamlit run app.py
```

---

# 本地文件分析 Agent 开发文档（Living）

> 目标：本地文件（以 Markdown 为主）可检索、可问答、可生成报表的智能 Agent；
> 前端用 Streamlit，Agent 框架用 CAMEL。

---

## 1. 项目概述

### 1.1 目标与非目标

**目标**
- 从本地目录读取/索引 Markdown 文件（当前样例位于 `data/`）。
- 通过检索与工具调用回答用户问题、生成总结/报表。
- 基于 CAMEL 构建可扩展的 Agent（含 ReAct/工具调用循环）。
- 使用 Streamlit 提供可交互 UI（聊天 + 报表 + 索引状态）。

**非目标（暂不做）**
- 复杂权限/多租户/云端部署。
- 大规模分布式检索与容器编排。
- 全量 OCR 或非文本格式的批量解析（后续可扩展）。

### 1.2 现有数据

本地样例数据：`data/`  
主要为 `.md` 文件，目录层级较深（中文目录名、OCR 语料为主）。

---

## 2. 技术选型与依据（初稿）

### 2.1 CAMEL（Agent 框架）
- CAMEL 是模块化的多智能体框架，提供 Agent、Society、Memory、RAG 等核心构建块。  
  适合构建具备工具调用与检索能力的 Agent。
- `RetrievalToolkit` 支持从本地向量存储检索，并接受本地文件路径、URL 或字符串内容作为 `contents`。  
  适合作为“本地文件检索”的最小实现路径。

### 2.2 Streamlit（前端）
- 适合快速交互式原型与数据应用。
- `st.session_state` 可跨 rerun 保存会话状态；在多页应用中也可共享状态，但浏览器刷新/URL 跳转会重置状态。
- `st.cache_data` 适合缓存解析或检索结果，但基于 pickle，需要注意安全边界。
- `st.file_uploader` 支持本地文件上传（默认 200MB 上限，可通过配置调节）。

### 2.3 依赖管理（uv）与引入方式（结论）

**结论：依赖优先使用 package（uv 管理），但 CAMEL 例外，作为 submodule 引入。**
- **CAMEL：使用本仓库 submodule（`camel/`，跟踪 `dev` 分支）**  
  目的：在 fork 上独立修复 bug，同时保留 `master` 跟进上游的能力。  
  如需回退为 PyPI 版本，请移除 submodule 并改回包依赖。
- **Streamlit：使用 PyPI 包**  
  Streamlit 仓库包含前端与构建链路，作为 submodule 引入成本高、对本项目无必要。

**uv 方式（建议）**
```
git submodule update --init --recursive
uv sync
.venv/bin/streamlit run app.py
```

---

## 3. 系统架构（草图）

```
┌─────────────────────────────────────────────────────┐
│                     Streamlit UI                    │
│  - Chat (问答)   - 报表   - 索引管理/状态             │
└───────────────┬─────────────────────────────────────┘
                │
                v
┌─────────────────────────────────────────────────────┐
│                    CAMEL Agent                      │
│  ReAct Loop: Reason -> Tool -> Observe -> Answer     │
│  - Tool: 本地文件检索                                │
│  - Tool: 统计/报表生成                               │
└───────────────┬─────────────────────────────────────┘
                │
                v
┌─────────────────────────────────────────────────────┐
│            Local File Ingestion & Index             │
│  - 扫描目录  - 解析 MD  - 分块/元数据                │
│  - 向量/关键词检索索引                               │
└─────────────────────────────────────────────────────┘
```

---

## 4. 关键模块设计（MVP）

### 4.1 文件扫描与解析
- 扫描 `data/` 及其子目录，识别 `.md` 文件。
- 读取文本并做基础清洗（去多余空行、修复编码）。
- 生成文档元数据：文件路径、标题、目录层级、更新时间等。

### 4.2 分块与索引
- 分块策略（MVP）：按段落/标题分块；必要时限制 chunk 大小。
- 索引策略：  
  - **MVP**：使用 CAMEL `RetrievalToolkit` 直接检索本地路径；  
  - **升级**：引入向量检索（CAMEL Retriever 模块支持向量检索/关键词检索）。

### 4.3 Agent（ReAct + Tool）
- Agent 负责解析用户问题，判断是否需要工具检索。
- Tool 设计（初版）：
  - `retrieve_local_docs(query, paths, top_k, threshold)`
  - `summarize(texts, style)`
  - `build_report(query, retrieval_context, template)`
- CAMEL `BaseToolkit` + `FunctionTool` 方式注册工具，Agent 通过工具调用获取检索结果。

### 4.4 Streamlit UI（初版）
页面建议拆分：
- **主页**：聊天问答 + 检索结果引用
- **索引管理**：目录选择、重新索引、索引状态
- **报表**：模板选择、结果导出（md/pdf）

多页面实现建议使用 `st.navigation` 或 `pages/` 目录；注意使用 Streamlit 内建导航以保留 `st.session_state`。

---

## 5. 工具/接口设计（草案）

### 5.1 本地检索工具
输入：
- query: string
- contents: list[str]（本地文件路径）
- top_k: int
- similarity_threshold: float

输出：
- 聚合后的文本片段 + 元数据（来源文件、章节）

### 5.2 报表生成工具
输入：
- query / topic
- 检索上下文
- 模板（如：摘要/综述/列表/表格）
输出：
- Markdown 报表

---

## 6. 里程碑（建议）

1) **MVP 问答**
- 单目录扫描、建立索引
- CAMEL Agent + RetrievalToolkit + 基础 Streamlit 聊天

2) **报表生成**
- 模板化输出（摘要/综述/结构化列表）
- 导出为 `.md`

3) **检索增强**
- 引入向量检索（可选：Qdrant 等）
- chunk 优化与去重

---

## 6.1 当前代码结构（MVP 脚手架）

```
app.py
pyproject.toml
src/local_file_agent/
  config.py           # 配置管理
  history.py          # 对话历史管理
  indexer.py          # 本地文件索引
  llm.py              # LLM 配置与端点管理
  mcp.py              # MCP toolkit 管理 + 大响应包装
  session_storage.py  # Session 文件存储（MCP 大响应）
  tools.py            # 本地文档工具
  code_agent/
    react_agent.py    # React code agent loop
    react_env.py      # LocalEnv 接口层（注入 LLM context）
    env_tools.py      # LocalEnv 运行时实现（MCP/Semantic/FileToolkit 适配）
    prompt_engine.py  # Prompt 构造与 env API 注入
cache/
  history/            # 对话历史存储目录
  indices/            # 索引缓存目录
  sessions/           # Session 文件存储目录
    <session_id>/
      mcp_responses/  # MCP 大响应文件
config/
  mcp_config.json     # MCP 配置文件
scripts/
  start_mcp_server.sh # MCP server 启动脚本
  stop_mcp_server.sh  # MCP server 停止脚本
mcp_servers/
  CbetaMCP/           # CBETA MCP server (git submodule)
```

### 6.2 对话历史管理

历史管理功能（`history.py`）：

- **存储格式**：每个对话以 `.jsonl` 文件存储在 `cache/history/`，每行一条记录（meta 或 message）
- **自动清理**：启动时自动删除没有 user message 的空对话
- **选择对话**：点击左侧历史列表可切换对话，右侧主区域显示对应的消息
- **删除对话**：点击历史项旁的 ⋮ 按钮打开菜单，选择 "🗑️ Delete" 可删除对话，删除后自动切换到最新的对话
- **新建对话**：点击 "New conversation" 按钮创建新对话
- **切换模型**：切换 Model type 不会新建对话，当前对话可随时切换使用不同的模型

主要函数：
- `cleanup_empty_sessions(history_dir)`: 删除所有空对话（没有 user message）
- `list_sessions(history_dir)`: 列出所有对话，按修改时间排序
- `load_messages(path)`: 加载对话的消息列表
- `delete_session(path)`: 删除指定对话

---

## 7. 风险与限制

- `st.cache_data` 使用 pickle，有安全风险，避免缓存不可信输入。
- Streamlit 会话状态在页面刷新、URL 跳转时会重置，需要在 UI 设计上规避。
- 大目录与大文件解析可能造成卡顿，后续需异步/增量索引。
- 索引会缓存到 `cache/indices`，数据目录变更后需手动点击 `Rebuild index`。

---

## 8. 配置建议（.env / base.yaml）

使用本地 `.env` 管理运行时配置；API key 与端点配置写入私有
`models/model_config/base.yaml`（不纳入版本控制）：

```
LOCAL_AGENT_MODEL_PLATFORM=openai
LOCAL_AGENT_MODEL_TYPE=deepseek-ai/deepseek-v3.2
```

说明：
- `LOCAL_AGENT_DATA_DIR` 指定文档目录（默认 `data`）。
- `LOCAL_AGENT_CHUNK_MAX_WORDS` 指定每个 chunk 的词窗口大小（默认 `400`）。
- `LOCAL_AGENT_CHUNK_OVERLAP_WORDS` 指定相邻 chunk 的重叠词数（默认 `100`）。
- `LOCAL_AGENT_INDEX_DIR` 指定索引缓存目录（默认 `cache/indices`）。
- `LOCAL_AGENT_MODEL_CONFIG` 指定模型配置文件路径（默认使用 `models/model_config/base.yaml`，不存在则回退到 `base.yaml.example`）。
- 未设置时回退到 CAMEL 默认值（由 `DEFAULT_MODEL_PLATFORM_TYPE` 与 `DEFAULT_MODEL_TYPE` 控制）。
- 可使用 `LOCAL_AGENT_SYSTEM_PROMPT` 覆盖系统提示词（可选）。
- `LOCAL_AGENT_MAX_TOKENS` 默认 65535，用于避免 CAMEL 关于 `max_tokens` 的警告。
- `LOCAL_AGENT_STREAM` 控制流式输出（默认 true）。设为 `false` 可完全禁用流式输出。
  - 建议禁用场景：thinking model 的 reasoning 内容在 stream 模式下不显示、模型不支持 stream 模式下的 tool_call
- `LOCAL_AGENT_NON_STREAM_PATTERNS` 指定不使用流式输出的模型名称模式，逗号分隔（默认 `gemini`）。
  - 例如：`gemini,claude` 会禁用所有包含 "gemini" 或 "claude" 的模型的流式输出。
  - 设为空字符串 `""` 可让所有模型使用流式输出（除非 `LOCAL_AGENT_STREAM=false`）。

### 8.1 多 Key 随机采样（参考 ttlive_strategy_agent）
当前项目支持从 `models/model_config/base.yaml` 读取模型端点列表（如不存在则回退到
`models/model_config/base.yaml.example`），并按 `weight` 随机采样 API Key 与 Base URL。
可用 `LOCAL_AGENT_MODEL_CONFIG` 指定自定义配置文件路径。
建议将 API key 直接写入 `base.yaml`，配置文件本身保持私有。

示例模型（base.yaml.example）：
- `deepseek-ai/deepseek-v3.2`
- `minimaxai/minimax-m2.1`
- `z-ai/glm4.7`
- `openai/gpt-oss-120b`

当前端点配置使用 `base_url` / `api_key` / `weight`，并可通过 `use_azure`
指定客户端类型（AzureOpenAI / OpenAI）。

**context_window 配置**：可为每个模型配置 `context_window`（上下文窗口大小，单位：tokens）。
这个值用于 CAMEL 的 ChatAgent 自动压缩 memory。如果模型不在 CAMEL 的 `ModelType` 枚举中，
需要手动配置此值以避免启动时的警告。

```yaml
deepseek-ai/deepseek-v3.2:
  context_window: 163840  # DeepSeek-V3.2 的上下文窗口大小
  endpoints:
    - base_url: "https://integrate.api.nvidia.com/v1"
      api_key: "YOUR_NVIDIA_API_KEY"
      # ...
```

常见模型的 context window 大小：
- DeepSeek-V3.2: 163,840 tokens
- GPT-4o: 128,000 tokens
- Claude 3.5 Sonnet: 200,000 tokens

**禁用无效 Key**：将 `weight` 设为 `0` 可禁用该 endpoint。

可用 `LOCAL_AGENT_REQUEST_TIMEOUT` 覆盖请求超时时间（默认 120 秒）。
可用 `LOCAL_AGENT_AZURE_API_VERSION` 覆盖默认 Azure API 版本（默认 `2024-12-01-preview`）。

**注意**：代码使用 `AzureOpenAI` 客户端的 `base_url` 参数（而非 `azure_endpoint`），因为配置的 URL 已是完整路径。

### 8.2 MCP (Model Context Protocol) 配置

项目支持通过 MCP 协议接入外部工具服务。当前已集成 **CbetaMCP**（CBETA 佛典搜索工具）作为 submodule。

**环境变量配置**：
```
LOCAL_AGENT_MCP_ENABLED=true
LOCAL_AGENT_MCP_CONFIG=config/mcp_config.json
```

**启动 MCP Server**：
```bash
# 前台启动（用于调试）
./scripts/start_mcp_server.sh

# 后台启动
./scripts/start_mcp_server.sh --bg

# 停止后台服务
./scripts/stop_mcp_server.sh
```

默认端口为 `8001`，可通过 `MCP_PORT` 环境变量覆盖。

**MCP 配置文件** (`config/mcp_config.json`)：
```json
{
  "mcpServers": {
    "cbeta": {
      "url": "http://localhost:8001/mcp/sse",
      "transport": "sse",
      "description": "CBETA Buddhist Scripture Search MCP Server"
    }
  }
}
```

**使用流程**：
1. 启动 MCP Server：`./scripts/start_mcp_server.sh --bg`
2. 设置环境变量启用 MCP：`LOCAL_AGENT_MCP_ENABLED=true LOCAL_AGENT_MCP_CONFIG=config/mcp_config.json`
3. 启动 App：`.venv/bin/streamlit run app.py`
4. 在侧边栏查看 MCP 连接状态
5. 向 Agent 询问佛典相关问题，如"搜索法华经相关的佛典"

**CbetaMCP 提供的工具**：
- `cbeta_fulltext_search`: CBETA 全文检索
- 其他 CBETA API 工具（目录、工作等）

详见 `mcp_servers/CbetaMCP/readme.md`

**MCP 超时配置**：
- MCP 工具执行默认超时为 60 秒（在 `mcp.py` 中配置）
- camel-ai MCPClient 默认超时是 10 秒，对于慢速 API（如 CBETA）会导致超时
- 如遇到 "Timed out while waiting for response to ClientRequest" 错误，可增加超时时间
- 注意：工具超时后，camel 框架可能会出现消息格式错误（缺少 assistant tool_calls），这是上游 bug

**MCP streamable-http 事件循环问题及修复** (2026-01-10)：
- **问题**：CAMEL 的同步流式响应使用 `ThreadPoolExecutor` 执行工具调用，每个线程会创建新的事件循环。但 MCP 的 `streamable-http` 传输是持久连接，session 绑定到原始事件循环。从新线程/事件循环调用会导致请求永远无法完成（超时）。
- **症状**：MCP 工具调用超时 30-60 秒，日志显示 "Timed out while waiting for response to ClientRequest"，但直接调用 MCP API 只需几秒。
- **修复**：在 `app.py` 中：
  1. MCP 连接时保存事件循环到 `st.session_state["mcp_event_loop"]`
  2. 新增 `run_agent_step_async()` 函数使用保存的事件循环运行 `agent.astep()`
  3. 当 MCP 启用时，使用异步模式执行 agent step，确保所有 MCP 调用在同一事件循环中执行
- **配置**：MCP 配置改为使用 `streamable_http` 类型（更高效）：
  ```json
  {
    "mcpServers": {
      "cbeta": {
        "url": "http://localhost:8001/mcp/",
        "type": "streamable_http",
        "description": "CBETA Buddhist Scripture Search MCP Server"
      }
    }
  }
  ```

**MCP Optional 类型参数问题修复** (2026-01-10)：
- **问题**：CAMEL 的 `MCPClient.generate_function_from_mcp_tool` 方法无法处理 JSON Schema 中类型为数组的参数（如 `["string", "null"]`）。当 MCP 工具使用 Optional 类型（如 `str | None`）时，fastmcp 会生成 `type: ["string", "null"]` 这样的 schema，导致 `type_map.get()` 调用报错 "unhashable type: 'list'"。
- **症状**：日志显示 "Failed to convert tool xxx: unhashable type: 'list'"，MCP 工具无法被正确加载。
- **修复**：在 `camel/camel/utils/mcp_client.py` 的 `generate_function_from_mcp_tool` 方法中添加类型数组处理：
  - 检测 `param_type` 是否为 list
  - 如果是，过滤掉 "null" 类型，使用第一个非 null 类型作为 Python 类型
  - 这样 `str | None` 会被正确解析为 `str` 类型

**并行工具调用消息格式问题修复** (2026-01-11)：
- **问题**：当模型返回并行工具调用时（一次请求返回多个 tool_calls），CAMEL 为每个工具调用创建独立的 assistant 消息。但 OpenAI API 要求并行工具调用应该是**一个** assistant 消息包含**多个** tool_calls，否则部分 API（如 NVIDIA）会返回 500 错误 "No tool calls but found tool output"。
- **症状**：工具调用后下一轮请求失败，日志显示 `InternalServerError: Error code: 500 - {'error': {'message': 'No tool calls but found tool output', ...}}`。
- **修复**：在 `camel/camel/memories/context_creators/score_based.py` 的 `create_context` 方法中添加消息合并逻辑：
  - 新增 `_merge_parallel_tool_calls()` 方法
  - 检测连续的 assistant 消息（带 tool_calls 但无 content）
  - 将它们合并为单个 assistant 消息，包含所有 tool_calls
  - 这样发送给模型的消息格式符合 OpenAI API 规范

**测试脚本**：
```bash
# 测试 GPT
.venv/bin/python scripts/smoke_openai_compatible.py -v --model gpt-5.2-2025-12-11

# 测试 Gemini
.venv/bin/python scripts/smoke_openai_compatible.py -v --model gemini-3-pro-preview-new
```

**调试模式**：设置 `DEBUG=1` 启用详细日志：
```bash
DEBUG=1 .venv/bin/streamlit run app.py
```

实测：部分网关会对 `gpt-i18n.byteintl.net` / `search-va.byteintl.net` 进行 301 跳转，导致 POST 变为 GET 引发 404。
建议在 `.env` 中直接填写最终网关域名（例如你们内部最终网关）以避免重定向。

---

## 8. MCP 大响应处理与 FileToolkit 集成

### 8.1 设计背景

CBETA MCP 工具返回的佛典内容通常非常大（如经文HTML、搜索结果等），直接返回到 LLM 上下文会造成：
- 上下文污染：大量无关内容占用有限的上下文窗口
- 成本增加：更多 token 意味着更高的 API 成本
- 效率下降：LLM 需要处理大量冗余信息

### 8.2 解决方案：Session-based File Storage

当 MCP 工具返回的内容超过阈值时，自动保存到文件并返回元数据：

```
cache/sessions/<session_id>/mcp_responses/
├── cbeta_fulltext_search_q=法鼓_abc12345_143025.json
├── get_juan_html_work=T0001_juan=1_def67890_143230.json
└── ...
```

**工作流程**：
1. MCP 工具返回响应
2. 检查响应大小是否超过阈值（默认 8000 字符）
3. 如超过，保存到 session 对应的文件夹
4. 返回元数据（文件路径、大小、预览等）
5. LLM 使用 FileToolkit 的 `search_files` 或 `read_file` 来访问内容

### 8.3 配置

**环境变量**：
```bash
# MCP 大响应阈值（字符数，默认 8000）
LOCAL_AGENT_MCP_CONTENT_THRESHOLD=8000
```

### 8.4 相关模块

- `session_storage.py`: Session 文件存储管理
- `mcp.py`: MCP 工具包装器，拦截大响应
- `agent.py`: 集成 FileToolkit 供 LLM 使用

### 8.5 FileToolkit 工具

Agent 现在包含以下文件操作工具：

| 工具 | 功能 |
|------|------|
| `write_to_file` | 写入文件（支持多种格式） |
| `read_file` | 读取文件内容 |
| `edit_file` | 编辑文件（替换内容） |
| `search_files` | 在文件中搜索文本模式 |

**搜索示例**：
```python
# LLM 可以调用 search_files 在保存的 MCP 响应中搜索
search_files(
    pattern="法鼓",
    file_types=["json", "txt"],
    path="/path/to/session/mcp_responses"
)
```

### 8.6 SemanticScholarToolkit

Agent 集成了 Semantic Scholar 学术论文搜索工具：

| 工具 | 功能 |
|------|------|
| `fetch_paper_data_title` | 按论文标题搜索 |
| `fetch_paper_data_id` | 按论文 ID 获取详情 |
| `fetch_bulk_paper_data` | 批量搜索论文（支持复杂查询） |
| `fetch_recommended_papers` | 获取推荐论文 |
| `fetch_author_data` | 获取作者信息 |

**注意**：与 `GoogleScholarToolkit` 不同，`SemanticScholarToolkit` 不需要在初始化时指定作者，更适合通用搜索场景。

---

## 9. 需要尽快确认的问题（待用户确认）
- 需要哪些报表模板？优先级？
- LLM/Embedding 的供应方式（本地/远程）？

---

## 9. 参考资料（权威来源）

- CAMEL 文档主页：https://docs.camel-ai.org/
- CAMEL RetrievalToolkit：https://docs.camel-ai.org/reference/camel.toolkits.retrieval_toolkit
- CAMEL Retrievers 概览：https://docs.camel-ai.org/key_modules/retrievers
- Streamlit Session State：https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- Streamlit Cache Data：https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data
- Streamlit file_uploader：https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
- Streamlit multipage apps：https://docs.streamlit.io/develop/concepts/multipage-apps/overview
