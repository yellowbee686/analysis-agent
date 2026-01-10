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

**结论：优先使用 package（uv 管理），不使用 submodule。**
- **CAMEL：使用本地路径依赖（editable/路径依赖）**  
  目的：能跟进本地源码调试/修补，同时避免子模块带来的仓库体积与更新成本。  
  当无需本地改动时，可切换为 PyPI 版本以便于部署。
- **Streamlit：使用 PyPI 包**  
  Streamlit 仓库包含前端与构建链路，作为 submodule 引入成本高、对本项目无必要。

**uv 方式（建议）**
```
uv sync
uv run streamlit run app.py
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
  agent.py
  config.py
  history.py
  indexer.py
  llm.py
  mcp.py        # MCP toolkit management
  tools.py
cache/
  history/      # 对话历史存储目录
  indices/      # 索引缓存目录
config/
  mcp_config.json  # MCP 配置文件
scripts/
  start_mcp_server.sh   # MCP server 启动脚本
  stop_mcp_server.sh    # MCP server 停止脚本
mcp_servers/
  CbetaMCP/     # CBETA MCP server (git submodule)
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

## 8. 配置建议（.env）

使用本地 `.env` 管理模型与密钥，避免写入源码与 UI：

```
LOCAL_AGENT_MODEL_PLATFORM=openai
LOCAL_AGENT_MODEL_TYPE=gpt-4.1-mini-2025-04-14
OPENAI_API_KEY=your_key_here
```

说明：
- `LOCAL_AGENT_DATA_DIR` 指定文档目录（默认 `data`）。
- `LOCAL_AGENT_CHUNK_MAX_CHARS` 指定分块上限（默认 `1200`）。
- `LOCAL_AGENT_INDEX_DIR` 指定索引缓存目录（默认 `cache/indices`）。
- 未设置时回退到 CAMEL 默认值（由 `DEFAULT_MODEL_PLATFORM_TYPE` 与 `DEFAULT_MODEL_TYPE` 控制）。
- 可使用 `LOCAL_AGENT_SYSTEM_PROMPT` 覆盖系统提示词（可选）。
- `LOCAL_AGENT_MAX_TOKENS` 默认 65535，用于避免 CAMEL 关于 `max_tokens` 的警告。
- `LOCAL_AGENT_STREAM` 控制流式输出（默认 true）。
- `LOCAL_AGENT_NON_STREAM_PATTERNS` 指定不使用流式输出的模型名称模式，逗号分隔（默认 `gemini`）。
  - 例如：`gemini,claude` 会禁用所有包含 "gemini" 或 "claude" 的模型的流式输出。
  - 设为空字符串 `""` 可让所有模型使用流式输出。

### 8.1 多 Key 随机采样（参考 ttlive_strategy_agent）
当前项目支持从 `models/model_config/base.yaml` 读取模型端点列表，并按 `weight` 随机采样 API Key 与 Base URL。
建议在 `.env` 中配置 key，配置文件只保留 `api_key_env` 引用。

示例模型（内置）：
- `gemini-3-pro-preview-new`
- `gpt-5.2-2025-12-11`

当前端点配置采用 `.env` 提供 `base_url` 与 `weight`：
```
GEMINI_3_PRO_BASE_URL_1=...
GEMINI_3_PRO_API_KEY_1=...
GEMINI_3_PRO_WEIGHT_1=1
```

**禁用无效 Key**：将 `weight` 设为 `0` 可禁用该 endpoint，如：`GPT5_2_WEIGHT_1=0`

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
3. 启动 App：`uv run streamlit run app.py`
4. 在侧边栏查看 MCP 连接状态
5. 向 Agent 询问佛典相关问题，如"搜索法华经相关的佛典"

**CbetaMCP 提供的工具**：
- `cbeta_fulltext_search`: CBETA 全文检索
- 其他 CBETA API 工具（目录、工作等）

详见 `mcp_servers/CbetaMCP/readme.md`

**测试脚本**：
```bash
# 测试 GPT
uv run python scripts/smoke_openai_compatible.py -v --model gpt-5.2-2025-12-11

# 测试 Gemini
uv run python scripts/smoke_openai_compatible.py -v --model gemini-3-pro-preview-new
```

**调试模式**：设置 `DEBUG=1` 启用详细日志：
```bash
DEBUG=1 uv run streamlit run app.py
```

实测：部分网关会对 `gpt-i18n.byteintl.net` / `search-va.byteintl.net` 进行 301 跳转，导致 POST 变为 GET 引发 404。
建议在 `.env` 中直接填写最终网关域名（例如你们内部最终网关）以避免重定向。

---

## 8. 需要尽快确认的问题（待用户确认）
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
