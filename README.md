# 本地文件分析 Agent

一个用于**本地 Markdown/文本资料检索、问答与报表生成**的轻量应用。前端使用 Streamlit，Agent 框架使用 CAMEL，并通过 MCP server 支持大响应落盘。

## 快速开始

### 前置条件
- Python 3.12
- Git
- macOS / Windows（Windows 需 Git Bash 或 WSL）

### macOS
```bash
./scripts/pull.sh
./scripts/install.sh
# 配置 .env 与 models/model_config/base.yaml
./scripts/start.sh
```

### Windows
```powershell
.\scripts\pull.cmd
.\scripts\install.cmd
# 配置 .env 与 models/model_config/base.yaml
.\scripts\start.cmd
```

> 说明：Windows 脚本是对 `*.sh` 的包装，默认调用 Git Bash 的 `bash`。如果使用 WSL，请在 WSL 里直接运行 `bash scripts/*.sh`。

## 配置说明
- 环境变量：复制 `.env.example` 为 `.env` 后按需修改。
- 模型配置：编辑 `models/model_config/base.yaml`。

## 目录结构（简要）
```
app.py                      # Streamlit 入口
src/local_file_agent/       # Agent 核心逻辑
config/                     # MCP 配置
scripts/                    # 启动/安装脚本
models/model_config/        # 模型配置
```
