# piY

[![CI](https://github.com/Swyoung-dd/pi-base-python/actions/workflows/ci.yml/badge.svg)](https://github.com/Swyoung-dd/pi-base-python/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[项目网站](https://swyoung-dd.github.io/piy-dev/)

piY 是一个用 Python 构建的终端 AI 编码 agent。它将多种 LLM provider、流式响应、
工具调用、持久会话和上下文管理整合为统一运行时，并同时提供交互式终端、单次命令、
Python SDK 和 JSONL RPC 接口。

## 目录

- [核心能力](#核心能力)
- [快速开始](#快速开始)
- [配置](#配置)
- [使用方式](#使用方式)
- [项目架构](#项目架构)
- [开发](#开发)
- [安全说明](#安全说明)
- [License](#license)

## 核心能力

- **多模型支持**：内置 OpenAI、Anthropic 和 DeepSeek 模型，可接入 Groq、Mistral、
  OpenRouter、xAI、LM Studio、Ollama 及其他 OpenAI 兼容服务。
- **完整 Agent 循环**：流式文本与思考输出、结构化工具调用、并行工具执行、取消与重试。
- **编码工具集**：读取、写入、精确编辑、目录浏览、文件查找、文本搜索和跨平台命令执行。
- **会话与上下文**：JSONL 持久会话、会话恢复与分支、模型切换、自动和主动上下文压缩。
- **子代理**：按任务派生独立 agent，支持只读分析、完整工具集、模型覆盖和有限深度嵌套。
- **项目级定制**：支持 `AGENTS.md` / `CLAUDE.md`、Skills、Prompt Templates、主题和
  Python 扩展。
- **多种集成方式**：交互式 TUI、print mode、Python SDK 和 stdin/stdout JSONL RPC。
- **跨平台运行**：支持 Windows、macOS 和 Linux，并在三种系统上持续集成测试。

## 快速开始

### 环境要求

- Python 3.12 或更高版本
- 推荐使用 [uv](https://docs.astral.sh/uv/) 和仓库内的 `uv.lock`
- 至少一个可用模型的 API Key，本地模型服务除外

### 从源码运行

```bash
git clone https://github.com/Swyoung-dd/pi-base-python.git
cd pi-base-python
python -m pip install uv==0.12.0
uv sync --locked --all-extras
```

首次运行时选择模型并配置凭据：

```bash
uv run piY --setup
uv run piY
```

也可以安装到当前 Python 环境：

```bash
python -m pip install .
piY --setup
piY
```

Windows PowerShell 无法识别 `piY` 时，可直接使用模块入口：

```powershell
.\.venv\Scripts\python.exe -m pi --setup
.\.venv\Scripts\python.exe -m pi
```

## 配置

项目配置位于 `.piy/config.yaml`。`piY --setup` 会写入所选模型、provider 和思考级别；
也可以手动维护配置：

```yaml
model: gpt-4o-mini
provider: openai
thinking_level: off
temperature: 0.2
max_tokens: 4096
theme: default
enable_skills: true
enable_prompt_templates: true
enable_context_files: true
```

以下环境变量可覆盖核心模型配置：

| 变量 | 作用 |
| --- | --- |
| `PIY_MODEL` | 模型 ID |
| `PIY_PROVIDER` | Provider ID |
| `PIY_THINKING` | 思考级别 |

常用 provider 凭据变量：

| Provider | 环境变量 |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| xAI | `XAI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

也可以使用通用形式 `PIY_API_KEY_<PROVIDER>`。交互式配置收集的凭据保存在用户目录
`~/.piy/auth.json`；输入时使用掩码，并阻止疑似 API Key 进入模型上下文或新的输入历史。

项目自定义模型可写入 `.piy/models.yaml`。模型定义和扩展配置见
[扩展指南](docs/extensions.md)。

## 使用方式

### 交互模式

```bash
uv run piY
```

交互模式支持流式输出、工具执行状态、模型选择、思考级别切换和会话管理。常用命令包括
`/model`、`/thinking`、`/compact` 和 `/help`。

### 单次任务

```bash
uv run piY -p "Summarize @README.md"
uv run piY -p "Review the session implementation" --thinking high
uv run piY -p "List the public APIs" --output json
```

提示中的 `@path` 会展开为文件内容。`--output` 支持 `text`、`json` 和 `jsonl`。

### 会话恢复

```bash
uv run piY --list-sessions
uv run piY --session <session-id>
uv run piY --continue
```

会话默认存储在当前项目的 `.piy/sessions`。`--continue` 恢复最近更新的会话。

### 子代理

主 agent 可以把独立调查或实现任务委派给 `subagent`。默认 `analysis` 模式只允许
read、ls、find 和 grep；`full` 模式额外允许写文件、编辑、执行命令和继续派生子代理。

```text
请使用 subagent 分析 src/pi/agent 的模块边界并汇报风险。
```

子代理默认最多运行 8 次模型调用，嵌套深度最多 2 层。其最终回答会作为工具结果返回
主对话。

### 其他入口

```bash
piY --list-models
piY --auth-list
piY --login xai
piY --rpc
python -m pi --version
piY --help
```

`--rpc` 在 stdin/stdout 上启动长期运行的 JSONL 服务，适合编辑器、桌面应用或其他进程
集成。Python 嵌入接口位于 `pi.coding_agent.sdk`。

## 项目架构

```text
src/pi/
|-- ai/             # 统一 LLM 类型、模型目录、凭据和 provider 适配
|-- agent/          # Agent 状态、模型工具循环、压缩和会话存储
|-- coding_agent/   # 编码场景装配、CLI、SDK、RPC、扩展与工具
`-- tui/            # 交互式终端和选择器
```

依赖方向保持为“交互入口 → 编码场景 → Agent 内核 → LLM 抽象与 provider”。详细的数据流、
设计取舍和扩展协议见：

- [项目架构](docs/architecture.md)
- [设计思想](docs/design-principles.md)
- [开发指南](docs/development.md)
- [扩展指南](docs/extensions.md)

## 开发

安装开发依赖后运行完整检查：

```bash
uv run --locked ruff check src tests scripts
uv run --locked python scripts/generate_models.py --check
uv run --locked python -m pytest -q
uv run --locked python -m build
uv run --locked python scripts/smoke_wheel.py
```

CI 覆盖 Ubuntu、Windows、macOS 以及 Python 3.12、3.13。版本号位于
`src/pi/__init__.py`，构建产物输出到 `dist/`。

提交变更前请阅读 [开发指南](docs/development.md)，并同步更新受影响的测试和文档。

## 安全说明

项目级 `.piy` 资源可以注入提示、模型定义或 Python 扩展。piY 会在加载受保护资源前
请求项目信任，非交互环境默认拒绝未授权资源，也可使用 `--approve` 或 `--no-approve`
控制当前运行。

项目信任不等同于执行沙箱。文件工具和命令工具仍以当前进程权限运行。处理不可信仓库、
提示或扩展时，应在容器、虚拟机或受限操作系统账户中运行，并避免暴露无关凭据。

## License

本项目基于 [MIT License](LICENSE) 开源。
