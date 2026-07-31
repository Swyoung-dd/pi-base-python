# 开发指南

## 1. 准备环境

项目要求 Python 3.12 或更高版本，推荐使用仓库锁定的 uv 环境：

```bash
python -m pip install uv==0.12.0
uv sync --locked --all-extras
```

安装完成后可直接从源码运行：

```bash
uv run piY --help
uv run piY --list-models
uv run piY -p "Summarize @README.md"
```

本地真实 provider 调试需要相应 API Key。单元测试应使用假的事件流或 HTTP mock，不能
依赖真实凭据和外部服务。

## 2. 理解改动落点

开始编码前先确定责任层：

| 需求 | 首选位置 |
| --- | --- |
| 公共消息、模型、事件语义 | `src/pi/ai/types.py`、`streaming.py` |
| 某厂商协议或鉴权差异 | `src/pi/ai/providers`、`auth.py`、`oauth*.py` |
| 模型与工具反馈循环 | `src/pi/agent/agent_loop.py` |
| 跨轮状态、队列、取消、压缩 | `src/pi/agent/agent.py`、`compaction.py` |
| 会话持久化 | `src/pi/agent/session` |
| 编码工具与资源加载 | `src/pi/coding_agent/tools`、`runtime.py` |
| 命令行、SDK、RPC | `src/pi/coding_agent/cli.py`、`sdk.py`、`rpc.py` |
| 终端显示和交互命令 | `src/pi/tui` |

避免从 provider 直接引用 TUI，或在多个入口分别装配一套不同工具。跨入口共享的行为应先
下沉到 Agent、`CodingAgent` 或运行时组合根。

## 3. 常见变更流程

### 3.1 修改公共类型

1. 更新 `pi.ai.types` 或 `pi.agent.types`。
2. 检查 OpenAI、Anthropic 等 provider 的序列化与解析。
3. 检查 agent 与 LLM 消息之间的转换。
4. 检查会话的 Pydantic 序列化和 RPC/JSON 输出。
5. 增加往返序列化、流事件和兼容性测试。

公共类型影响面较大，不要仅根据单个 provider 的返回样例设计字段。

### 3.2 新增内置工具

1. 在 `src/pi/coding_agent/tools` 创建模块，定义 JSON Schema 和异步 `execute`。
2. 返回 `AgentToolResult`，可预期错误使用 `is_error=True`，不要直接向终端输出。
3. 在工具包 `__init__.py` 导出创建函数。
4. 在 `runtime.build_tools()` 注册。
5. 为成功、参数错误、执行异常、路径或取消边界增加测试。

同一轮工具默认可能并行执行。工具若修改共享状态，应明确其并发语义。

### 3.3 修改会话格式

会话是面向用户的持久数据。新增条目类型时需要：

1. 保持旧 JSONL 记录仍能加载。
2. 定义该条目在 `get_context_messages()` 和分支恢复中的行为。
3. 覆盖末尾残缺行、非末尾损坏行、分支与压缩检查点测试。
4. 如果无法向后兼容，先引入显式格式版本和迁移路径。

### 3.4 修改系统提示或资源顺序

系统提示由配置、扩展、内置提示、上下文文件和 Skills 元数据共同组成。修改时要验证：

- 各片段顺序是否仍符合预期。
- 项目未受信任时，受保护资源是否确实未加载。
- 新内容是否值得长期占用每次请求的上下文。
- Windows 与 POSIX 平台说明是否一致。

## 4. 测试策略

测试集中在 `tests`，并按功能拆分。变更应优先增加最接近责任层的测试：

- provider：请求转换、SSE/流解析、错误与重试。
- agent loop：事件顺序、工具回填、并行/串行、取消和最大轮数。
- session：追加、恢复、分支、压缩检查点和损坏容错。
- coding agent：配置优先级、信任边界、资源发现和扩展隔离。
- CLI/SDK/RPC/TUI：外部协议和主要用户流程。

测试应确定性运行，不访问网络，不依赖用户主目录中已有的 `.piy` 文件，并使用临时目录
隔离文件系统状态。

运行快速检查：

```bash
uv run --locked ruff check src tests scripts
uv run --locked python -m pytest -q
```

模型目录发生变化时额外运行：

```bash
uv run --locked python scripts/generate_models.py --check
```

## 5. 模型目录

模型源数据位于 `src/pi/ai/models.catalog.json`，生成结果位于
`src/pi/ai/models_generated.py`。不要只手工修改生成文件。

更新源数据后执行：

```bash
uv run --locked python scripts/generate_models.py
uv run --locked python scripts/generate_models.py --check
```

项目级临时模型可放在 `.piy/models.yaml`，不应为私有部署模型修改内置目录。

## 6. 构建与发布验证

版本号位于 `src/pi/__init__.py`。发布前执行完整检查：

```bash
uv run --locked ruff check src tests scripts
uv run --locked python scripts/generate_models.py --check
uv run --locked python -m pytest -q
uv run --locked python -m build
uv run --locked python scripts/smoke_wheel.py
```

CI 在 Ubuntu、Windows、macOS 以及 Python 3.12、3.13 的组合上运行同类检查。构建产物
位于 `dist`，wheel smoke test 会在临时虚拟环境验证安装后的版本、帮助和模型列表入口。

## 7. 完成标准

一个可合并的变更至少应满足：

1. 代码位于正确层，没有为单个入口复制核心逻辑。
2. 失败、取消、并发和持久化边界已按风险覆盖测试。
3. Ruff、模型目录检查和测试通过。
4. 用户可见行为、公共协议或设计约束变化时，相关文档已同步。
5. 不提交凭据、本地 `.piy` 状态、虚拟环境或临时构建文件。

