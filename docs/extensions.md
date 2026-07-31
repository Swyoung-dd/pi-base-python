# 扩展指南

piY 提供多种扩展层级。选择最窄的机制可以降低维护成本：

| 目标 | 推荐机制 |
| --- | --- |
| 增加一个模型配置 | `.piy/models.yaml` |
| 增加新的厂商协议 | `BaseProvider` 实现 |
| 增加模型可调用能力 | `AgentTool` |
| 提供任务专用操作说明 | Skill |
| 提供可复用提示入口 | Prompt Template |
| 同时注册工具、模型、命令或生命周期逻辑 | Python Extension |

## 1. 自定义模型

项目模型文件为 `.piy/models.yaml`，可以是列表，也可以放在 `models` 键下。记录遵循
`pi.ai.types.Model`：

```yaml
models:
  - id: example-chat
    name: Example Chat
    api: openai-chat-completions
    provider: example
    base_url: https://api.example.com/v1
    reasoning: false
    input: [text]
    context_window: 128000
    max_tokens: 8192
```

模型由 `(provider, id)` 唯一标识。只传模型 ID 时如果匹配多个 provider，解析会失败，
调用方应同时指定 provider。

项目模型文件属于受保护资源，只有项目被信任时才加载。

## 2. Provider

新协议实现应继承 `pi.ai.providers.base.BaseProvider`，至少提供 `provider_id` 和异步
`stream()`：

```python
from pi.ai.providers.base import BaseProvider


class ExampleProvider(BaseProvider):
    @property
    def provider_id(self) -> str:
        return "example"

    async def stream(self, model, context, options=None):
        # 转换统一 Context，发起请求，并返回 EventStream。
        ...
```

实现必须遵守以下契约：

- 输入只依赖统一 `Model`、`Context` 和 `StreamOptions`。
- 输出为 `EventStream`，正常或异常都以 `done` 或 `error` 事件终止。
- 最终助手消息包含准确的 provider、model、stop reason 和 usage。
- API Key 优先从 `StreamOptions` 读取，再通过统一凭据解析逻辑获取。
- HTTP 重试只覆盖可重试错误，并保留 retry 事件供上层观察。

若服务兼容 OpenAI Chat Completions，可优先使用
`register_openai_compatible()`，而不是复制完整 provider。

## 3. AgentTool

工具由名称、描述、参数 JSON Schema 和异步执行函数组成：

```python
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent


async def execute(call: AgentToolCall, context) -> AgentToolResult:
    name = call.arguments.get("name", "world")
    return AgentToolResult(
        tool_call_id=call.id,
        tool_name=call.name,
        content=[TextContent(text=f"Hello, {name}")],
    )


def create_hello_tool() -> AgentTool:
    return AgentTool(
        name="hello",
        description="Return a greeting.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
        execute=execute,
    )
```

工具实现约束：

- 参数始终视为不可信输入，并在执行前校验类型和范围。
- 可预期失败返回 `AgentToolResult(is_error=True)`；未预期异常由 loop 转换为错误结果。
- 使用 `ToolContext.cwd` 解析工作目录，不依赖进程启动时的隐式全局状态。
- 输出可能进入模型上下文，应限制体积并避免返回凭据。
- 默认可能与同一轮其他工具并行执行。

内置工具在 `runtime.build_tools()` 注册。只服务特定部署的工具通常更适合通过 Extension
注册。

## 4. Skill

Skill 是带 YAML frontmatter 的 Markdown 文件，常见目录结构为：

```text
.piy/skills/example/SKILL.md
```

最小内容：

```markdown
---
name: example
description: Use when handling Example service deployment tasks.
---

# Example workflow

Follow these steps when the skill matches a task...
```

名称必须由小写字母、数字和连字符组成，最长 64 个字符；描述不能为空且最长 1024 个
字符。设置 `disable-model-invocation: true` 后，Skill 不会暴露给模型自动选择。

发现顺序是配置路径、项目 `.piy/skills`、用户 `~/.piy/skills`。同名 Skill 保留最先
加载的定义。运行时只把元数据放入系统提示，模型匹配后再读取正文及相对资源。

## 5. Prompt Template

Prompt Template 用于把常用提示注册为交互入口，支持项目和用户目录发现。模板由 Markdown
文件及其 frontmatter 定义，加载逻辑位于 `pi.coding_agent.prompt_templates`。

模板适合参数化的重复请求；如果内容是模型在特定任务中必须遵守的完整流程，应使用 Skill；
如果需要执行 Python 逻辑或注册工具，应使用 Extension。

## 6. Python Extension

扩展模块暴露 `setup(context)`，可以是同步或异步函数：

```python
def setup(context):
    context.add_tool(create_hello_tool())
    context.add_system_prompt("Use the hello tool only for greeting tasks.")
```

`ExtensionContext` 支持：

- `add_tool()`：增加工具。
- `add_model()`：注册运行时模型。
- `add_provider()`：注册 provider。
- `add_system_prompt()`：增加系统提示片段。
- `add_command()`：增加 TUI 命令。
- `on()`：订阅 `session_start`、`session_shutdown`、`session_switch` 或 `agent_event`。

项目扩展路径在 `.piy/config.yaml` 的 `extensions` 中声明，相对路径基于 `.piy` 目录：

```yaml
extensions:
  - extensions/example.py
enable_entrypoint_extensions: false
```

也可以发布 `piy.extensions` Python entry point，但自动发现默认关闭，需要明确启用
`enable_entrypoint_extensions`。

扩展加载失败会阻止运行时构建，因为此时能力集合不完整；已经加载的扩展在处理生命周期
事件时发生异常，则会被记录为 `ExtensionFailure`，其他处理器继续运行。

内置工具与扩展工具不能重名，扩展命令之间也不能重名。

## 7. 信任与测试

模型文件、项目 Skills、Prompts、Themes、系统提示和 Python Extension 都可能影响模型
行为或执行代码，因此受项目信任机制保护。扩展作者不应把“项目已受信任”理解为拥有无限
权限的授权，仍应最小化文件、网络、子进程和凭据访问。

扩展测试至少应覆盖注册结果、参数校验、异步 setup、名称冲突和事件处理器失败隔离。
provider 与工具测试应使用本地 fake，不依赖真实网络和用户凭据。
