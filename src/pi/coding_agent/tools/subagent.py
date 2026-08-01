"""Sub-agent 工具。

派生一个子 Agent 完成独立任务。子 agent 运行自己的 agent 循环，使用受限工具集，
并把最终回答作为工具结果返回，主 agent 可以在不阻塞自身循环的情况下委派工作。
"""

from __future__ import annotations

import asyncio
from typing import Any

from pi.agent.agent import Agent, AgentOptions
from pi.agent.session import MemoryStorage
from pi.agent.tools.base import ToolContext, truncate_output
from pi.agent.types import AgentAssistantMessage, AgentTool, AgentToolCall, AgentToolResult
from pi.ai.models import get_model
from pi.ai.types import TextContent

MAX_SUBAGENT_DEPTH = 2
DEFAULT_MAX_TURNS = 8
_CONTEXT_TAIL_MESSAGES = 6

_SUBAGENT_SYSTEM_PROMPT = """You are a sub-agent of piY, spawned by the main coding
agent to complete one task.

Working directory: {cwd}
Available tools: {tools}

Guidelines:
- Complete the task autonomously; make reasonable assumptions instead of asking for clarification.
- Use tools only when necessary; prefer reading over guessing.
- If the task is impossible or unsafe, say so directly instead of fabricating results.
- When done, report the outcome concisely: key findings, any files changed, and remaining risks."""

PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task for the sub-agent. Be specific and self-contained.",
        },
        "model": {
            "type": "string",
            "description": "Optional model override as 'provider/model' or a unique model id.",
        },
        "max_turns": {
            "type": "integer",
            "description": "Maximum model-tool rounds for the sub-agent. Default: 8.",
        },
        "tools": {
            "type": "string",
            "enum": ["none", "analysis", "full"],
            "description": (
                "Tool set for the sub-agent: 'analysis' (read-only, default), "
                "'full' (all tools, sub-agents can nest), or 'none'."
            ),
        },
        "include_context": {
            "type": "boolean",
            "description": "Include the last few parent conversation turns as context.",
        },
    },
    "required": ["task"],
}


def _error(call: AgentToolCall, message: str) -> AgentToolResult:
    """构造一个错误结果。"""
    return AgentToolResult(
        tool_call_id=call.id,
        tool_name=call.name,
        content=[TextContent(text=message)],
        is_error=True,
    )


def _resolve_model(raw: Any, parent: Agent) -> Any:
    """解析模型覆盖参数；未指定时继承父 agent 的模型。"""
    if raw is None:
        return parent.state.model
    value = str(raw).strip()
    if not value:
        return None
    if "/" in value:
        provider, _, model_id = value.partition("/")
        return get_model(model_id, provider)
    return get_model(value)


def _build_child_tools(mode: str) -> list[AgentTool]:
    """按模式构建子 agent 工具集：none / analysis（只读）/ full（全部）。"""
    if mode == "none":
        return []
    if mode == "analysis":
        from pi.coding_agent.tools.find import create_find_tool
        from pi.coding_agent.tools.grep import create_grep_tool
        from pi.coding_agent.tools.ls import create_ls_tool
        from pi.coding_agent.tools.read import create_read_tool

        return [
            create_read_tool(),
            create_ls_tool(),
            create_find_tool(),
            create_grep_tool(),
        ]
    from pi.coding_agent.tools.bash import create_bash_tool
    from pi.coding_agent.tools.edit import create_edit_tool
    from pi.coding_agent.tools.write import create_write_tool

    return [
        *(_build_child_tools("analysis")),
        create_write_tool(),
        create_edit_tool(),
        create_bash_tool(),
        create_subagent_tool(),
    ]


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    """创建并运行子 agent，把其最终回答作为工具结果返回。"""
    task = call.arguments.get("task", "")
    if not isinstance(task, str) or not task.strip():
        return _error(call, "task must be a non-empty string")
    if ctx is None:
        return _error(call, "subagent tool requires tool context")
    parent = ctx.state.get("agent")
    if not isinstance(parent, Agent):
        return _error(call, "subagent tool requires a running agent")
    depth = ctx.state.get("subagent_depth", 0)
    if not isinstance(depth, int) or depth < 0:
        depth = 0
    if depth >= MAX_SUBAGENT_DEPTH:
        return _error(call, f"sub-agent depth limit ({MAX_SUBAGENT_DEPTH}) reached")

    tools_mode = call.arguments.get("tools", "analysis")
    if tools_mode not in ("none", "analysis", "full"):
        return _error(call, f"invalid tools mode: {tools_mode}")

    max_turns = call.arguments.get("max_turns", DEFAULT_MAX_TURNS)
    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or not 1 <= max_turns <= 100:
        return _error(call, "max_turns must be an integer between 1 and 100")

    model = _resolve_model(call.arguments.get("model"), parent)
    if model is None:
        return _error(call, "unknown or ambiguous model: use provider/model or a unique model id")

    child_tools = _build_child_tools(tools_mode)
    tool_names = [tool.name for tool in child_tools]
    system_prompt = _SUBAGENT_SYSTEM_PROMPT.format(
        cwd=ctx.cwd,
        tools=", ".join(tool_names) if tool_names else "none",
    )

    prompt = task
    if call.arguments.get("include_context") is True:
        tail = parent.state.messages[-_CONTEXT_TAIL_MESSAGES:]
        if tail:
            from pi.agent.compaction import format_messages_for_summary

            transcript = truncate_output(format_messages_for_summary(tail), 20000)
            prompt = f"{task}\n\nParent conversation tail:\n{transcript}"

    child_tool_context = ToolContext(cwd=ctx.cwd, state={"subagent_depth": depth + 1})
    child = Agent(
        AgentOptions(
            model=model,
            system_prompt=system_prompt,
            tools=child_tools,
            stream_fn=parent.stream_fn,
            session_storage=MemoryStorage(),
            tool_context=child_tool_context,
            thinking_level=parent.thinking_level,
            temperature=parent.temperature,
            max_tokens=parent.max_tokens,
            max_turns=max_turns,
        )
    )

    collected: dict[str, Any] = {
        "texts": [],
        "errors": [],
        "tool_calls": 0,
        "usage": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 0,
        },
    }

    async def collect(event: Any) -> None:
        if event.type == "message_end" and isinstance(event.message, AgentAssistantMessage):
            text = "\n".join(
                block.text
                for block in event.message.content
                if isinstance(block, TextContent) and block.text
            )
            if text:
                collected["texts"].append(text)
            if event.message.error_message:
                collected["errors"].append(event.message.error_message)
        elif event.type == "tool_execution_start":
            collected["tool_calls"] += 1
        elif event.type == "turn_end":
            usage = event.message.usage
            collected["usage"]["input"] += usage.input
            collected["usage"]["output"] += usage.output
            collected["usage"]["cache_read"] += usage.cache_read
            collected["usage"]["cache_write"] += usage.cache_write
            collected["usage"]["total_tokens"] += usage.total_tokens

    child.subscribe(collect)

    try:
        await child.prompt(prompt)
    except asyncio.CancelledError:
        child.abort()
        raise
    except Exception as exc:
        return _error(call, f"sub-agent failed: {exc}")

    details = {
        "depth": depth + 1,
        "tool_calls": collected["tool_calls"],
        "usage": collected["usage"],
        "assistant_messages": len(collected["texts"]),
    }

    if collected["errors"]:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[TextContent(text=truncate_output("\n".join(collected["errors"])))],
            details=details,
            is_error=True,
        )
    if collected["texts"]:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[TextContent(text=truncate_output(collected["texts"][-1]))],
            details=details,
        )
    return AgentToolResult(
        tool_call_id=call.id,
        tool_name=call.name,
        content=[
            TextContent(
                text=f"Sub-agent finished without a final answer after {max_turns} turn(s)."
            )
        ],
        details=details,
    )


def create_subagent_tool() -> AgentTool:
    """创建 subagent 工具定义。"""
    return AgentTool(
        name="subagent",
        description=(
            "Spawn a sub-agent to complete an independent task. Use when a task can run "
            "separately from the main conversation (e.g. exploring a subsystem, drafting a "
            "report, or running a bounded investigation) and you want to keep the main thread "
            "focused. The sub-agent runs its own loop with restricted tools and returns a "
            "concise result. Prefer solving small tasks inline; use sub-agents for parallel "
            "or isolated work."
        ),
        parameters=PARAMETERS,
        execute=execute,
    )
