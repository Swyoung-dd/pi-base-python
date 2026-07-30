"""面向模型上下文窗口的消息压缩。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from pi.agent.types import (
    AgentAssistantMessage,
    AgentMessage,
    AgentTool,
    AgentToolResultMessage,
    AgentUserMessage,
)
from pi.ai.types import ImageContent, TextContent, ThinkingContent, ToolCall, Usage


@dataclass
class CompactionResult:
    """一次上下文压缩的结果与统计。"""

    messages: list[AgentMessage]
    original_tokens: int
    compacted_tokens: int
    dropped_messages: int
    usage: Usage | None = None


@dataclass(frozen=True)
class CompactionPlan:
    """等待摘要的旧消息与需要原样保留的最近消息。"""

    dropped: list[AgentMessage]
    kept: list[AgentMessage]
    original_tokens: int
    summary_budget: int


@dataclass(frozen=True)
class ContextUsageEstimate:
    """当前上下文 token 使用量的估算结果。"""

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def _message_text(message: AgentMessage) -> str:
    if isinstance(message, AgentUserMessage):
        if isinstance(message.content, str):
            return message.content
        parts = []
        for block in message.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ImageContent):
                parts.append(f"[图片 {block.mime_type}，{len(block.data)} 个 base64 字符]")
        return "\n".join(parts)
    if isinstance(message, AgentAssistantMessage):
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                parts.append(block.thinking)
            elif isinstance(block, ToolCall):
                arguments = json.dumps(block.arguments, ensure_ascii=False, sort_keys=True)
                parts.append(f"调用工具 {block.name}: {arguments}")
        return "\n".join(parts)
    if isinstance(message, AgentToolResultMessage):
        return "\n".join(block.text for block in message.content if isinstance(block, TextContent))
    return ""


def estimate_message_tokens(message: AgentMessage) -> int:
    """使用字符数近似估算 token，并计入消息结构开销。"""
    return max(1, (len(_message_text(message)) + 3) // 4) + 4


def estimate_messages_tokens(messages: list[AgentMessage]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def calculate_context_tokens(message: AgentAssistantMessage) -> int:
    """从 provider 用量中提取该响应结束时的上下文 token 数。"""
    usage = message.usage
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    """以最近一次有效 provider 用量为基线，估算当前上下文大小。"""
    last_usage_index: int | None = None
    usage_tokens = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, AgentAssistantMessage):
            continue
        if message.stop_reason in ("aborted", "error"):
            continue
        usage_tokens = calculate_context_tokens(message)
        if usage_tokens > 0:
            last_usage_index = index
            break

    trailing_start = last_usage_index + 1 if last_usage_index is not None else 0
    trailing_tokens = estimate_messages_tokens(messages[trailing_start:])
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=last_usage_index,
    )


def estimate_context_tokens_with_overhead(
    messages: list[AgentMessage],
    system_prompt: str,
    tools: list[AgentTool],
) -> int:
    """估算完整请求；已有 provider usage 时不重复计算固定开销。"""
    estimate = estimate_context_tokens(messages)
    if estimate.usage_tokens > 0:
        return estimate.tokens
    tool_text = "".join(
        f"{tool.name}\n{tool.description}\n"
        f"{json.dumps(tool.parameters, ensure_ascii=False, sort_keys=True)}\n"
        for tool in tools
    )
    overhead = max(0, (len(system_prompt) + len(tool_text) + 3) // 4)
    return estimate.tokens + overhead


def _group_turns(messages: list[AgentMessage]) -> list[list[AgentMessage]]:
    groups: list[list[AgentMessage]] = []
    for message in messages:
        if isinstance(message, AgentUserMessage) or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def format_messages_for_summary(messages: list[AgentMessage]) -> str:
    """将待压缩消息序列化为模型可总结的带角色文本。"""
    labels = {
        "user": "用户",
        "assistant": "助手",
        "toolResult": "工具",
    }
    return "\n\n".join(f"{labels[message.role]}:\n{_message_text(message)}" for message in messages)


def _summary_message(summary: str, token_budget: int, timestamp: int) -> AgentUserMessage:
    prefix = "[已压缩的早期对话]\n"
    max_chars = max(0, token_budget * 4 - len(prefix))
    text = summary.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "..." if max_chars else ""
    return AgentUserMessage(content=prefix + text, timestamp=timestamp)


def prepare_compaction(
    messages: list[AgentMessage],
    target_tokens: int,
    original_tokens: int | None = None,
) -> CompactionPlan:
    """选择要总结的旧轮次，并原样保留最近完整轮次。"""
    measured_tokens = original_tokens or estimate_messages_tokens(messages)
    if measured_tokens <= target_tokens or len(messages) <= 1:
        return CompactionPlan([], messages[:], measured_tokens, 0)

    groups = _group_turns(messages)
    kept_groups: list[list[AgentMessage]] = []
    kept_tokens = 0
    recent_budget = max(1, target_tokens * 3 // 4)
    for group in reversed(groups):
        group_tokens = estimate_messages_tokens(group)
        if kept_groups and kept_tokens + group_tokens > recent_budget:
            break
        kept_groups.append(group)
        kept_tokens += group_tokens
    kept_groups.reverse()

    kept_count = sum(len(group) for group in kept_groups)
    dropped = messages[:-kept_count] if kept_count else messages[:]
    kept = [message for group in kept_groups for message in group]
    return CompactionPlan(
        dropped=dropped,
        kept=kept,
        original_tokens=measured_tokens,
        summary_budget=max(8, target_tokens - kept_tokens),
    )


def apply_compaction_summary(
    plan: CompactionPlan,
    summary: str,
    usage: Usage | None = None,
) -> CompactionResult:
    """将模型摘要与最近消息组合成新的上下文检查点。"""
    if not plan.dropped:
        return CompactionResult(
            messages=plan.kept[:],
            original_tokens=plan.original_tokens,
            compacted_tokens=estimate_messages_tokens(plan.kept),
            dropped_messages=0,
            usage=usage,
        )
    timestamp = plan.dropped[-1].timestamp
    kept = deepcopy(plan.kept)
    for message in kept:
        if isinstance(message, AgentAssistantMessage):
            message.usage = Usage()
    compacted = [
        _summary_message(summary, plan.summary_budget, timestamp),
        *kept,
    ]
    return CompactionResult(
        messages=compacted,
        original_tokens=plan.original_tokens,
        compacted_tokens=estimate_messages_tokens(compacted),
        dropped_messages=len(plan.dropped),
        usage=usage,
    )


def compact_messages(
    messages: list[AgentMessage],
    target_tokens: int,
    original_tokens: int | None = None,
) -> CompactionResult:
    """使用确定性摘要压缩消息，作为模型摘要失败时的回退。"""
    plan = prepare_compaction(messages, target_tokens, original_tokens)
    return apply_compaction_summary(
        plan,
        format_messages_for_summary(plan.dropped),
    )
