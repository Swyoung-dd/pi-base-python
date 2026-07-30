"""面向模型上下文窗口的消息压缩。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pi.agent.types import (
    AgentAssistantMessage,
    AgentMessage,
    AgentToolResultMessage,
    AgentUserMessage,
)
from pi.ai.types import TextContent, ThinkingContent, ToolCall


@dataclass
class CompactionResult:
    """一次上下文压缩的结果与统计。"""

    messages: list[AgentMessage]
    original_tokens: int
    compacted_tokens: int
    dropped_messages: int


def _message_text(message: AgentMessage) -> str:
    if isinstance(message, AgentUserMessage):
        if isinstance(message.content, str):
            return message.content
        return "\n".join(block.text for block in message.content if isinstance(block, TextContent))
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


def _group_turns(messages: list[AgentMessage]) -> list[list[AgentMessage]]:
    groups: list[list[AgentMessage]] = []
    for message in messages:
        if isinstance(message, AgentUserMessage) or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return groups


def _summarize(messages: list[AgentMessage], token_budget: int) -> AgentUserMessage:
    labels = {
        "user": "用户",
        "assistant": "助手",
        "toolResult": "工具",
    }
    lines = [f"{labels[message.role]}: {_message_text(message)}" for message in messages]
    prefix = "[已压缩的早期对话]\n"
    max_chars = max(0, token_budget * 4 - len(prefix))
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[-max_chars:] if max_chars else ""
        text = "..." + text
    return AgentUserMessage(content=prefix + text, timestamp=messages[-1].timestamp)


def compact_messages(messages: list[AgentMessage], target_tokens: int) -> CompactionResult:
    """将早期轮次压成摘要，并保留最近轮次的结构化消息。"""
    original_tokens = estimate_messages_tokens(messages)
    if original_tokens <= target_tokens or len(messages) <= 1:
        return CompactionResult(messages[:], original_tokens, original_tokens, 0)

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
    summary_budget = max(8, target_tokens - kept_tokens)
    compacted = [_summarize(dropped, summary_budget), *kept] if dropped else kept
    return CompactionResult(
        messages=compacted,
        original_tokens=original_tokens,
        compacted_tokens=estimate_messages_tokens(compacted),
        dropped_messages=len(dropped),
    )
