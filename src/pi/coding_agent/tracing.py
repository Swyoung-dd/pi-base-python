"""结构化 tracing，默认不记录提示、工具参数和凭据。

提供安全的 tracing 机制：记录 agent 运行时事件的结构化数据，
但默认过滤掉敏感信息（提示文本、工具参数、API 密钥等）。
可通过配置启用详细记录用于调试。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class TraceConfig:
    """Tracing 配置。"""

    enabled: bool = False
    # 是否记录提示文本（默认不记录）
    record_prompts: bool = False
    # 是否记录工具参数（默认不记录）
    record_tool_args: bool = False
    # 是否记录凭据相关字段（默认不记录）
    record_credentials: bool = False
    # 输出文件路径；None 表示仅内存
    output_path: Path | None = None
    # 敏感字段名（部分匹配时脱敏）
    sensitive_keys: set[str] = field(
        default_factory=lambda: {
            "api_key",
            "apikey",
            "token",
            "secret",
            "password",
            "credential",
            "authorization",
            "auth_token",
        }
    )


@dataclass
class TraceEvent:
    """单个 trace 事件。"""

    timestamp: float
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "ts": self.timestamp,
            "type": self.event_type,
        }
        if self.data:
            result["data"] = self.data
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result


class Tracer:
    """结构化 tracer，安全地记录运行时事件。

    默认配置下不记录提示文本、工具参数和凭据。
    """

    def __init__(self, config: TraceConfig | None = None) -> None:
        self._config = config or TraceConfig()
        self._events: list[TraceEvent] = []
        self._file: TextIO | None = None
        if self._config.enabled and self._config.output_path:
            self._config.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(  # noqa: SIM115
                self._config.output_path,
                "a",
                encoding="utf-8",
            )

    @property
    def config(self) -> TraceConfig:
        return self._config

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def _sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """脱敏敏感字段。"""
        if self._config.record_credentials:
            return data
        return self._sanitize_recursive(data)

    def _sanitize_recursive(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                key_lower = str(key).lower()
                if key_lower in self._config.sensitive_keys or any(
                    key_lower.endswith(f"_{s}") or key_lower == s
                    for s in self._config.sensitive_keys
                ):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self._sanitize_recursive(value)
            return result
        if isinstance(obj, list):
            return [self._sanitize_recursive(item) for item in obj]
        return obj

    def trace(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """记录一个 trace 事件。"""
        if not self._config.enabled:
            return
        sanitized_data = self._sanitize(data or {})
        event = TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=sanitized_data,
            duration_ms=duration_ms,
        )
        self._events.append(event)
        if self._file is not None:
            self._file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            self._file.flush()

    def trace_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any] | None = None,
        is_error: bool = False,
        duration_ms: float | None = None,
    ) -> None:
        """记录工具调用，默认不记录参数。"""
        data: dict[str, Any] = {
            "tool": tool_name,
            "call_id": tool_call_id,
            "error": is_error,
        }
        if self._config.record_tool_args and arguments:
            data["arguments"] = self._sanitize(arguments)
        self.trace("tool_call", data, duration_ms=duration_ms)

    def trace_llm_request(
        self,
        provider: str,
        model: str,
        message_count: int,
        tool_count: int,
        duration_ms: float | None = None,
    ) -> None:
        """记录 LLM 请求，不记录提示内容。"""
        self.trace(
            "llm_request",
            {
                "provider": provider,
                "model": model,
                "message_count": message_count,
                "tool_count": tool_count,
            },
            duration_ms=duration_ms,
        )

    def trace_llm_response(
        self,
        provider: str,
        model: str,
        stop_reason: str,
        usage: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """记录 LLM 响应。"""
        data: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "stop_reason": stop_reason,
        }
        if usage:
            data["usage"] = usage
        self.trace("llm_response", data, duration_ms=duration_ms)

    def trace_compaction(
        self,
        original_tokens: int,
        compacted_tokens: int,
        dropped_messages: int,
        duration_ms: float | None = None,
    ) -> None:
        """记录上下文压缩。"""
        self.trace(
            "compaction",
            {
                "original_tokens": original_tokens,
                "compacted_tokens": compacted_tokens,
                "dropped_messages": dropped_messages,
            },
            duration_ms=duration_ms,
        )

    def trace_session_event(self, event_type: str, session_id: str | None = None) -> None:
        """记录会话生命周期事件。"""
        data: dict[str, Any] = {}
        if session_id:
            data["session_id"] = session_id
        self.trace(event_type, data)

    def clear(self) -> None:
        """清空内存中的事件。"""
        self._events.clear()

    def close(self) -> None:
        """关闭文件输出。"""
        if self._file is not None:
            self._file.close()
            self._file = None

    def export_jsonl(self) -> str:
        """导出所有事件为 JSONL 字符串。"""
        lines = [json.dumps(event.to_dict(), ensure_ascii=False) for event in self._events]
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """返回 trace 统计摘要。"""
        type_counts: dict[str, int] = {}
        total_duration = 0.0
        for event in self._events:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
            if event.duration_ms is not None:
                total_duration += event.duration_ms
        return {
            "total_events": len(self._events),
            "event_types": type_counts,
            "total_duration_ms": total_duration,
        }


_default_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """获取全局默认 tracer。"""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer()
    return _default_tracer


def set_tracer(tracer: Tracer) -> None:
    """设置全局默认 tracer。"""
    global _default_tracer
    _default_tracer = tracer


def reset_tracer() -> None:
    """重置全局 tracer 为默认值。"""
    global _default_tracer
    if _default_tracer is not None:
        _default_tracer.close()
    _default_tracer = None
