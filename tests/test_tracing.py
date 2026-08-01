"""结构化 tracing 测试。"""

from __future__ import annotations

import json

from pi.coding_agent.tracing import (
    TraceConfig,
    Tracer,
    get_tracer,
    reset_tracer,
    set_tracer,
)


def test_tracer_disabled_by_default():
    tracer = Tracer()
    tracer.trace("test", {"key": "value"})
    assert tracer.events == []


def test_tracer_records_events_when_enabled():
    tracer = Tracer(TraceConfig(enabled=True))
    tracer.trace("test_event", {"key": "value"})
    assert len(tracer.events) == 1
    assert tracer.events[0].event_type == "test_event"
    assert tracer.events[0].data == {"key": "value"}


def test_tracer_redacts_sensitive_fields():
    tracer = Tracer(TraceConfig(enabled=True))
    tracer.trace("test", {
        "api_key": "sk-secret",
        "token": "abc123",
        "normal_field": "ok",
        "nested": {"password": "hidden", "data": "visible"},
    })
    event = tracer.events[0]
    assert event.data["api_key"] == "[REDACTED]"
    assert event.data["token"] == "[REDACTED]"
    assert event.data["normal_field"] == "ok"
    assert event.data["nested"]["password"] == "[REDACTED]"
    assert event.data["nested"]["data"] == "visible"


def test_tracer_does_not_record_tool_args_by_default():
    tracer = Tracer(TraceConfig(enabled=True))
    tracer.trace_tool_call(
        tool_name="bash",
        tool_call_id="call-1",
        arguments={"command": "rm -rf /"},
    )
    event = tracer.events[0]
    assert "arguments" not in event.data
    assert event.data["tool"] == "bash"


def test_tracer_records_tool_args_when_configured():
    tracer = Tracer(TraceConfig(enabled=True, record_tool_args=True))
    tracer.trace_tool_call(
        tool_name="write",
        tool_call_id="call-1",
        arguments={"path": "test.txt", "content": "hello", "api_key": "secret"},
    )
    event = tracer.events[0]
    assert event.data["arguments"]["path"] == "test.txt"
    assert event.data["arguments"]["api_key"] == "[REDACTED]"


def test_tracer_llm_request_response():
    tracer = Tracer(TraceConfig(enabled=True))
    tracer.trace_llm_request("openai", "gpt-4", message_count=5, tool_count=3)
    tracer.trace_llm_response(
        "openai", "gpt-4", "stop",
        usage={"input": 100, "output": 50, "total_tokens": 150},
    )
    assert len(tracer.events) == 2
    assert tracer.events[0].event_type == "llm_request"
    assert tracer.events[1].event_type == "llm_response"


def test_tracer_compaction():
    tracer = Tracer(TraceConfig(enabled=True))
    tracer.trace_compaction(1000, 500, 10)
    event = tracer.events[0]
    assert event.data["original_tokens"] == 1000
    assert event.data["compacted_tokens"] == 500


def test_tracer_export_jsonl():
    tracer = Tracer(TraceConfig(enabled=True))
    tracer.trace("event1", {"a": 1})
    tracer.trace("event2", {"b": 2})
    jsonl = tracer.export_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["type"] == "event1"
    assert parsed["data"]["a"] == 1


def test_tracer_summary():
    tracer = Tracer(TraceConfig(enabled=True))
    tracer.trace("llm_request")
    tracer.trace("llm_response", duration_ms=100)
    tracer.trace("tool_call", duration_ms=50)
    summary = tracer.summary()
    assert summary["total_events"] == 3
    assert summary["event_types"]["llm_request"] == 1
    assert summary["total_duration_ms"] == 150


def test_tracer_file_output(tmp_path):
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(TraceConfig(enabled=True, output_path=path))
    tracer.trace("test", {"key": "value"})
    tracer.close()
    content = path.read_text(encoding="utf-8").strip()
    parsed = json.loads(content)
    assert parsed["type"] == "test"
    assert parsed["data"]["key"] == "value"


def test_global_tracer_singleton():
    reset_tracer()
    t1 = get_tracer()
    t2 = get_tracer()
    assert t1 is t2
    t3 = Tracer(TraceConfig(enabled=True))
    set_tracer(t3)
    assert get_tracer() is t3
    reset_tracer()
