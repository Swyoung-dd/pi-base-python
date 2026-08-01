"""EventStream tests."""

from __future__ import annotations

import asyncio

from pi.ai.streaming import (
    DoneEvent,
    ErrorEvent,
    EventStream,
    RetryEvent,
    StartEvent,
    TextDeltaEvent,
)
from pi.ai.types import AssistantMessage, StopReason

_MSG = AssistantMessage(
    content=[],
    api="test",
    provider="test",
    model="test",
    stop_reason=StopReason.STOP,
    timestamp=1,
)


async def test_event_stream_basic_iteration():
    stream = EventStream()
    await stream.push(StartEvent())
    await stream.push(TextDeltaEvent(delta="hello"))
    await stream.end(_MSG)
    events = [e async for e in stream]
    assert len(events) == 2
    assert events[0].type == "start"
    assert events[1].type == "text_delta"


async def test_event_stream_result():
    stream = EventStream()
    msg = _MSG
    await stream.end(msg)
    assert stream.result is msg
    assert stream.closed


async def test_event_stream_cancel():
    stream = EventStream()
    await stream.cancel()
    assert stream.closed
    events = [e async for e in stream]
    assert events == []


async def test_event_stream_push_after_close():
    stream = EventStream()
    await stream.end(_MSG)
    await stream.push(TextDeltaEvent(delta="ignored"))
    events = [e async for e in stream]
    assert events == []


async def test_event_stream_retry_event():
    stream = EventStream()
    await stream.push(RetryEvent(attempt=1, max_retries=3, delay_ms=100, error="timeout"))
    await stream.end(_MSG)
    events = [e async for e in stream]
    assert events[0].type == "retry"
    assert events[0].attempt == 1


async def test_event_stream_done_event():
    msg = _MSG
    stream = EventStream()
    await stream.push(DoneEvent(message=msg))
    await stream.end(msg)
    events = [e async for e in stream]
    assert events[0].type == "done"


async def test_event_stream_error_event():
    msg = AssistantMessage(
        content=[],
        api="test",
        provider="test",
        model="test",
        stop_reason=StopReason.ERROR,
        error_message="test error",
        timestamp=1,
    )
    stream = EventStream()
    await stream.push(ErrorEvent(error=msg))
    await stream.end(msg)
    events = [e async for e in stream]
    assert events[0].type == "error"


async def test_event_stream_producer_task_cancel():
    stream = EventStream()
    task = asyncio.create_task(stream.push(TextDeltaEvent(delta="x")))
    stream.set_producer_task(task)
    await asyncio.sleep(0.01)
    await stream.cancel()
    assert stream.closed
