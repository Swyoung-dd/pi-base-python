"""Core agent loop.

Implements the LLM <-> tool execution cycle:
1. Send context to LLM via the stream function
2. Receive assistant message (possibly with tool calls)
3. Execute tool calls
4. Feed tool results back as new context
5. Repeat until the LLM stops with no tool calls

Translates between AgentMessage (agent-level) and Message (LLM-level) at the
stream boundary.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pi.agent.compaction import compact_messages, estimate_messages_tokens
from pi.agent.types import (
    AgentAssistantMessage,
    AgentContext,
    AgentEndEvent,
    AgentEventSink,
    AgentMessage,
    AgentToolCall,
    AgentToolResult,
    AgentToolResultMessage,
    AgentUserMessage,
    ContextCompactedEvent,
    MessageEndEvent,
    MessageStartEvent,
    ProviderRetryEvent,
    TextDeltaUpdateEvent,
    ThinkingDeltaUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)
from pi.ai.streaming import EventStream
from pi.ai.types import (
    AssistantMessage,
    Context,
    Message,
    StopReason,
    StreamOptions,
    TextContent,
    ToolCall,
)
from pi.ai.types import (
    ToolResultMessage as AiToolResultMessage,
)
from pi.ai.types import (
    UserMessage as AiUserMessage,
)


async def _next_stream_event(
    stream: EventStream,
    iterator: Any,
    abort_event: asyncio.Event | None,
) -> tuple[Any | None, bool]:
    """读取下一个流事件，并在收到取消信号时终止生产者任务。"""
    if abort_event is None:
        try:
            return await anext(iterator), False
        except StopAsyncIteration:
            return None, False

    next_task = asyncio.create_task(anext(iterator))
    abort_task = asyncio.create_task(abort_event.wait())
    done, pending = await asyncio.wait(
        {next_task, abort_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if abort_task in done:
        await stream.cancel()
        return None, True
    try:
        return next_task.result(), False
    except StopAsyncIteration:
        return None, False


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Convert agent messages to LLM messages for the provider call."""
    result: list[Message] = []
    for msg in messages:
        if isinstance(msg, AgentUserMessage):
            result.append(
                AiUserMessage(
                    content=msg.content if isinstance(msg.content, str) else msg.content,
                    timestamp=msg.timestamp,
                )
            )
        elif isinstance(msg, AgentAssistantMessage):
            result.append(
                AssistantMessage(
                    content=msg.content,
                    api=msg.api,
                    provider=msg.provider,
                    model=msg.model,
                    usage=msg.usage,
                    stop_reason=StopReason(msg.stop_reason) if msg.stop_reason else StopReason.STOP,
                    error_message=msg.error_message,
                    timestamp=msg.timestamp,
                )
            )
        elif isinstance(msg, AgentToolResultMessage):
            result.append(
                AiToolResultMessage(
                    tool_call_id=msg.tool_call_id,
                    tool_name=msg.tool_name,
                    content=msg.content,
                    details=msg.details,
                    is_error=msg.is_error,
                    timestamp=msg.timestamp,
                )
            )
    return result


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    model: Any,
    stream_fn: Any,
    sink: AgentEventSink,
    options: StreamOptions | None = None,
) -> list[AgentMessage]:
    """Run the agent loop with a new prompt.

    Args:
        prompts: New user messages to start the turn.
        context: Agent context (system prompt, messages, tools).
        model: The Model to use.
        stream_fn: Async function (model, context, options) -> EventStream.
        sink: Event sink for lifecycle events.
        options: Stream options (api_key, etc).

    Returns:
        All new messages produced during this run.
    """
    all_messages = list(context.messages)
    all_messages.extend(prompts)
    new_messages: list[AgentMessage] = list(prompts)

    tools_map = {t.name: t for t in context.tools}
    ai_tools = [t.to_ai_tool() for t in context.tools] if context.tools else None

    while True:
        context_messages = all_messages
        if options and options.context_token_limit:
            estimated_tokens = estimate_messages_tokens(all_messages)
            if estimated_tokens > options.context_token_limit:
                target_tokens = options.compact_to_tokens or (options.context_token_limit * 3 // 4)
                compacted = compact_messages(all_messages, target_tokens)
                context_messages = compacted.messages
                await sink(
                    ContextCompactedEvent(
                        original_tokens=compacted.original_tokens,
                        compacted_tokens=compacted.compacted_tokens,
                        dropped_messages=compacted.dropped_messages,
                    )
                )
        llm_messages = convert_to_llm(context_messages)
        llm_context = Context(
            system_prompt=context.system_prompt,
            messages=llm_messages,
            tools=ai_tools,
        )

        stream: EventStream = await stream_fn(model, llm_context, options)

        # 先发出空消息，后续增量事件会持续更新它。
        partial_msg = AgentAssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            timestamp=int(time.time() * 1000),
        )
        await sink(MessageStartEvent(message=partial_msg))

        final_assistant: AssistantMessage | None = None
        partial_text: list[str] = []
        iterator = stream.__aiter__()
        while True:
            abort_event = options.abort_event if options else None
            event, aborted = await _next_stream_event(stream, iterator, abort_event)
            if aborted:
                final_assistant = AssistantMessage(
                    content=[TextContent(text="".join(partial_text))],
                    api=model.api,
                    provider=model.provider,
                    model=model.id,
                    stop_reason=StopReason.ABORTED,
                    error_message="Request aborted",
                    timestamp=int(time.time() * 1000),
                )
                break
            if event is None:
                break
            if event.type == "text_delta":
                partial_text.append(event.delta)
                await sink(
                    TextDeltaUpdateEvent(
                        delta=event.delta,
                        content_index=event.content_index,
                    )
                )
            elif event.type == "thinking_delta":
                await sink(
                    ThinkingDeltaUpdateEvent(
                        delta=event.delta,
                        content_index=event.content_index,
                    )
                )
            elif event.type == "retry":
                await sink(
                    ProviderRetryEvent(
                        attempt=event.attempt,
                        max_retries=event.max_retries,
                        delay_ms=event.delay_ms,
                        error=event.error,
                    )
                )
            if event.type in ("done", "error"):
                final_assistant = event.message if event.type == "done" else event.error

        if final_assistant is None:
            await sink(MessageEndEvent(message=partial_msg))
            break

        agent_msg = AgentAssistantMessage(
            content=final_assistant.content,
            api=final_assistant.api,
            provider=final_assistant.provider,
            model=final_assistant.model,
            usage=final_assistant.usage,
            stop_reason=final_assistant.stop_reason.value,
            error_message=final_assistant.error_message,
            timestamp=final_assistant.timestamp or int(time.time() * 1000),
        )

        await sink(MessageEndEvent(message=agent_msg))
        all_messages.append(agent_msg)
        new_messages.append(agent_msg)

        if agent_msg.stop_reason in ("error", "aborted"):
            await sink(TurnEndEvent(message=agent_msg, tool_results=[]))
            break

        tool_calls = [block for block in agent_msg.content if isinstance(block, ToolCall)]

        if not tool_calls:
            await sink(TurnEndEvent(message=agent_msg, tool_results=[]))
            break

        tool_results: list[AgentToolResult] = []
        for tc in tool_calls:
            await sink(
                ToolExecutionStartEvent(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                )
            )

            tool = tools_map.get(tc.name)
            if tool is None:
                result = AgentToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=[TextContent(text=f"Unknown tool: {tc.name}")],
                    is_error=True,
                )
            else:
                call = AgentToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                try:
                    result = await tool.execute(call, None)
                except Exception as exc:
                    result = AgentToolResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text=f"Tool error: {exc}")],
                        is_error=True,
                    )

            tool_results.append(result)
            await sink(
                ToolExecutionEndEvent(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=result,
                )
            )

            tr_msg = AgentToolResultMessage(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=result.content,
                details=result.details,
                is_error=result.is_error,
                timestamp=int(time.time() * 1000),
            )
            all_messages.append(tr_msg)
            new_messages.append(tr_msg)

        await sink(TurnEndEvent(message=agent_msg, tool_results=tool_results))

    await sink(AgentEndEvent(messages=new_messages))
    return new_messages
