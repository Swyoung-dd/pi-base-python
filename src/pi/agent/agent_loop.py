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

import time
from typing import Any

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
    MessageEndEvent,
    MessageStartEvent,
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


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Convert agent messages to LLM messages for the provider call."""
    result: list[Message] = []
    for msg in messages:
        if isinstance(msg, AgentUserMessage):
            result.append(AiUserMessage(
                content=msg.content if isinstance(msg.content, str) else msg.content,
                timestamp=msg.timestamp,
            ))
        elif isinstance(msg, AgentAssistantMessage):
            result.append(AssistantMessage(
                content=msg.content,
                api=msg.api,
                provider=msg.provider,
                model=msg.model,
                usage=msg.usage,
                stop_reason=StopReason(msg.stop_reason) if msg.stop_reason else StopReason.STOP,
                error_message=msg.error_message,
                timestamp=msg.timestamp,
            ))
        elif isinstance(msg, AgentToolResultMessage):
            result.append(AiToolResultMessage(
                tool_call_id=msg.tool_call_id,
                tool_name=msg.tool_name,
                content=msg.content,
                details=msg.details,
                is_error=msg.is_error,
                timestamp=msg.timestamp,
            ))
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
        llm_messages = convert_to_llm(all_messages)
        llm_context = Context(
            system_prompt=context.system_prompt,
            messages=llm_messages,
            tools=ai_tools,
        )

        stream: EventStream = await stream_fn(model, llm_context, options)

        # 发射 message_start，使用空 partial
        partial_msg = AgentAssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            timestamp=int(time.time() * 1000),
        )
        await sink(MessageStartEvent(message=partial_msg))

        # 流式消费：透传 text_delta / thinking_delta 事件给 UI 层
        final_assistant: AssistantMessage | None = None
        async for event in stream:
            if event.type == "text_delta":
                await sink(TextDeltaUpdateEvent(
                    delta=event.delta,
                    content_index=event.content_index,
                ))
            elif event.type == "thinking_delta":
                await sink(ThinkingDeltaUpdateEvent(
                    delta=event.delta,
                    content_index=event.content_index,
                ))
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

        await sink(MessageStartEvent(message=agent_msg))
        await sink(MessageEndEvent(message=agent_msg))
        all_messages.append(agent_msg)
        new_messages.append(agent_msg)

        if agent_msg.stop_reason in ("error", "aborted"):
            await sink(TurnEndEvent(message=agent_msg, tool_results=[]))
            break

        tool_calls = [
            block for block in agent_msg.content if isinstance(block, ToolCall)
        ]

        if not tool_calls:
            await sink(TurnEndEvent(message=agent_msg, tool_results=[]))
            break

        tool_results: list[AgentToolResult] = []
        for tc in tool_calls:
            await sink(ToolExecutionStartEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
            ))

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
            await sink(ToolExecutionEndEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=result,
            ))

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
