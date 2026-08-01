"""核心 agent 循环。

实现 LLM 与工具执行之间的循环：
1. 通过流函数把上下文发送给 LLM
2. 接收助手消息（可能包含工具调用）
3. 执行工具调用
4. 把工具结果作为新上下文回填
5. 重复执行，直到 LLM 停止且不再请求工具

在流边界处完成 AgentMessage（agent 层）与 Message（LLM 层）之间的转换。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi.agent.compaction import (
    CompactionResult,
    compact_messages,
    estimate_context_tokens_with_overhead,
)
from pi.agent.types import (
    AfterToolCallFn,
    AgentAssistantMessage,
    AgentContext,
    AgentEndEvent,
    AgentEventSink,
    AgentMessage,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    AgentToolResultMessage,
    AgentUserMessage,
    BeforeToolCallFn,
    ContextCompactedEvent,
    ContextCompactionRequest,
    MessageEndEvent,
    MessageStartEvent,
    ProviderRetryEvent,
    TextDeltaUpdateEvent,
    ThinkingDeltaUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionMode,
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

CompactFn = Callable[[list[AgentMessage], int, int], Awaitable[CompactionResult]]


def _validate_tool_arguments(
    arguments: dict[str, Any],
    parameters: dict[str, Any],
) -> str | None:
    """根据工具的 JSON Schema 校验参数，返回错误描述或 None。

    仅检查 required 字段和基本类型，不引入 jsonschema 依赖。
    """
    required = parameters.get("required", [])
    for field_name in required:
        if field_name not in arguments:
            return f"Missing required parameter: {field_name}"

    props = parameters.get("properties", {})
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for field_name, value in arguments.items():
        if field_name not in props:
            continue
        field_schema = props[field_name]
        expected_type = field_schema.get("type")
        if expected_type and expected_type in type_map:
            py_type = type_map[expected_type]
            if expected_type == "integer" and isinstance(value, bool):
                return f"Parameter '{field_name}' must be integer, got boolean"
            if expected_type == "number" and isinstance(value, bool):
                return f"Parameter '{field_name}' must be number, got boolean"
            if not isinstance(value, py_type):
                return (
                    f"Parameter '{field_name}' must be {expected_type},"
                    f" got {type(value).__name__}"
                )
            enum_values = field_schema.get("enum")
        if enum_values and value not in enum_values:
            return f"Parameter '{field_name}' must be one of {enum_values}, got {value!r}"
    return None


@dataclass
class AgentLoopResult:
    """一次 agent loop 产生的消息与最终有效上下文。"""

    messages: list[AgentMessage]
    context_messages: list[AgentMessage]
    compactions: list[CompactionResult] = field(default_factory=list)


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


async def _execute_tool_call(
    tool_call: ToolCall,
    tool: AgentTool | None,
    abort_event: asyncio.Event | None,
    tool_context: Any = None,
) -> AgentToolResult:
    """执行单个工具，并允许取消长时间运行的任务。"""
    if tool is None:
        return AgentToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=[TextContent(text=f"Unknown tool: {tool_call.name}")],
            is_error=True,
        )

    call = AgentToolCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=tool_call.arguments,
    )
    execution_task = asyncio.create_task(tool.execute(call, tool_context))
    try:
        if abort_event is None:
            return await execution_task
        abort_task = asyncio.create_task(abort_event.wait())
        done, pending = await asyncio.wait(
            {execution_task, abort_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if abort_task in done:
            return AgentToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=[TextContent(text="Tool execution aborted")],
                is_error=True,
            )
        return execution_task.result()
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=[TextContent(text=f"Tool error: {exc}")],
            is_error=True,
        )


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """将 agent 消息转换为 provider 调用所需的 LLM 消息。"""
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
    tool_execution: ToolExecutionMode = ToolExecutionMode.PARALLEL,
    tool_context: Any = None,
    get_steering_messages: Callable[[], list[AgentMessage]] | None = None,
   compact_fn: CompactFn | None = None,
   max_turns: int | None = None,
    before_tool_call: BeforeToolCallFn | None = None,
    after_tool_call: AfterToolCallFn | None = None,
) -> AgentLoopResult:
    """使用新提示词运行一次 agent 循环。

    Args:
        prompts: 开启本轮的用户消息。
        context: Agent 上下文（system prompt、消息、工具）。
        model: 要使用的 Model。
        stream_fn: 异步函数 (model, context, options) -> EventStream。
        sink: 生命周期事件的事件接收器。
        options: 流选项（api_key 等）。
        max_turns: 本次循环运行的最大模型调用次数。达到上限后
            会在下一次模型调用之前优雅停止。

    Returns:
        本次运行产生的全部新消息。
    """
    all_messages = list(context.messages)
    all_messages.extend(prompts)
    new_messages: list[AgentMessage] = list(prompts)
    compactions: list[CompactionResult] = []

    tools_map = {t.name: t for t in context.tools}
    ai_tools = [t.to_ai_tool() for t in context.tools] if context.tools else None

    turn_count = 0
    while True:
        if max_turns is not None and turn_count >= max_turns:
            break
        turn_count += 1
        context_messages = all_messages
        if options and options.context_token_limit:
            estimated_tokens = estimate_context_tokens_with_overhead(
                all_messages,
                context.system_prompt,
                context.tools,
            )
            if estimated_tokens > options.context_token_limit:
                target_tokens = options.compact_to_tokens or (options.context_token_limit * 3 // 4)
                compacted = (
                    await compact_fn(all_messages, target_tokens, estimated_tokens)
                    if compact_fn
                    else compact_messages(all_messages, target_tokens, estimated_tokens)
                )
                all_messages = compacted.messages
                context_messages = all_messages
                compactions.append(compacted)
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
            steering = get_steering_messages() if get_steering_messages else []
            if not steering:
                break
            all_messages.extend(steering)
            new_messages.extend(steering)
            continue

        # Truncated response guard: when stop_reason is "length", the model
        # likely produced incomplete tool calls — refuse to execute them.
        if agent_msg.stop_reason == "length":
            tool_results = [
                AgentToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    content=[
                        TextContent(
                            text="Tool calls were not executed because the "
                            "model response was truncated (stop_reason=length). "
                            "Please retry with a shorter response."
                        )
                    ],
                    is_error=True,
                )
                for tc in tool_calls
            ]
            for tc, result in zip(tool_calls, tool_results, strict=True):
                await sink(
                    ToolExecutionStartEvent(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                    )
                )
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
            steering = get_steering_messages() if get_steering_messages else []
            all_messages.extend(steering)
            new_messages.extend(steering)
            continue

        async def execute_and_emit(tool_call: ToolCall) -> AgentToolResult:
            tool = tools_map.get(tool_call.name)
            await sink(
                ToolExecutionStartEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                )
            )
            # Validate arguments against tool's JSON Schema before execution.
            if tool is not None:
                validation_error = _validate_tool_arguments(
                    tool_call.arguments, tool.parameters,
                )
                if validation_error:
                    result = AgentToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text=validation_error)],
                        is_error=True,
                    )
                    await sink(
                        ToolExecutionEndEvent(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=result,
                        )
                    )
                    return result
            # Prepare arguments if the tool has a prepare_arguments hook.
            if tool is not None and tool.prepare_arguments is not None:
                try:
                    prepared = tool.prepare_arguments(dict(tool_call.arguments))
                    tool_call = ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=prepared,
                    )
                except Exception as exc:
                    result = AgentToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text=f"Argument preparation failed: {exc}")],
                        is_error=True,
                    )
                    await sink(
                        ToolExecutionEndEvent(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=result,
                        )
                    )
                    return result
            # before_tool_call hook
            agent_tool_call = AgentToolCall(
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
            if before_tool_call is not None:
                try:
                    await before_tool_call(agent_tool_call, tool)
                except Exception as exc:
                    result = AgentToolResult(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                        content=[TextContent(text=f"before_tool_call hook error: {exc}")],
                        is_error=True,
                    )
                    await sink(
                        ToolExecutionEndEvent(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=result,
                        )
                    )
                    return result
            abort_event = options.abort_event if options else None
            result = await _execute_tool_call(
                tool_call,
                tool,
                abort_event,
                tool_context,
            )
            # after_tool_call hook
            if after_tool_call is not None:
                with contextlib.suppress(Exception):
                    await after_tool_call(agent_tool_call, tool, result)
            await sink(
                ToolExecutionEndEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=result,
                )
            )
            return result

        # Per-tool execution_mode override: if any tool in the batch requests
        # sequential execution, run the entire batch sequentially.
        batch_mode = tool_execution
        for tc in tool_calls:
            tool = tools_map.get(tc.name)
            if tool is not None and tool.execution_mode == ToolExecutionMode.SEQUENTIAL:
                batch_mode = ToolExecutionMode.SEQUENTIAL
                break
        if batch_mode == ToolExecutionMode.PARALLEL:
            tool_results = await asyncio.gather(
                *(execute_and_emit(tool_call) for tool_call in tool_calls)
            )
        else:
            tool_results = []
            for tool_call in tool_calls:
                tool_results.append(await execute_and_emit(tool_call))

        for tc, result in zip(tool_calls, tool_results, strict=True):
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

        compaction_requests = [
            result.details
            for result in tool_results
            if isinstance(result.details, ContextCompactionRequest)
        ]
        if compaction_requests:
            estimated_tokens = estimate_context_tokens_with_overhead(
                all_messages,
                context.system_prompt,
                context.tools,
            )
            requested_target = compaction_requests[-1].target_tokens
            if requested_target is None:
                if options and options.compact_to_tokens:
                    requested_target = options.compact_to_tokens
                elif options and options.context_token_limit:
                    requested_target = options.context_token_limit * 3 // 4
                else:
                    requested_target = max(1, estimated_tokens * 3 // 4)
            compacted = (
                await compact_fn(all_messages, requested_target, estimated_tokens)
                if compact_fn
                else compact_messages(all_messages, requested_target, estimated_tokens)
            )
            all_messages = compacted.messages
            compactions.append(compacted)
            await sink(
                ContextCompactedEvent(
                    original_tokens=compacted.original_tokens,
                    compacted_tokens=compacted.compacted_tokens,
                   dropped_messages=compacted.dropped_messages,
               )
            )

        await sink(TurnEndEvent(message=agent_msg, tool_results=tool_results))
        # If all tools in the batch requested termination, stop the loop.
        if tool_results and all(r.terminate for r in tool_results):
            break
        steering = get_steering_messages() if get_steering_messages else []
        all_messages.extend(steering)
        new_messages.extend(steering)

    await sink(AgentEndEvent(messages=new_messages))
    return AgentLoopResult(
        messages=new_messages,
        context_messages=all_messages,
        compactions=compactions,
    )
