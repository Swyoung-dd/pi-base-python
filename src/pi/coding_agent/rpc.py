"""基于 stdin/stdout JSONL 的 piY RPC 控制协议。"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import TypeAdapter

from pi.agent.types import AgentEvent, AgentMessage, create_user_message
from pi.ai.models import get_model
from pi.coding_agent.output import event_to_dict
from pi.coding_agent.sdk import CodingAgent

_MESSAGES_ADAPTER = TypeAdapter(list[AgentMessage])
RpcSend = Callable[[dict[str, Any]], Awaitable[None]]


class RpcServer:
    """把 JSON 请求映射到一个长期存活的 CodingAgent。"""

    def __init__(self, runtime: CodingAgent, send: RpcSend) -> None:
        self.runtime = runtime
        self._send = send
        self._prompt_task: asyncio.Task[None] | None = None
        self._active_request_id: Any = None
        self._unsubscribe = runtime.agent.subscribe(self._on_event)

    async def start(self) -> None:
        await self.runtime.start()
        await self._send({"type": "ready", "session_id": self.runtime.session_id})

    async def handle(self, request: dict[str, Any]) -> bool:
        """处理单个请求；返回是否继续读取后续请求。"""
        request_id = request.get("id")
        request_type = request.get("type")
        if request_type == "prompt":
            message = request.get("message")
            if not isinstance(message, str) or not message.strip():
                await self._error(request_id, "prompt.message must be a non-empty string")
            elif self._prompt_task is not None and not self._prompt_task.done():
                await self._error(request_id, "agent is already processing")
            else:
                self._active_request_id = request_id
                await self._send({"type": "accepted", "id": request_id})
                self._prompt_task = asyncio.create_task(self._run_prompt(request_id, message))
            return True
        if request_type in {"steer", "follow_up"}:
            message = request.get("message")
            if not isinstance(message, str) or not message.strip():
                await self._error(request_id, f"{request_type}.message must be a string")
            elif not self.runtime.agent.is_busy:
                await self._error(request_id, "agent is not processing")
            else:
                queued = create_user_message(message)
                if request_type == "steer":
                    self.runtime.agent.steer(queued)
                else:
                    self.runtime.agent.follow_up(queued)
                await self._send({"type": "accepted", "id": request_id})
            return True
        if request_type == "abort":
            self.runtime.agent.abort()
            await self._send({"type": "accepted", "id": request_id})
            return True
        if request_type == "get_state":
            await self._send(
                {
                    "type": "response",
                    "id": request_id,
                    "state": self._state_dict(),
                }
            )
            return True
        if request_type == "set_model":
            provider = request.get("provider")
            model_id = request.get("model")
            model = get_model(model_id, provider) if isinstance(model_id, str) else None
            if model is None:
                await self._error(request_id, "unknown or ambiguous model")
            elif self.runtime.agent.is_busy:
                await self._error(request_id, "cannot change model while processing")
            else:
                await self.runtime.set_model(model)
                await self._send({"type": "response", "id": request_id, "ok": True})
            return True
        if request_type == "shutdown":
            if self._prompt_task is not None and not self._prompt_task.done():
                await asyncio.sleep(0)
                self.runtime.agent.abort()
                await self._prompt_task
            await self._send({"type": "response", "id": request_id, "ok": True})
            return False
        await self._error(request_id, f"unknown request type: {request_type}")
        return True

    async def wait(self) -> None:
        if self._prompt_task is not None:
            await self._prompt_task

    async def close(self) -> None:
        await self.wait()
        self._unsubscribe()
        await self.runtime.close()

    async def _run_prompt(self, request_id: Any, message: str) -> None:
        try:
            messages = await self.runtime.prompt(message)
            await self._send(
                {
                    "type": "response",
                    "id": request_id,
                    "messages": _MESSAGES_ADAPTER.dump_python(messages, mode="json"),
                }
            )
        except Exception as exc:
            await self._error(request_id, str(exc))
        finally:
            self._active_request_id = None

    async def _on_event(self, event: AgentEvent) -> None:
        await self._send(
            {
                "type": "event",
                "id": self._active_request_id,
                "event": event_to_dict(event),
            }
        )

    async def _error(self, request_id: Any, message: str) -> None:
        await self._send({"type": "error", "id": request_id, "error": message})

    def _state_dict(self) -> dict[str, Any]:
        model = self.runtime.agent.state.model
        usage = self.runtime.agent.get_context_usage()
        return {
            "busy": self.runtime.agent.is_busy,
            "model": ({"provider": model.provider, "id": model.id} if model is not None else None),
            "session_id": self.runtime.session_id,
            "context": (
                {
                    "tokens": usage.tokens,
                    "context_window": usage.context_window,
                    "percent": usage.percent,
                }
                if usage is not None
                else None
            ),
            "messages": _MESSAGES_ADAPTER.dump_python(
                self.runtime.agent.state.messages,
                mode="json",
            ),
        }


async def serve_stdio(runtime: CodingAgent) -> None:
    """在标准输入输出上运行 JSONL RPC 服务。"""
    write_lock = asyncio.Lock()

    async def send(payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        async with write_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    server = RpcServer(runtime, send)
    await server.start()
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                await send({"type": "error", "id": None, "error": str(exc)})
                continue
            if not await server.handle(request):
                break
    finally:
        await server.close()
