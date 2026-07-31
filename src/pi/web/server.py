"""Local HTTP server that exposes piY sessions and agent events to the browser."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import queue
import threading
import webbrowser
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import TypeVar
from urllib.parse import parse_qs, quote, unquote, urlparse

from pydantic import TypeAdapter

from pi.agent.session import JsonlStorage
from pi.agent.types import AgentEvent, AgentMessage
from pi.ai.models import get_model, list_models
from pi.ai.types import ModelThinkingLevel
from pi.coding_agent.config import Config, save_config
from pi.coding_agent.file_references import expand_file_references
from pi.coding_agent.output import event_to_dict
from pi.coding_agent.sdk import CodingAgent, create_coding_agent
from pi.coding_agent.sessions import (
    list_sessions,
    new_session_id,
    session_path,
    validate_session_id,
)

_T = TypeVar("_T")
_MESSAGES_ADAPTER = TypeAdapter(list[AgentMessage])
_MAX_REQUEST_BYTES = 1_000_000
_MAX_TEXT_FILE_BYTES = 2_000_000
_SKIPPED_DIRECTORIES = {
    ".git",
    ".idea",
    ".impeccable",
    ".piy",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-run",
    "__pycache__",
    "dist",
    "node_modules",
}


class WebApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def resolve_workspace_path(root: Path, raw_path: str) -> Path:
    """Resolve a browser-supplied relative path without leaving the project root."""
    workspace = root.resolve()
    candidate = (workspace / unquote(raw_path)).resolve()
    if not candidate.is_relative_to(workspace):
        raise WebApiError(HTTPStatus.FORBIDDEN, "Path is outside the project workspace")
    return candidate


def list_workspace_directory(root: Path, raw_path: str = "") -> dict[str, object]:
    directory = resolve_workspace_path(root, raw_path)
    if not directory.is_dir():
        raise WebApiError(HTTPStatus.NOT_FOUND, "Directory not found")
    entries: list[dict[str, object]] = []
    try:
        children = sorted(
            directory.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.casefold()),
        )
    except OSError as exc:
        raise WebApiError(HTTPStatus.FORBIDDEN, f"Cannot read directory: {exc}") from exc
    for child in children:
        if child.is_dir() and child.name in _SKIPPED_DIRECTORIES:
            continue
        try:
            relative = child.relative_to(root.resolve()).as_posix()
            entries.append(
                {
                    "name": child.name,
                    "path": relative,
                    "kind": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        except OSError:
            continue
        if len(entries) >= 400:
            break
    parent = None
    if directory != root.resolve():
        parent = directory.parent.relative_to(root.resolve()).as_posix()
        if parent == ".":
            parent = ""
    relative_directory = directory.relative_to(root.resolve()).as_posix()
    return {
        "path": "" if relative_directory == "." else relative_directory,
        "parent": parent,
        "entries": entries,
    }


async def session_payload(config: Config, session_id: str) -> dict[str, object]:
    storage = JsonlStorage(session_path(config.sessions_dir, validate_session_id(session_id)))
    branch = await storage.get_branch()
    messages = [entry.message for entry in branch if entry.message is not None]
    model_selection = await storage.get_model_selection()
    return {
        "id": session_id,
        "messages": _MESSAGES_ADAPTER.dump_python(messages, mode="json"),
        "model": (
            {"provider": model_selection[0], "id": model_selection[1]}
            if model_selection is not None
            else None
        ),
    }


@dataclass
class PromptStream:
    events: queue.Queue[dict[str, object] | None]
    future: Future[list[AgentMessage]]


class RuntimeHost:
    """Own long-lived CodingAgent instances on one dedicated asyncio loop."""

    def __init__(self, config: Config, cwd: Path, project_trusted: bool) -> None:
        self.config = config
        self.cwd = cwd.resolve()
        self.project_trusted = project_trusted
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="piy-web-agent", daemon=True)
        self._runtimes: dict[str, CodingAgent] = {}
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coroutine: Coroutine[object, object, _T]) -> Future[_T]:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def call(self, coroutine: Coroutine[object, object, _T]) -> _T:
        return self.submit(coroutine).result()

    async def _get_runtime(self, session_id: str) -> CodingAgent:
        validate_session_id(session_id)
        runtime = self._runtimes.get(session_id)
        if runtime is not None:
            return runtime
        runtime = await create_coding_agent(
            config=self.config,
            cwd=self.cwd,
            session_id=session_id,
            persist_session=True,
            project_trusted=self.project_trusted,
        )
        await runtime.start()
        self._runtimes[session_id] = runtime
        return runtime

    def prompt(self, session_id: str, message: str) -> PromptStream:
        events: queue.Queue[dict[str, object] | None] = queue.Queue()

        async def run() -> list[AgentMessage]:
            runtime = await self._get_runtime(session_id)

            async def forward(event: AgentEvent) -> None:
                events.put({"type": "event", "event": event_to_dict(event)})

            unsubscribe = runtime.agent.subscribe(forward)
            try:
                return await runtime.prompt(expand_file_references(message, self.cwd))
            finally:
                unsubscribe()
                events.put(None)

        return PromptStream(events=events, future=self.submit(run()))

    async def set_model(self, provider: str, model_id: str, session_id: str | None) -> None:
        model = get_model(model_id, provider)
        if model is None:
            raise WebApiError(HTTPStatus.BAD_REQUEST, "Unknown model")
        if session_id:
            runtime = await self._get_runtime(validate_session_id(session_id))
            await runtime.set_model(model)
        self.config.model = model.id
        self.config.provider = model.provider
        save_config(self.config)

    async def set_thinking(self, level: str, session_id: str) -> None:
        try:
            thinking = ModelThinkingLevel(level)
        except ValueError as exc:
            raise WebApiError(HTTPStatus.BAD_REQUEST, "Unknown thinking level") from exc
        runtime = await self._get_runtime(validate_session_id(session_id))
        runtime.agent.set_thinking_level(thinking)
        self.config.thinking_level = thinking.value
        save_config(self.config)

    async def abort(self, session_id: str) -> None:
        runtime = self._runtimes.get(validate_session_id(session_id))
        if runtime is not None:
            runtime.agent.abort()

    async def delete_session(self, session_id: str) -> None:
        validated = validate_session_id(session_id)
        runtime = self._runtimes.pop(validated, None)
        if runtime is not None:
            await runtime.close()
        path = session_path(self.config.sessions_dir, validated)
        if path.exists():
            path.unlink()

    async def close(self) -> None:
        for runtime in list(self._runtimes.values()):
            await runtime.close()
        self._runtimes.clear()

    def shutdown(self) -> None:
        try:
            self.call(self.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=3)


class PiYWebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        config: Config,
        cwd: Path,
        project_trusted: bool,
    ) -> None:
        self.config = config
        self.cwd = cwd.resolve()
        super().__init__(address, PiYRequestHandler)
        self.runtime_host = RuntimeHost(config, cwd, project_trusted)

    def server_close(self) -> None:
        self.runtime_host.shutdown()
        super().server_close()


class PiYRequestHandler(BaseHTTPRequestHandler):
    server: PiYWebServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except WebApiError as exc:
            self._json(exc.status, {"error": exc.message})
        except (OSError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except WebApiError as exc:
            self._json(exc.status, {"error": exc.message})
        except (OSError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/sessions/"):
                raise WebApiError(HTTPStatus.NOT_FOUND, "Endpoint not found")
            session_id = parsed.path.removeprefix("/api/sessions/")
            self.server.runtime_host.call(self.server.runtime_host.delete_session(session_id))
            self._json(HTTPStatus.OK, {"ok": True})
        except WebApiError as exc:
            self._json(exc.status, {"error": exc.message})
        except (OSError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "project": str(self.server.cwd)})
            return
        if parsed.path == "/api/sessions":
            sessions = self.server.runtime_host.call(list_sessions(self.server.config.sessions_dir))
            self._json(
                HTTPStatus.OK,
                {
                    "sessions": [
                        {
                            "id": item.session_id,
                            "updatedAt": item.updated_at.isoformat(),
                            "messageCount": item.message_count,
                            "preview": item.preview,
                        }
                        for item in sessions
                    ]
                },
            )
            return
        if parsed.path.startswith("/api/sessions/"):
            session_id = parsed.path.removeprefix("/api/sessions/")
            payload = self.server.runtime_host.call(
                session_payload(self.server.config, session_id)
            )
            self._json(HTTPStatus.OK, payload)
            return
        if parsed.path == "/api/models":
            selected_model = get_model(self.server.config.model, self.server.config.provider)
            self._json(
                HTTPStatus.OK,
                {
                    "selected": {
                        "provider": (
                            selected_model.provider
                            if selected_model is not None
                            else self.server.config.provider
                        ),
                        "id": (
                            selected_model.id
                            if selected_model is not None
                            else self.server.config.model
                        ),
                    },
                    "thinking": self.server.config.thinking_level,
                    "models": [
                        {
                            "id": model.id,
                            "name": model.name,
                            "provider": model.provider,
                            "reasoning": model.reasoning,
                            "contextWindow": model.context_window,
                        }
                        for model in list_models()
                    ],
                },
            )
            return
        if parsed.path == "/api/files":
            raw_path = query.get("path", [""])[0]
            self._json(HTTPStatus.OK, list_workspace_directory(self.server.cwd, raw_path))
            return
        if parsed.path == "/api/file":
            raw_path = query.get("path", [""])[0]
            path = resolve_workspace_path(self.server.cwd, raw_path)
            if not path.is_file():
                raise WebApiError(HTTPStatus.NOT_FOUND, "File not found")
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if mime_type.startswith("image/") or mime_type == "application/pdf":
                self._json(
                    HTTPStatus.OK,
                    {
                        "path": raw_path,
                        "name": path.name,
                        "kind": "image" if mime_type.startswith("image/") else "pdf",
                        "mimeType": mime_type,
                        "rawUrl": f"/api/raw?path={quote(raw_path)}",
                    },
                )
                return
            if path.stat().st_size > _MAX_TEXT_FILE_BYTES:
                raise WebApiError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "File is too large to preview",
                )
            data = path.read_bytes()
            if b"\x00" in data:
                raise WebApiError(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "Binary preview is not supported",
                )
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WebApiError(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "File is not valid UTF-8",
                ) from exc
            self._json(
                HTTPStatus.OK,
                {
                    "path": raw_path,
                    "name": path.name,
                    "kind": "text",
                    "mimeType": mime_type,
                    "content": content,
                },
            )
            return
        if parsed.path == "/api/raw":
            raw_path = query.get("path", [""])[0]
            path = resolve_workspace_path(self.server.cwd, raw_path)
            if not path.is_file():
                raise WebApiError(HTTPStatus.NOT_FOUND, "File not found")
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    self.wfile.write(chunk)
            return
        if parsed.path == "/":
            self._static("index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/styles.css":
            self._static("styles.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._static("app.js", "text/javascript; charset=utf-8")
            return
        raise WebApiError(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _handle_post(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/api/sessions":
            self._json(HTTPStatus.CREATED, {"id": new_session_id()})
            return
        if parsed.path == "/api/chat":
            session_id = payload.get("sessionId")
            message = payload.get("message")
            if (
                not isinstance(session_id, str)
                or not isinstance(message, str)
                or not message.strip()
            ):
                raise WebApiError(
                    HTTPStatus.BAD_REQUEST,
                    "sessionId and a non-empty message are required",
                )
            self._stream_prompt(validate_session_id(session_id), message.strip())
            return
        if parsed.path == "/api/model":
            provider = payload.get("provider")
            model_id = payload.get("model")
            session_id = payload.get("sessionId")
            if not isinstance(provider, str) or not isinstance(model_id, str):
                raise WebApiError(HTTPStatus.BAD_REQUEST, "provider and model are required")
            if session_id is not None and not isinstance(session_id, str):
                raise WebApiError(HTTPStatus.BAD_REQUEST, "sessionId must be a string")
            self.server.runtime_host.call(
                self.server.runtime_host.set_model(provider, model_id, session_id)
            )
            self._json(HTTPStatus.OK, {"ok": True})
            return
        if parsed.path == "/api/thinking":
            level = payload.get("level")
            session_id = payload.get("sessionId")
            if not isinstance(level, str) or not isinstance(session_id, str):
                raise WebApiError(HTTPStatus.BAD_REQUEST, "level and sessionId are required")
            self.server.runtime_host.call(
                self.server.runtime_host.set_thinking(level, session_id)
            )
            self._json(HTTPStatus.OK, {"ok": True})
            return
        if parsed.path == "/api/abort":
            session_id = payload.get("sessionId")
            if not isinstance(session_id, str):
                raise WebApiError(HTTPStatus.BAD_REQUEST, "sessionId is required")
            self.server.runtime_host.call(self.server.runtime_host.abort(session_id))
            self._json(HTTPStatus.OK, {"ok": True})
            return
        raise WebApiError(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _read_json(self) -> dict[str, object]:
        length_text = self.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError as exc:
            raise WebApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from exc
        if length <= 0 or length > _MAX_REQUEST_BYTES:
            raise WebApiError(HTTPStatus.BAD_REQUEST, "Invalid request body size")
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise WebApiError(HTTPStatus.BAD_REQUEST, "Invalid JSON request") from exc
        if not isinstance(value, dict):
            raise WebApiError(HTTPStatus.BAD_REQUEST, "Request body must be an object")
        return value

    def _stream_prompt(self, session_id: str, message: str) -> None:
        stream = self.server.runtime_host.prompt(session_id, message)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                try:
                    item = stream.events.get(timeout=0.1)
                except queue.Empty:
                    if stream.future.done():
                        break
                    continue
                if item is None:
                    break
                self._write_line(item)
            messages = stream.future.result()
            self._write_line(
                {
                    "type": "complete",
                    "messages": _MESSAGES_ADAPTER.dump_python(messages, mode="json"),
                }
            )
        except (BrokenPipeError, ConnectionResetError):
            self.server.runtime_host.call(self.server.runtime_host.abort(session_id))
        except Exception as exc:
            with suppress(BrokenPipeError, ConnectionResetError):
                self._write_line({"type": "error", "error": str(exc)})

    def _write_line(self, payload: dict[str, object]) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode())
        self.wfile.flush()

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, name: str, content_type: str) -> None:
        resource = files("pi.web.static").joinpath(name)
        data = resource.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def create_web_server(
    config: Config,
    cwd: Path,
    host: str = "127.0.0.1",
    port: int = 43127,
    project_trusted: bool = False,
) -> PiYWebServer:
    return PiYWebServer((host, port), config, cwd, project_trusted)


def serve_web(
    config: Config,
    cwd: Path,
    host: str = "127.0.0.1",
    port: int = 43127,
    open_browser: bool = True,
    project_trusted: bool = False,
) -> None:
    server = create_web_server(config, cwd, host, port, project_trusted)
    bound_host, bound_port = server.server_address[:2]
    display_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
    url = f"http://{display_host}:{bound_port}"
    print(f"piY Web: {url}")
    print(f"Project: {cwd.resolve()}")
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
