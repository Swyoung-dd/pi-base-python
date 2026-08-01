"""可注入的执行环境抽象。

将工具对文件系统和进程的直接依赖抽象为统一接口，
支持本机、容器、远程执行和只读测试实现。
路径策略和审批包装器可在上层叠加。
"""

from __future__ import annotations

import abc
import asyncio
import fnmatch
import os
import re
import shutil
import signal
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi.ai.types import ImageContent


@dataclass
class ExecResult:
    """命令执行结果。"""

    stdout: bytes
    stderr: bytes
    returncode: int
    pid: int = 0


@dataclass
class FileInfo:
    """文件或目录的元数据。"""

    name: str
    path: str
    is_dir: bool
    is_file: bool
    size: int = 0
    modified: float = 0.0


class ExecutionEnv(abc.ABC):
    """工具执行环境的抽象基类。"""

    @abc.abstractmethod
    def resolve(self, path: str | Path) -> Path: ...

    @abc.abstractmethod
    async def read(self, path: str | Path, *, offset: int = 0, limit: int = 0) -> str: ...

    @abc.abstractmethod
    async def write(self, path: str | Path, content: str) -> None: ...

    @abc.abstractmethod
    async def edit(self, path: str | Path, old_text: str, new_text: str) -> int: ...

    @abc.abstractmethod
    async def stat(self, path: str | Path) -> FileInfo: ...

    @abc.abstractmethod
    async def list_dir(self, path: str | Path, *, show_all: bool = False) -> list[FileInfo]: ...

    @abc.abstractmethod
    async def find(
        self, path: str | Path, pattern: str, *, max_results: int = 100
    ) -> list[str]: ...

    @abc.abstractmethod
    async def grep(
        self,
        pattern: str,
        path: str | Path,
        *,
        include: str | None = None,
        max_results: int = 50,
    ) -> list[str]: ...

    @abc.abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: int = 120,
        shell: str = "auto",
    ) -> ExecResult: ...

    @abc.abstractmethod
    def temp_file(self, suffix: str = "") -> Path: ...

    @abc.abstractmethod
    async def read_image(self, path: str | Path) -> ImageContent: ...


class LocalExecutionEnv(ExecutionEnv):
    """本机文件系统和进程执行环境。"""

    def __init__(self, cwd: str | Path | None = None) -> None:
        self._cwd = Path(cwd) if cwd else Path.cwd()

    @property
    def cwd(self) -> Path:
        return self._cwd

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self._cwd / p).resolve()

    async def read(self, path: str | Path, *, offset: int = 0, limit: int = 0) -> str:
        file_path = self.resolve(path)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        if offset > 0 or limit > 0:
            lines = content.split("\n")
            start = offset if offset > 0 else 0
            end = start + limit if limit > 0 else len(lines)
            return "\n".join(lines[start:end])
        return content

    async def write(self, path: str | Path, content: str) -> None:
        file_path = self.resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    async def edit(self, path: str | Path, old_text: str, new_text: str) -> int:
        file_path = self.resolve(path)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_text)
        if count > 0:
            new_content = content.replace(old_text, new_text)
            file_path.write_text(new_content, encoding="utf-8")
        return count

    async def stat(self, path: str | Path) -> FileInfo:
        file_path = self.resolve(path)
        st = file_path.stat()
        return FileInfo(
            name=file_path.name,
            path=str(file_path),
            is_dir=file_path.is_dir(),
            is_file=file_path.is_file(),
            size=st.st_size,
            modified=st.st_mtime,
        )

    async def list_dir(self, path: str | Path, *, show_all: bool = False) -> list[FileInfo]:
        dir_path = self.resolve(path)
        result: list[FileInfo] = []
        for entry in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if not show_all and entry.name.startswith("."):
                continue
            st = entry.stat()
            result.append(
                FileInfo(
                    name=entry.name,
                    path=str(entry),
                    is_dir=entry.is_dir(),
                    is_file=entry.is_file(),
                    size=st.st_size,
                    modified=st.st_mtime,
                )
            )
        return result

    async def find(self, path: str | Path, pattern: str, *, max_results: int = 100) -> list[str]:
        search_dir = self.resolve(path)
        results: list[str] = []
        for entry in sorted(search_dir.rglob("*")):
            if fnmatch.fnmatch(entry.name, pattern):
                rel = entry.relative_to(self._cwd)
                results.append(str(rel))
                if len(results) >= max_results:
                    break
        return results

    async def grep(
        self,
        pattern: str,
        path: str | Path,
        *,
        include: str | None = None,
        max_results: int = 50,
    ) -> list[str]:
        search_path = self.resolve(path)
        regex = re.compile(pattern)
        results: list[str] = []
        if search_path.is_file():
            files = [search_path]
        else:
            files = sorted(search_path.rglob("*"))
            files = [f for f in files if f.is_file()]
            if include:
                files = [f for f in files if fnmatch.fnmatch(f.name, include)]
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line_num, line in enumerate(text.split("\n"), 1):
                if regex.search(line):
                    rel = file_path.relative_to(self._cwd)
                    results.append(f"{rel}:{line_num}: {line.strip()}")
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
        return results

    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: int = 120,
        shell: str = "auto",
    ) -> ExecResult:
        resolved_shell = shell if shell != "auto" else ("powershell" if os.name == "nt" else "bash")
        argv, use_shell = self._build_argv(resolved_shell, command)
        work_dir = str(cwd) if cwd else str(self._cwd)
        process_options: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        proc = subprocess.Popen(
            argv,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=work_dir,
            **process_options,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.to_thread(proc.communicate),
                timeout=timeout,
            )
        except TimeoutError:
            await self._terminate_process_tree(proc)
            raise
        return ExecResult(stdout=stdout, stderr=stderr, returncode=proc.returncode, pid=proc.pid)

    def temp_file(self, suffix: str = "") -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        return Path(path)

    async def read_image(self, path: str | Path) -> ImageContent:
        import base64

        file_path = self.resolve(path)
        data = file_path.read_bytes()
        ext = file_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/png")
        return ImageContent(
            data=base64.b64encode(data).decode("ascii"),
            mime_type=mime_type,
        )

    @staticmethod
    def _build_argv(shell: str, command: str) -> tuple[list[str], bool]:
        if shell == "cmd" and os.name == "nt":
            return [command], True
        if shell == "powershell":
            executable = shutil.which("pwsh") or shutil.which("powershell")
            if executable is None:
                raise RuntimeError("PowerShell not found on PATH")
            return [executable, "-NoProfile", "-NonInteractive", "-Command", command], False
        if shell in ("bash", "sh"):
            executable = shutil.which(shell) or shutil.which("bash") or shutil.which("sh")
            if executable is None:
                raise RuntimeError(f"{shell} not found on PATH")
            return [executable, "-c", command], False
        raise RuntimeError(f"Unsupported shell: {shell}")

    @staticmethod
    async def _terminate_process_tree(proc: subprocess.Popen) -> None:
        if proc.returncode is not None:
            return
        if os.name == "nt":
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.to_thread(
                        subprocess.run,
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=10,
                    ),
                    timeout=15,
                )
            with suppress(ProcessLookupError):
                proc.kill()
        else:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)


class ReadOnlyExecutionEnv(ExecutionEnv):
    """只读执行环境包装器，阻止所有写操作。"""

    def __init__(self, inner: ExecutionEnv) -> None:
        self._inner = inner

    def resolve(self, path: str | Path) -> Path:
        return self._inner.resolve(path)

    async def read(self, path: str | Path, *, offset: int = 0, limit: int = 0) -> str:
        return await self._inner.read(path, offset=offset, limit=limit)

    async def write(self, path: str | Path, content: str) -> None:
        raise PermissionError("ReadOnlyExecutionEnv does not allow write operations")

    async def edit(self, path: str | Path, old_text: str, new_text: str) -> int:
        raise PermissionError("ReadOnlyExecutionEnv does not allow edit operations")

    async def stat(self, path: str | Path) -> FileInfo:
        return await self._inner.stat(path)

    async def list_dir(self, path: str | Path, *, show_all: bool = False) -> list[FileInfo]:
        return await self._inner.list_dir(path, show_all=show_all)

    async def find(self, path: str | Path, pattern: str, *, max_results: int = 100) -> list[str]:
        return await self._inner.find(path, pattern, max_results=max_results)

    async def grep(
        self,
        pattern: str,
        path: str | Path,
        *,
        include: str | None = None,
        max_results: int = 50,
    ) -> list[str]:
        return await self._inner.grep(pattern, path, include=include, max_results=max_results)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: int = 120,
        shell: str = "auto",
    ) -> ExecResult:
        raise PermissionError("ReadOnlyExecutionEnv does not allow exec operations")

    def temp_file(self, suffix: str = "") -> Path:
        raise PermissionError("ReadOnlyExecutionEnv does not allow temp_file operations")

    async def read_image(self, path: str | Path) -> ImageContent:
        return await self._inner.read_image(path)


@dataclass
class WriteQueueExecutionEnv(ExecutionEnv):
    """同文件写入串行队列包装器。"""

    env: ExecutionEnv
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict, init=False, repr=False)

    def _lock_for(self, path: str | Path) -> asyncio.Lock:
        key = str(self.resolve(path))
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def resolve(self, path: str | Path) -> Path:
        return self.env.resolve(path)

    async def read(self, path: str | Path, *, offset: int = 0, limit: int = 0) -> str:
        return await self.env.read(path, offset=offset, limit=limit)

    async def write(self, path: str | Path, content: str) -> None:
        async with self._lock_for(path):
            await self.env.write(path, content)

    async def edit(self, path: str | Path, old_text: str, new_text: str) -> int:
        async with self._lock_for(path):
            return await self.env.edit(path, old_text, new_text)

    async def stat(self, path: str | Path) -> FileInfo:
        return await self.env.stat(path)

    async def list_dir(self, path: str | Path, *, show_all: bool = False) -> list[FileInfo]:
        return await self.env.list_dir(path, show_all=show_all)

    async def find(self, path: str | Path, pattern: str, *, max_results: int = 100) -> list[str]:
        return await self.env.find(path, pattern, max_results=max_results)

    async def grep(
        self,
        pattern: str,
        path: str | Path,
        *,
        include: str | None = None,
        max_results: int = 50,
    ) -> list[str]:
        return await self.env.grep(pattern, path, include=include, max_results=max_results)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: int = 120,
        shell: str = "auto",
    ) -> ExecResult:
        return await self.env.exec(command, cwd=cwd, timeout=timeout, shell=shell)

    def temp_file(self, suffix: str = "") -> Path:
        return self.env.temp_file(suffix)

    async def read_image(self, path: str | Path) -> ImageContent:
        return await self.env.read_image(path)


class ApprovalExecutionEnv(ExecutionEnv):
    """审批包装器：写操作和命令执行需要审批回调。"""

    def __init__(self, env: ExecutionEnv, approval_fn: Any) -> None:
        self._env = env
        self._approval_fn = approval_fn

    def resolve(self, path: str | Path) -> Path:
        return self._env.resolve(path)

    async def read(self, path: str | Path, *, offset: int = 0, limit: int = 0) -> str:
        return await self._env.read(path, offset=offset, limit=limit)

    async def write(self, path: str | Path, content: str) -> None:
        await self._check_approval("write", path=path, size=len(content))
        await self._env.write(path, content)

    async def edit(self, path: str | Path, old_text: str, new_text: str) -> int:
        await self._check_approval("edit", path=path)
        return await self._env.edit(path, old_text, new_text)

    async def stat(self, path: str | Path) -> FileInfo:
        return await self._env.stat(path)

    async def list_dir(self, path: str | Path, *, show_all: bool = False) -> list[FileInfo]:
        return await self._env.list_dir(path, show_all=show_all)

    async def find(self, path: str | Path, pattern: str, *, max_results: int = 100) -> list[str]:
        return await self._env.find(path, pattern, max_results=max_results)

    async def grep(
        self,
        pattern: str,
        path: str | Path,
        *,
        include: str | None = None,
        max_results: int = 50,
    ) -> list[str]:
        return await self._env.grep(pattern, path, include=include, max_results=max_results)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: int = 120,
        shell: str = "auto",
    ) -> ExecResult:
        await self._check_approval("exec", command=command, cwd=cwd)
        return await self._env.exec(command, cwd=cwd, timeout=timeout, shell=shell)

    def temp_file(self, suffix: str = "") -> Path:
        return self._env.temp_file(suffix)

    async def read_image(self, path: str | Path) -> ImageContent:
        return await self._env.read_image(path)

    async def _check_approval(self, action: str, **kwargs: Any) -> None:
        import inspect

        result = self._approval_fn(action, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        if not result:
            raise PermissionError(f"Operation '{action}' was not approved")
