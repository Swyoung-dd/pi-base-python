"""ExecutionEnv abstract tests."""

from __future__ import annotations

import asyncio
import base64

import pytest

from pi.agent.tools.execution_env import (
    ApprovalExecutionEnv,
    ExecResult,
    FileInfo,
    LocalExecutionEnv,
    ReadOnlyExecutionEnv,
    WriteQueueExecutionEnv,
)


@pytest.mark.asyncio
async def test_local_env_read_write_round_trip(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    await env.write("test.txt", "hello world")
    content = await env.read("test.txt")
    assert content == "hello world"


@pytest.mark.asyncio
async def test_local_env_read_with_offset_limit(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    lines = "line1" + chr(10) + "line2" + chr(10) + "line3" + chr(10) + "line4"
    await env.write("lines.txt", lines)
    content = await env.read("lines.txt", offset=1, limit=2)
    assert "line2" in content
    assert "line3" in content
    assert "line4" not in content


@pytest.mark.asyncio
async def test_local_env_edit(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    await env.write("edit.txt", "foo bar baz")
    count = await env.edit("edit.txt", "bar", "qux")
    assert count == 1
    assert await env.read("edit.txt") == "foo qux baz"


@pytest.mark.asyncio
async def test_local_env_stat(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    await env.write("stat.txt", "data")
    info = await env.stat("stat.txt")
    assert isinstance(info, FileInfo)
    assert info.name == "stat.txt"
    assert info.is_file
    assert not info.is_dir
    assert info.size == 4


@pytest.mark.asyncio
async def test_local_env_list_dir(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "file2.py").write_text("b")
    (tmp_path / "subdir").mkdir()
    entries = await env.list_dir(".")
    names = [e.name for e in entries]
    assert "file1.txt" in names
    assert "file2.py" in names
    assert "subdir" in names


@pytest.mark.asyncio
async def test_local_env_list_dir_hidden(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    (tmp_path / ".hidden").write_text("secret")
    (tmp_path / "visible.txt").write_text("ok")
    without_hidden = await env.list_dir(".", show_all=False)
    names = [e.name for e in without_hidden]
    assert ".hidden" not in names
    assert "visible.txt" in names
    with_hidden = await env.list_dir(".", show_all=True)
    names = [e.name for e in with_hidden]
    assert ".hidden" in names


@pytest.mark.asyncio
async def test_local_env_find(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("z")
    results = await env.find(".", "*.py")
    assert "a.py" in results
    assert "b.txt" not in results


@pytest.mark.asyncio
async def test_local_env_grep(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    nl = chr(10)
    (tmp_path / "search.py").write_text("import os" + nl + "def hello():" + nl + "    pass")
    results = await env.grep("hello", ".")
    assert len(results) == 1
    assert "hello" in results[0]


@pytest.mark.asyncio
async def test_local_env_exec_echo(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    result = await env.exec("echo hello", timeout=5)
    assert isinstance(result, ExecResult)
    assert b"hello" in result.stdout
    assert result.returncode == 0


@pytest.mark.asyncio
async def test_local_env_temp_file(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    path = env.temp_file(suffix=".tmp")
    assert path.exists()
    assert path.suffix == ".tmp"
    path.unlink()


@pytest.mark.asyncio
async def test_local_env_read_image(tmp_path):
    env = LocalExecutionEnv(cwd=tmp_path)
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAH"
        "ggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    (tmp_path / "test.png").write_bytes(base64.b64decode(png_b64))
    img = await env.read_image("test.png")
    assert img.mime_type == "image/png"
    assert len(img.data) > 0


@pytest.mark.asyncio
async def test_readonly_env_blocks_writes(tmp_path):
    env = ReadOnlyExecutionEnv(LocalExecutionEnv(cwd=tmp_path))
    (tmp_path / "existing.txt").write_text("data")
    assert await env.read("existing.txt") == "data"
    with pytest.raises(PermissionError):
        await env.write("new.txt", "content")
    with pytest.raises(PermissionError):
        await env.edit("existing.txt", "data", "modified")


@pytest.mark.asyncio
async def test_readonly_env_blocks_exec(tmp_path):
    env = ReadOnlyExecutionEnv(LocalExecutionEnv(cwd=tmp_path))
    with pytest.raises(PermissionError):
        await env.exec("echo hello")


@pytest.mark.asyncio
async def test_write_queue_serializes_same_file(tmp_path):
    env = WriteQueueExecutionEnv(LocalExecutionEnv(cwd=tmp_path))
    order: list[str] = []

    async def write_slow(content: str, label: str) -> None:
        await env.write("shared.txt", content)
        order.append(label)

    await asyncio.gather(
        write_slow("first", "first"),
        write_slow("second", "second"),
    )
    final = await env.read("shared.txt")
    assert final in ("first", "second")
    assert len(order) == 2


@pytest.mark.asyncio
async def test_approval_env_blocks_unapproved(tmp_path):
    env = ApprovalExecutionEnv(
        LocalExecutionEnv(cwd=tmp_path),
        approval_fn=lambda action, **kw: False,
    )
    with pytest.raises(PermissionError):
        await env.write("test.txt", "data")
    with pytest.raises(PermissionError):
        await env.exec("echo hello")


@pytest.mark.asyncio
async def test_approval_env_allows_approved(tmp_path):
    env = ApprovalExecutionEnv(
        LocalExecutionEnv(cwd=tmp_path),
        approval_fn=lambda action, **kw: True,
    )
    await env.write("test.txt", "approved")
    assert await env.read("test.txt") == "approved"


@pytest.mark.asyncio
async def test_approval_env_async_callback(tmp_path):
    async def approve(action, **kw):
        return True

    env = ApprovalExecutionEnv(LocalExecutionEnv(cwd=tmp_path), approval_fn=approve)
    await env.write("test.txt", "async approved")
    assert await env.read("test.txt") == "async approved"
