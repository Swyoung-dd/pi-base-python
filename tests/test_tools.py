"""编码工具测试。"""

import tempfile
from pathlib import Path

import pytest

from pi.agent.tools.base import ToolContext
from pi.agent.types import AgentToolCall
from pi.coding_agent.tools.bash import create_bash_tool
from pi.coding_agent.tools.edit import create_edit_tool
from pi.coding_agent.tools.ls import create_ls_tool
from pi.coding_agent.tools.read import create_read_tool
from pi.coding_agent.tools.write import create_write_tool


@pytest.mark.asyncio
async def test_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = ToolContext(cwd=Path(tmpdir))

        write_tool = create_write_tool()
        call = AgentToolCall(
            id="1",
            name="write",
            arguments={"path": "test.txt", "content": "hello world"},
        )
        result = await write_tool.execute(call, ctx)
        assert not result.is_error
        assert (Path(tmpdir) / "test.txt").read_text() == "hello world"

        read_tool = create_read_tool()
        call = AgentToolCall(id="2", name="read", arguments={"path": "test.txt"})
        result = await read_tool.execute(call, ctx)
        assert not result.is_error
        assert "hello world" in result.content[0].text


@pytest.mark.asyncio
async def test_edit():
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = ToolContext(cwd=Path(tmpdir))
        (Path(tmpdir) / "test.txt").write_text("foo bar baz")

        edit_tool = create_edit_tool()
        call = AgentToolCall(
            id="1",
            name="edit",
            arguments={"path": "test.txt", "old_text": "bar", "new_text": "qux"},
        )
        result = await edit_tool.execute(call, ctx)
        assert not result.is_error
        assert (Path(tmpdir) / "test.txt").read_text() == "foo qux baz"


@pytest.mark.asyncio
async def test_edit_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = ToolContext(cwd=Path(tmpdir))
        (Path(tmpdir) / "test.txt").write_text("hello")

        edit_tool = create_edit_tool()
        call = AgentToolCall(
            id="1",
            name="edit",
            arguments={"path": "test.txt", "old_text": "world", "new_text": "pi"},
        )
        result = await edit_tool.execute(call, ctx)
        assert result.is_error


@pytest.mark.asyncio
async def test_ls():
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = ToolContext(cwd=Path(tmpdir))
        (Path(tmpdir) / "file1.txt").write_text("a")
        (Path(tmpdir) / "file2.py").write_text("b")
        Path(tmpdir, "subdir").mkdir()

        ls_tool = create_ls_tool()
        call = AgentToolCall(id="1", name="ls", arguments={})
        result = await ls_tool.execute(call, ctx)
        assert not result.is_error
        text = result.content[0].text
        assert "file1.txt" in text
        assert "file2.py" in text
        assert "subdir/" in text


@pytest.mark.asyncio
async def test_bash():
    bash_tool = create_bash_tool()
    call = AgentToolCall(id="1", name="bash", arguments={"command": "echo hello"})
    result = await bash_tool.execute(call, None)
    assert not result.is_error
    assert "hello" in result.content[0].text
