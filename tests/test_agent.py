"""Agent 运行时测试。"""

import pytest

from pi.agent.types import (
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    create_user_message,
)
from pi.ai.types import TextContent


@pytest.mark.asyncio
async def test_create_user_message():
    msg = create_user_message("hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.timestamp > 0


@pytest.mark.asyncio
async def test_create_user_message_with_images():
    from pi.ai.types import ImageContent
    img = ImageContent(data="base64data", mime_type="image/png")
    msg = create_user_message("hello", [img])
    assert msg.role == "user"
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2


@pytest.mark.asyncio
async def test_tool_execution():
    async def execute(call: AgentToolCall, ctx) -> AgentToolResult:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=[TextContent(text=f"Result for {call.name}")],
        )

    tool = AgentTool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    call = AgentToolCall(id="call_1", name="test_tool", arguments={})
    result = await tool.execute(call, None)
    assert result.tool_name == "test_tool"
    assert result.content[0].text == "Result for test_tool"
    assert not result.is_error
