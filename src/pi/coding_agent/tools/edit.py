"""文件编辑工具 — 基于搜索替换的编辑。"""

from __future__ import annotations

from pathlib import Path

from pi.agent.tools.base import ToolContext
from pi.agent.types import AgentTool, AgentToolCall, AgentToolResult
from pi.ai.types import TextContent

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to edit. Relative to cwd.",
        },
        "old_text": {
            "type": "string",
            "description": "The exact text to find in the file.",
        },
        "new_text": {
            "type": "string",
            "description": "The replacement text.",
        },
    },
    "required": ["path", "old_text", "new_text"],
}


async def execute(call: AgentToolCall, ctx: ToolContext | None) -> AgentToolResult:
    path_arg = call.arguments.get("path", "")
    old_text = call.arguments.get("old_text", "")
    new_text = call.arguments.get("new_text", "")

    cwd = ctx.cwd if ctx else Path.cwd()
    file_path = (cwd / path_arg).resolve()

    try:
        if not file_path.exists():
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="edit",
                content=[TextContent(text=f"File not found: {path_arg}")],
                is_error=True,
            )

        content = file_path.read_text(encoding="utf-8", errors="replace")

        count = content.count(old_text)
        if count == 0:
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="edit",
                content=[TextContent(text=f"old_text not found in {path_arg}")],
                is_error=True,
            )
        if count > 1:
            return AgentToolResult(
                tool_call_id=call.id,
                tool_name="edit",
                content=[TextContent(
                    text=(
                        f"old_text found {count} times in {path_arg}."
                        " Provide more context to make it unique."
                    )
                )],
                is_error=True,
            )

        new_content = content.replace(old_text, new_text, 1)
        file_path.write_text(new_content, encoding="utf-8")

        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="edit",
            content=[TextContent(text=f"Edited {path_arg}: 1 replacement")],
        )
    except Exception as exc:
        return AgentToolResult(
            tool_call_id=call.id,
            tool_name="edit",
            content=[TextContent(text=f"Error editing file: {exc}")],
            is_error=True,
        )


def create_edit_tool() -> AgentTool:
    return AgentTool(
        name="edit",
        description=(
            "Edit a file by replacing old_text with new_text."
            " The old_text must be unique in the file."
        ),
        parameters=PARAMETERS,
        execute=execute,
    )
