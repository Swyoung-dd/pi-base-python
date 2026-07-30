"""显式扩展加载测试。"""

import pytest

from pi.coding_agent.extensions import load_extensions


async def test_local_extension_registers_prompt_and_command(tmp_path):
    extension = tmp_path / "sample_extension.py"
    extension.write_text(
        """async def greet(argument, agent):
    return f"hello {argument}"

def setup(context):
    context.add_system_prompt("extension prompt")
    context.add_command("greet", greet)
""",
        encoding="utf-8",
    )

    context = await load_extensions([extension])
    result = await context.commands["greet"]("pi", None)

    assert context.system_prompt_sections == ["extension prompt"]
    assert result == "hello pi"


async def test_extension_duplicate_command_fails_startup(tmp_path):
    extension = tmp_path / "invalid_extension.py"
    extension.write_text(
        """def command(argument, agent):
    return None

def setup(context):
    context.add_command("same", command)
    context.add_command("same", command)
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate extension command"):
        await load_extensions([extension])
