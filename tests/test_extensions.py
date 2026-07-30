"""显式扩展加载测试。"""

import pytest

from pi.coding_agent.extensions import ExtensionContext, load_extensions


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


async def test_entrypoint_extensions_use_piy_group(monkeypatch):
    captured = {}

    def entry_points(*, group):
        captured["group"] = group
        return []

    monkeypatch.setattr("pi.coding_agent.extensions.importlib.metadata.entry_points", entry_points)

    await load_extensions([], enable_entrypoints=True)

    assert captured["group"] == "piy.extensions"


async def test_extension_lifecycle_handlers_run_in_order_and_isolate_failures(tmp_path):
    extension = tmp_path / "lifecycle.py"
    extension.write_text(
        """async def first(event, agent):
    agent.append((event.type, event.data))

def failing(event, agent):
    raise RuntimeError("broken hook")

def last(event, agent):
    agent.append(("last", event.data))

def setup(context):
    context.on("session_start", first)
    context.on("session_start", failing)
    context.on("session_start", last)
""",
        encoding="utf-8",
    )
    context = await load_extensions([extension])
    events = []

    failures = await context.emit("session_start", {"id": "one"}, events)

    assert events == [
        ("session_start", {"id": "one"}),
        ("last", {"id": "one"}),
    ]
    assert len(failures) == 1
    assert failures[0].source == str(extension)
    assert str(failures[0].error) == "broken hook"


def test_extension_rejects_unknown_lifecycle_event():
    context = ExtensionContext()

    with pytest.raises(ValueError, match="Unknown extension event"):
        context.on("unknown", lambda event, agent: None)
