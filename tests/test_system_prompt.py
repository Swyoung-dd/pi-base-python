"""编码 agent 系统提示词测试。"""

from pi.coding_agent.system_prompt import build_system_prompt


def test_system_prompt_avoids_tools_for_simple_conversation(tmp_path):
    prompt = build_system_prompt(tmp_path, ["read", "ls"])

    assert "greetings, casual conversation, or simple questions" in prompt
    assert "without inspecting the workspace or calling tools" in prompt
    assert "Do not explore the\n  project unless" in prompt
    assert "Respond in the language used by the user" in prompt


def test_system_prompt_includes_platform_and_shell(tmp_path):
    prompt = build_system_prompt(tmp_path, ["read", "ls"])

    assert "Platform:" in prompt
    assert "bash tool runs commands with" in prompt
    assert "PowerShell" in prompt or "bash" in prompt
