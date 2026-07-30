"""Agent Skills 发现与提示格式测试。"""

import pytest

from pi.coding_agent.skills import format_skills_for_prompt, load_skill, load_skills


def _write_skill(path, name, description, disabled=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    disabled_line = "disable-model-invocation: true\n" if disabled else ""
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{disabled_line}---\n\n# Instructions\n",
        encoding="utf-8",
    )


def test_discovery_and_progressive_prompt_metadata(tmp_path):
    visible_path = tmp_path / "visible" / "SKILL.md"
    manual_path = tmp_path / "manual" / "SKILL.md"
    _write_skill(visible_path, "visible-skill", "Visible description")
    _write_skill(manual_path, "manual-skill", "Manual description", disabled=True)

    skills = load_skills([tmp_path])
    prompt = format_skills_for_prompt(skills)

    assert [skill.name for skill in skills] == ["manual-skill", "visible-skill"]
    assert "visible-skill" in prompt
    assert str(visible_path.resolve()) in prompt
    assert "# Instructions" not in prompt
    assert "manual-skill" not in prompt
    assert skills[0].read().endswith("# Instructions\n")


def test_invalid_skill_name_is_rejected(tmp_path):
    path = tmp_path / "SKILL.md"
    _write_skill(path, "Invalid Name", "description")

    with pytest.raises(ValueError, match="Invalid skill name"):
        load_skill(path)
