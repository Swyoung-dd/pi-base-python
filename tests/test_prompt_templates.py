"""Markdown 提示模板发现与展开测试。"""

import pytest

from pi.coding_agent.prompt_templates import load_prompt_template, load_prompt_templates


def _write_template(path, name, body, description=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_prompt_templates_use_path_precedence_and_expand_arguments(tmp_path):
    project = tmp_path / "project" / "review.md"
    user = tmp_path / "user" / "review.md"
    _write_template(project, "review", "Review $ARGUMENTS", "Project review")
    _write_template(user, "review", "User review $ARGUMENTS")

    templates = load_prompt_templates([project.parent, user.parent])

    assert len(templates) == 1
    assert templates[0].description == "Project review"
    assert templates[0].render("src/pi") == "Review src/pi"


def test_prompt_template_appends_arguments_without_placeholder(tmp_path):
    path = tmp_path / "explain.md"
    _write_template(path, "explain", "Explain this code")

    template = load_prompt_template(path)

    assert template.render("src/main.py") == "Explain this code\n\nsrc/main.py"


def test_invalid_prompt_template_is_rejected(tmp_path):
    path = tmp_path / "invalid.md"
    _write_template(path, "Invalid Name", "body")

    with pytest.raises(ValueError, match="Invalid prompt template name"):
        load_prompt_template(path)


def test_templates_are_discovered_inside_dot_piy_root(tmp_path):
    root = tmp_path / ".piy" / "prompts"
    _write_template(root / "commit.md", "commit", "Create a commit")

    templates = load_prompt_templates([root])

    assert [template.name for template in templates] == ["commit"]
