"""Agent Skills 发现、校验和提示格式化。"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    file_path: Path
    disable_model_invocation: bool = False

    @property
    def base_dir(self) -> Path:
        return self.file_path.parent

    def read(self) -> str:
        return self.file_path.read_text(encoding="utf-8")


def _frontmatter(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        return {}
    end = content.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"Skill frontmatter is not closed: {path}")
    value = yaml.safe_load(content[4:end]) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Skill frontmatter must be a mapping: {path}")
    return value


def load_skill(path: Path) -> Skill:
    """读取并按 Agent Skills 命名约束校验单个技能。"""
    metadata = _frontmatter(path)
    name = str(metadata.get("name") or path.parent.name)
    description = str(metadata.get("description") or "").strip()
    if len(name) > 64 or not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid skill name: {name}")
    if not description or len(description) > 1024:
        raise ValueError(f"Invalid skill description: {path}")
    return Skill(
        name=name,
        description=description,
        file_path=path.resolve(),
        disable_model_invocation=metadata.get("disable-model-invocation") is True,
    )


def _discover_directory(directory: Path) -> list[Skill]:
    if not directory.is_dir():
        return []
    root_skill = directory / "SKILL.md"
    if root_skill.is_file():
        return [load_skill(root_skill)]
    skills: list[Skill] = []
    for child in sorted(directory.iterdir(), key=lambda path: path.name):
        if child.name.startswith(".") or child.name == "node_modules":
            continue
        if child.is_dir():
            skills.extend(_discover_directory(child))
        elif child.suffix.lower() == ".md":
            skills.append(load_skill(child))
    return skills


def load_skills(paths: list[Path]) -> list[Skill]:
    """加载文件或目录，技能名冲突时保留最先出现的定义。"""
    loaded: dict[str, Skill] = {}
    seen_files: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        candidates = _discover_directory(resolved) if resolved.is_dir() else [load_skill(resolved)]
        for skill in candidates:
            canonical = skill.file_path.resolve()
            if canonical in seen_files:
                continue
            seen_files.add(canonical)
            loaded.setdefault(skill.name, skill)
    return list(loaded.values())


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """仅暴露技能元数据；模型匹配任务后再使用 read 工具读取正文。"""
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""
    lines = [
        "The following skills provide specialized instructions for matching tasks.",
        "Use the read tool to load a matching skill file before applying it.",
        "Resolve relative references against the directory containing SKILL.md.",
        "<available_skills>",
    ]
    for skill in visible:
        lines.extend(
            [
                "  <skill>",
                f"    <name>{html.escape(skill.name)}</name>",
                f"    <description>{html.escape(skill.description)}</description>",
                f"    <location>{html.escape(str(skill.file_path))}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)
