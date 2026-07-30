"""Markdown 提示模板的发现、校验和参数展开。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class PromptTemplate:
    """可通过斜杠命令调用的提示模板。"""

    name: str
    description: str
    content: str
    file_path: Path

    def render(self, arguments: str = "") -> str:
        """展开模板参数；无占位符时把参数追加到正文。"""
        rendered = self.content.replace("${ARGUMENTS}", arguments).replace(
            "$ARGUMENTS",
            arguments,
        )
        if arguments and rendered == self.content:
            rendered = f"{rendered.rstrip()}\n\n{arguments}"
        return rendered.strip()


def load_prompt_template(path: Path) -> PromptTemplate:
    """读取单个 Markdown 模板。"""
    content = path.read_text(encoding="utf-8")
    metadata: dict = {}
    body = content
    if content.startswith("---\n"):
        end = content.find("\n---\n", 4)
        if end < 0:
            raise ValueError(f"Prompt template frontmatter is not closed: {path}")
        value = yaml.safe_load(content[4:end]) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Prompt template frontmatter must be a mapping: {path}")
        metadata = value
        body = content[end + 5 :]
    name = str(metadata.get("name") or path.stem)
    description = str(metadata.get("description") or "").strip()
    if len(name) > 64 or not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid prompt template name: {name}")
    if len(description) > 1024:
        raise ValueError(f"Invalid prompt template description: {path}")
    if not body.strip():
        raise ValueError(f"Prompt template is empty: {path}")
    return PromptTemplate(
        name=name,
        description=description,
        content=body.strip(),
        file_path=path.resolve(),
    )


def _discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".md" else []
    if not path.is_dir():
        return []
    candidates = []
    for candidate in path.rglob("*.md"):
        relative_parts = candidate.relative_to(path).parts
        if any(part.startswith(".") or part == "node_modules" for part in relative_parts):
            continue
        candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: str(candidate))


def load_prompt_templates(paths: list[Path]) -> list[PromptTemplate]:
    """按路径优先级加载模板；同名时保留最先出现的定义。"""
    loaded: dict[str, PromptTemplate] = {}
    seen_files: set[Path] = set()
    for path in paths:
        for candidate in _discover(path.resolve()):
            canonical = candidate.resolve()
            if canonical in seen_files:
                continue
            seen_files.add(canonical)
            template = load_prompt_template(canonical)
            loaded.setdefault(template.name, template)
    return list(loaded.values())
