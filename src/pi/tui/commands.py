"""交互式 TUI 的斜杠命令注册、校验与分发。"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from pi.agent.session.jsonl import JsonlStorage
from pi.ai.models import list_models
from pi.ai.types import Model, ModelThinkingLevel
from pi.coding_agent.file_references import expand_file_references
from pi.coding_agent.model_auth import ensure_model_auth
from pi.coding_agent.sessions import (
    format_session_tree,
    list_sessions,
    new_session_id,
    resolve_entry_id,
    session_path,
)

if TYPE_CHECKING:
    from pi.tui.interactive import InteractiveSession


BUILTIN_COMMANDS = frozenset(
    {
        "branch",
        "clear",
        "compact",
        "exit",
        "follow-up",
        "help",
        "model",
        "new",
        "resume",
        "sessions",
        "skill",
        "steer",
        "templates",
        "thinking",
        "tree",
    }
)


def validate_command_names(
    extension_commands: set[str],
    prompt_templates: set[str],
) -> None:
    """拒绝会遮蔽内置命令或扩展命令的动态名称。"""
    conflicts = prompt_templates & (BUILTIN_COMMANDS | extension_commands)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"Prompt template command conflicts: {names}")
    extension_conflicts = extension_commands & BUILTIN_COMMANDS
    if extension_conflicts:
        names = ", ".join(sorted(extension_conflicts))
        raise ValueError(f"Extension command conflicts: {names}")


def build_command_names(
    extension_commands: set[str],
    skill_names: set[str],
    prompt_templates: set[str],
) -> list[str]:
    """生成 prompt-toolkit 使用的完整命令补全集。"""
    commands = [f"/{name}" for name in BUILTIN_COMMANDS]
    commands.extend(f"/{name}" for name in extension_commands)
    commands.extend(f"/skill:{name}" for name in skill_names)
    commands.extend(f"/{name}" for name in prompt_templates)
    return sorted(commands)


class CommandDispatcher:
    """将输入命令路由到独立的领域处理方法。"""

    def __init__(self, session: InteractiveSession, select_option_fn: Any) -> None:
        self._session = session
        self._select_option = select_option_fn

    async def handle(self, prompt: str) -> tuple[bool, bool]:
        """处理命令，返回（是否已处理，是否退出）。"""
        parts = prompt.strip().split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command in ("exit", "quit", "/q", "/quit", "/exit"):
            return True, True
        if command == "/clear":
            self._session._console.clear()
            return True, False

        handlers = {
            "/help": self._help,
            "/thinking": self._thinking,
            "/compact": self._compact,
            "/templates": self._templates,
            "/model": self._model,
            "/tree": self._tree,
            "/branch": self._branch,
            "/sessions": self._sessions,
        }
        handler = handlers.get(command)
        if handler is not None:
            await handler(argument)
            return True, False
        if command == "/skill" or command.startswith("/skill:"):
            await self._skill(command, argument)
            return True, False
        if command in ("/steer", "/follow-up"):
            self._queue_message(command, argument)
            return True, False
        if command in ("/new", "/resume"):
            await self._switch_session(command, argument)
            return True, False
        if await self._dynamic_command(command, argument):
            return True, False
        if prompt.startswith("/"):
            self._session._console.print(
                f"Unknown command: {command}",
                style=self._session._theme.warning,
            )
            return True, False
        return False, False

    async def _help(self, _argument: str) -> None:
        self._session._console.print(
            "/new  /resume <id>  /sessions  /model [provider/id]  "
            "/tree  /branch <entry-id>  "
            "/compact [target_tokens]  "
            "/thinking [level]  "
            "/templates  /<template> [arguments]  "
            "/skill <name> [task]  /steer <message>  /follow-up <message>  "
            "/clear  /help  /exit",
            style=self._session._theme.muted,
        )

    async def _thinking(self, argument: str) -> None:
        model = self._session._agent.state.model
        if model is None or not model.reasoning:
            self._session._console.print(
                "Current model does not support thinking.",
                style=self._session._theme.warning,
            )
            return
        if not argument:
            level = await self._select_option(
                "Thinking level",
                [(candidate, candidate.value) for candidate in ModelThinkingLevel],
                default=self._session._agent.thinking_level,
            )
            if level is None:
                self._session._console.print(
                    "Thinking selection cancelled.",
                    style=self._session._theme.muted,
                )
                return
        else:
            try:
                level = ModelThinkingLevel(argument.lower())
            except ValueError:
                self._session._console.print(
                    f"Invalid thinking level: {argument}",
                    style=self._session._theme.error,
                )
                return
        try:
            self._session._agent.set_thinking_level(level)
        except RuntimeError as exc:
            self._session._console.print(str(exc), style=self._session._theme.warning)
            return
        if self._session._on_thinking_selected is not None:
            self._session._on_thinking_selected(level)
        self._session._console.print(
            f"Thinking: {level.value}",
            style=self._session._theme.muted,
        )

    async def _compact(self, argument: str) -> None:
        if self._session._agent.is_busy:
            self._session._console.print(
                "Agent is busy. Wait for idle.",
                style=self._session._theme.warning,
            )
            return
        target: int | None = None
        if argument:
            try:
                target = int(argument)
            except ValueError:
                self._session._console.print(
                    f"Invalid token count: {argument}",
                    style=self._session._theme.error,
                )
                return
        try:
            result = await self._session._agent.compact(target)
        except RuntimeError as exc:
            self._session._console.print(str(exc), style=self._session._theme.error)
            return
        if result.dropped_messages > 0:
            self._session._console.print(
                f"Compacted: {result.original_tokens} -> {result.compacted_tokens} tokens "
                f"({result.dropped_messages} messages dropped)",
                style=self._session._theme.muted,
            )
        else:
            self._session._console.print(
                f"No compaction needed ({result.original_tokens} tokens)",
                style=self._session._theme.muted,
            )

    async def _templates(self, _argument: str) -> None:
        if not self._session._prompt_templates:
            self._session._console.print(
                "No prompt templates.",
                style=self._session._theme.muted,
            )
            return
        for template in self._session._prompt_templates.values():
            description = f" - {template.description}" if template.description else ""
            self._session._console.print(
                f"/{template.name}{description}",
                style=self._session._theme.muted,
            )

    async def _model(self, argument: str) -> None:
        models = list_models()
        selected: Model | None = None
        if argument:
            matches = [
                model
                for model in models
                if argument in (model.id, f"{model.provider}/{model.id}")
            ]
            if len(matches) != 1:
                self._session._console.print(
                    f"Unknown or ambiguous model: {argument}",
                    style=self._session._theme.error,
                )
                return
            selected = matches[0]
        else:
            current = self._session._agent.state.model
            model_options = []
            for model in models:
                is_current = current and (model.provider, model.id) == (
                    current.provider,
                    current.id,
                )
                suffix = "  current" if is_current else ""
                model_options.append((model, f"{model.provider}/{model.id}{suffix}"))
            selected = await self._select_option(
                "Select model",
                model_options,
                default=current,
            )
            if selected is None:
                self._session._console.print(
                    "Model selection cancelled.",
                    style=self._session._theme.muted,
                )
                return

        if not await ensure_model_auth(selected, self._session._credential_store):
            return
        self._session._agent.set_model(selected)
        if self._session._session_storage is not None:
            await self._session._session_storage.append_model_change(
                selected.provider,
                selected.id,
            )
        if self._session._on_model_selected is not None:
            self._session._on_model_selected(selected)
        self._session._console.print(
            f"Model: {selected.provider}/{selected.id}",
            style=self._session._theme.muted,
        )

    async def _tree(self, _argument: str) -> None:
        if not self._require_session_storage():
            return
        storage = self._session._session_storage
        entries = await storage.get_entries()
        leaf_id = await storage.get_leaf_id()
        self._session._console.print(
            format_session_tree(entries, leaf_id),
            style=self._session._theme.muted,
        )

    async def _branch(self, argument: str) -> None:
        if not self._require_session_storage():
            return
        if not argument:
            self._session._console.print(
                "Usage: /branch <entry-id>",
                style=self._session._theme.warning,
            )
            return
        storage = self._session._session_storage
        try:
            entry_id = resolve_entry_id(await storage.get_entries(), argument)
            await storage.branch_from(entry_id)
        except (KeyError, ValueError) as exc:
            self._session._console.print(str(exc), style=self._session._theme.error)
            return
        await self._session._agent.switch_session(storage, self._session._session_id)
        await self._session._restore_session_model()
        await self._session._emit_extension_event(
            "session_switch",
            {"reason": "branch", "session_id": self._session._session_id},
        )
        self._session._console.print(
            f"Branched from: {entry_id}",
            style=self._session._theme.muted,
        )

    async def _skill(self, command: str, argument: str) -> None:
        if command.startswith("/skill:"):
            skill_name = command.split(":", 1)[1]
            skill_argument = argument
        else:
            skill_parts = argument.split(maxsplit=1)
            skill_name = skill_parts[0] if skill_parts else ""
            skill_argument = skill_parts[1] if len(skill_parts) > 1 else ""
        if not skill_name:
            self._session._console.print(
                "  ".join(sorted(self._session._skills)) or "No skills.",
                style=self._session._theme.muted,
            )
            return
        skill = self._session._skills.get(skill_name)
        if skill is None:
            self._session._console.print(
                f"Skill not found: {skill_name}",
                style=self._session._theme.error,
            )
            return
        prompt = (
            f"Apply the following skill instructions.\n\n{skill.read()}"
            f"\n\nTask:\n{skill_argument or 'Follow the skill instructions.'}"
        )
        self._session._submit_agent_prompt(prompt)

    def _queue_message(self, command: str, argument: str) -> None:
        if not argument:
            self._session._console.print(
                f"Usage: {command} <message>",
                style=self._session._theme.warning,
            )
            return
        self._session._submit_agent_prompt(
            expand_file_references(argument, self._session._cwd),
            follow_up=command == "/follow-up",
        )

    async def _sessions(self, _argument: str) -> None:
        if self._session._sessions_dir is None:
            self._session._console.print(
                "Session storage is disabled.",
                style=self._session._theme.warning,
            )
            return
        sessions = await list_sessions(self._session._sessions_dir)
        if not sessions:
            self._session._console.print("No sessions.", style=self._session._theme.muted)
        for item in sessions:
            self._session._console.print(
                f"{item.session_id}  {item.message_count:>4}  {item.preview}",
                style=self._session._theme.muted,
            )

    async def _switch_session(self, command: str, argument: str) -> None:
        if self._session._sessions_dir is None:
            self._session._console.print(
                "Session storage is disabled.",
                style=self._session._theme.warning,
            )
            return
        session_id = argument if command == "/resume" else new_session_id()
        if not session_id:
            self._session._console.print(
                "Usage: /resume <session-id>",
                style=self._session._theme.warning,
            )
            return
        try:
            path = session_path(self._session._sessions_dir, session_id)
        except ValueError as exc:
            self._session._console.print(str(exc), style=self._session._theme.error)
            return
        if command == "/resume" and not path.exists():
            self._session._console.print(
                f"Session not found: {session_id}",
                style=self._session._theme.error,
            )
            return
        storage = JsonlStorage(path)
        if command == "/new":
            current_model = self._session._agent.state.model
            if current_model is not None:
                await storage.append_model_change(current_model.provider, current_model.id)
        await self._session._agent.switch_session(storage, session_id)
        self._session._session_storage = storage
        self._session._session_id = session_id
        await self._session._restore_session_model()
        await self._session._emit_extension_event(
            "session_switch",
            {"reason": command.lstrip("/"), "session_id": session_id},
        )
        self._session._console.print(
            f"Session: {session_id}",
            style=self._session._theme.muted,
        )

    async def _dynamic_command(self, command: str, argument: str) -> bool:
        extension_command = self._session._commands.get(command.lstrip("/"))
        if extension_command is not None:
            result = extension_command(argument, self._session._agent)
            if inspect.isawaitable(result):
                result = await result
            if result:
                self._session._console.print(result)
            return True
        prompt_template = self._session._prompt_templates.get(command.lstrip("/"))
        if prompt_template is None:
            return False
        self._session._submit_agent_prompt(
            expand_file_references(
                prompt_template.render(argument),
                self._session._cwd,
            )
        )
        return True

    def _require_session_storage(self) -> bool:
        if self._session._session_storage is not None:
            return True
        self._session._console.print(
            "Session storage is disabled.",
            style=self._session._theme.warning,
        )
        return False

