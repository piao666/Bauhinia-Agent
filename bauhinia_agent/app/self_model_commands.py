"""User controls and read-only snapshot rendering for the runtime Self Model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bauhinia_agent.app.commands import CommandResult


class SelfModelRuntimeLike(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def runtime_scope(self) -> str: ...

    def set_enabled(self, enabled: bool) -> None: ...

    def render_user_snapshot(self) -> str: ...


@dataclass(slots=True)
class SelfModelCommandHandler:
    """Handle the project-process switch without becoming a profile truth source."""

    runtime: SelfModelRuntimeLike

    def handle(self, text: str) -> CommandResult:
        command = " ".join(text.strip().split())
        if command == "/self-model":
            return CommandResult(handled=True, output=self.runtime.render_user_snapshot())
        if not command.startswith("/self-model "):
            return CommandResult(handled=False)

        action = command.split(" ", 1)[1].casefold()
        if action in {"on", "enable", "enabled"}:
            self.runtime.set_enabled(True)
            return CommandResult(
                handled=True,
                output=f"Self Model enabled for {self.runtime.runtime_scope}",
            )
        if action in {"off", "disable", "disabled"}:
            self.runtime.set_enabled(False)
            return CommandResult(
                handled=True,
                output=f"Self Model disabled for {self.runtime.runtime_scope}",
            )
        return CommandResult(
            handled=True,
            output="Usage: /self-model [on|off]",
        )
