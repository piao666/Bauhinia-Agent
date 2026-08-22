"""Best-effort bridge from real AgentLoop activity to Evo facts.

This adapter observes the existing loop; it never executes tools, changes a
permission decision, or changes the response returned by the loop.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bauhinia_agent.agent.tool_execution import ToolExecutionEvent
from bauhinia_agent.evolution.compiler import ExperienceCompiler
from bauhinia_agent.evolution.evidence import (
    EvidenceAdapter,
    EvidenceInput,
    EvidenceRecordResult,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.evolution.store import EvoEventStore

if TYPE_CHECKING:
    from bauhinia_agent.agent.session import AgentSession
    from bauhinia_agent.providers.base import ChatProvider
    from bauhinia_agent.providers.types import ChatResponse


@dataclass(frozen=True, slots=True)
class AgentEvoRunResult:
    """Best-effort Evo result for one completed Agent user turn."""

    run_id: str
    evidence_count: int
    evidence_ids: tuple[str, ...] = ()
    outcome_event_id: str | None = None
    outcome: str | None = None
    outcome_category: str | None = None
    outcome_confidence: float = 0.0
    candidate_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class AgentEvoObserver:
    """Observe a real AgentLoop turn and write only append-only Evo facts."""

    def __init__(
        self,
        *,
        session: AgentSession,
        provider: ChatProvider,
        run_id: str | None = None,
        compile_candidates: bool = True,
    ) -> None:
        self._session = session
        self._provider = provider
        self._store = EvoEventStore(session.store.root)
        self._evidence = EvidenceAdapter(self._store)
        self._outcomes = OutcomeClassifier(self._store)
        self._compiler = ExperienceCompiler(self._store)
        self._fixed_run_id = None if run_id is None else require_evo_id(run_id, field="run_id", kind="run")
        self._fixed_run_consumed = False
        self._compile_candidates = bool(compile_candidates)
        self._active_run_id: str | None = None
        self._evidence_count = 0
        self._evidence_ids: list[str] = []
        self._diagnostics: list[str] = []
        self.last_result: AgentEvoRunResult | None = None

    def begin_turn(self) -> str:
        """Start one independent Evo Run for a new user turn."""

        if self._active_run_id is not None:
            raise RuntimeError("an Evo Run is already active")
        if self._fixed_run_id is not None:
            if self._fixed_run_consumed:
                raise RuntimeError("a fixed Evo Run observer can only observe one turn")
            self._fixed_run_consumed = True
            self._active_run_id = self._fixed_run_id
        else:
            self._active_run_id = new_evo_id("run")
        self._evidence_count = 0
        self._evidence_ids = []
        self._diagnostics = []
        return self._active_run_id

    @property
    def active_run_id(self) -> str | None:
        """Current parent/child Run identity for runtime adapters."""

        return self._active_run_id

    def observe_tool_event(self, event: ToolExecutionEvent) -> None:
        """Record completed tool and permission facts without affecting the loop."""

        run_id = self._active_run_id
        if run_id is None or event.result is None:
            return
        try:
            if event.kind == "denied" and event.permission_request is not None:
                self._record_permission(run_id, event)
                return
            if event.kind not in {"finished", "denied", "interrupted", "skipped"}:
                return
            verification_kind = _verification_kind(event) if self._uses_trusted_verification_executor(event.tool_call.name) else None
            self._record_tool(run_id, event, verification_kind=verification_kind)
            if verification_kind is not None:
                self._record_verification(run_id, event, verification_kind)
        except Exception as error:  # noqa: BLE001 - observer failures cannot change AgentLoop behavior
            self._diagnostics.append(f"evo_observer_failed:{type(error).__name__}:{error}")

    def complete_turn(self, response: ChatResponse) -> AgentEvoRunResult | None:
        """Classify and compile the observed Run after the loop persisted its response."""

        del response  # Classification is evidence-based, never model self-evaluation based.
        run_id = self._active_run_id
        if run_id is None:
            return self.last_result
        outcome_event_id: str | None = None
        outcome_value: str | None = None
        outcome_category: str | None = None
        outcome_confidence = 0.0
        candidate_ids: tuple[str, ...] = ()
        if self._evidence_count:
            try:
                outcome = self._outcomes.classify(run_id)
                if outcome.outcome is not None:
                    outcome_event_id = outcome.outcome.event_id
                    outcome_value = outcome.outcome.payload.outcome
                    outcome_category = outcome.outcome.payload.category
                    outcome_confidence = outcome.outcome.payload.confidence
                if outcome.diagnostic is not None:
                    self._diagnostics.append(f"outcome:{outcome.diagnostic.code}:{outcome.diagnostic.message}")
                if self._compile_candidates:
                    candidate = self._compiler.compile(
                        run_id,
                        environment_summary=self._environment_summary(),
                    )
                    candidate_ids = tuple(item.candidate_id for item in candidate.candidates)
                    if candidate.diagnostic is not None:
                        self._diagnostics.append(f"candidate:{candidate.diagnostic.code}:{candidate.diagnostic.message}")
            except Exception as error:  # noqa: BLE001 - compilation cannot affect the completed turn
                self._diagnostics.append(f"evo_completion_failed:{type(error).__name__}:{error}")
        result = AgentEvoRunResult(
            run_id=run_id,
            evidence_count=self._evidence_count,
            evidence_ids=tuple(self._evidence_ids),
            outcome_event_id=outcome_event_id,
            outcome=outcome_value,
            outcome_category=outcome_category,
            outcome_confidence=outcome_confidence,
            candidate_ids=candidate_ids,
            diagnostics=tuple(self._diagnostics),
        )
        self.last_result = result
        self._active_run_id = None
        return result

    def _record_tool(self, run_id: str, event: ToolExecutionEvent, *, verification_kind: str | None) -> None:
        result = event.result
        if result is None:
            return
        # A verification command has its own evidence record with the command's
        # exit status.  Keep the accompanying generic tool fact outcome-neutral,
        # so a non-zero test exit is classified as verification failure rather
        # than being masked as a generic tool execution failure.
        recorded = self._evidence.record(
            EvidenceInput(
                run_id=run_id,
                evidence_type="tool",
                source=event.tool_call.name,
                summary=result.content or result.error or f"tool {event.tool_call.name} completed",
                locator=event.tool_call.id,
                verified=False,
                input_summary=_arguments_summary(event.tool_call.arguments),
                exit_code=None if verification_kind is not None else (0 if result.ok else 1),
            )
        )
        self._remember_recording_result(recorded)

    def _record_verification(self, run_id: str, event: ToolExecutionEvent, evidence_type: str) -> None:
        result = event.result
        if result is None:
            return
        data = result.data if isinstance(result.data, dict) else {}
        command = _text(data.get("command")) or _text(event.tool_call.arguments.get("command"))
        exit_code = _exit_code(data.get("exit_code"), fallback=0 if result.ok else 1)
        recorded = self._evidence.record(
            EvidenceInput(
                run_id=run_id,
                evidence_type=evidence_type,
                source=f"agent_tool:{event.tool_call.name}",
                summary=result.content or result.error or f"{event.tool_call.name} verification completed",
                verified=True,
                command=command,
                cwd=_text(data.get("cwd")),
                exit_code=exit_code,
            )
        )
        self._remember_recording_result(recorded)

    def _record_permission(self, run_id: str, event: ToolExecutionEvent) -> None:
        request = event.permission_request
        result = event.result
        if request is None or result is None:
            return
        action = getattr(request, "action", event.tool_call.name)
        target = getattr(request, "target", event.tool_call.name)
        recorded = self._evidence.record_permission(
            run_id=run_id,
            action=str(getattr(action, "value", action)),
            target=str(target),
            decision="deny",
            reason=result.content or result.error or "permission denied",
        )
        self._remember_recording_result(recorded)

    def _remember_recording_result(self, result: EvidenceRecordResult) -> None:
        if result.persisted:
            self._evidence_count += 1
            if result.evidence is not None:
                self._evidence_ids.append(result.evidence.evidence_id)
        if result.diagnostic is not None:
            code = getattr(result.diagnostic, "code", "unknown")
            message = getattr(result.diagnostic, "message", str(result.diagnostic))
            self._diagnostics.append(f"evidence:{code}:{message}")

    def _environment_summary(self) -> str:
        return f"runtime=agent_loop; provider={self._provider.name}; model={self._provider.model}"

    def _uses_trusted_verification_executor(self, tool_name: str) -> bool:
        """Accept verification facts only from the built-in shell boundary.

        A provider or plugin can register a tool whose public name is ``shell``.
        Its self-reported command/exit code is not sufficient evidence.  The
        observer therefore checks the concrete executor installed in the
        session registry before interpreting a result as deterministic.
        """

        registry_get = getattr(self._session.tool_registry, "get", None)
        if not callable(registry_get):
            return False
        tool = registry_get(tool_name)
        if tool is None:
            return False
        executor = tool.executor
        return getattr(executor, "__module__", "") == "bauhinia_agent.tools.shell" and getattr(executor, "__qualname__", "").startswith("create_shell_tool.<locals>.shell")


def _verification_kind(event: ToolExecutionEvent) -> str | None:
    result = event.result
    if result is None or event.tool_call.name != "shell" or result.name != "shell":
        return None
    data = result.data if isinstance(result.data, dict) else {}
    command = _text(data.get("command")) or _text(event.tool_call.arguments.get("command")) or ""
    tokens = _direct_command_tokens(command)
    if not tokens:
        return None
    executable = _executable_name(tokens[0])
    arguments = tuple(token.lower() for token in tokens[1:])
    if _is_python_entrypoint(executable):
        if len(arguments) < 2 or arguments[0] != "-m":
            return None
        return _module_verification_kind(arguments[1])
    direct = {
        "pytest": "test",
        "py.test": "test",
        "unittest": "test",
        "ruff": "lint",
        "flake8": "lint",
        "eslint": "lint",
        "pylint": "lint",
        "mypy": "type_check",
        "pyright": "type_check",
        "tsc": "type_check",
    }.get(executable)
    if direct is not None:
        return direct
    if executable in {"npm", "pnpm", "yarn"}:
        script = _package_script(arguments)
        return {
            "test": "test",
            "lint": "lint",
            "typecheck": "type_check",
            "type-check": "type_check",
            "build": "build",
        }.get(script)
    if executable == "cargo" and arguments:
        return {"test": "test", "build": "build"}.get(arguments[0])
    return None


def _direct_command_tokens(command: str) -> tuple[str, ...]:
    if not command.strip() or _has_unquoted_shell_control(command):
        return ()
    try:
        raw = shlex.split(command, posix=False)
    except ValueError:
        return ()
    tokens = tuple(_strip_matching_quotes(token) for token in raw)
    return tokens if all(tokens) else ()


def _has_unquoted_shell_control(command: str) -> bool:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            escaped = False
            continue
        if quote is not None:
            if character == quote:
                quote = None
            elif character == "`" or (quote == '"' and character == "\\"):
                escaped = True
            continue
        if character in {"'", '"'}:
            quote = character
            continue
        if character in "\r\n;&|<>()^`":
            return True
        if character == "$" and command[index : index + 2] == "$(":
            return True
    return quote is not None


def _strip_matching_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _executable_name(token: str) -> str:
    normalized = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".cmd", ".bat"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _is_python_entrypoint(executable: str) -> bool:
    return executable == "py" or re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable) is not None


def _module_verification_kind(module: str) -> str | None:
    return {
        "pytest": "test",
        "unittest": "test",
        "ruff": "lint",
        "flake8": "lint",
        "pylint": "lint",
        "mypy": "type_check",
        "pyright": "type_check",
        "build": "build",
    }.get(module.lower())


def _package_script(arguments: tuple[str, ...]) -> str | None:
    if not arguments:
        return None
    if arguments[0] == "run":
        return arguments[1] if len(arguments) > 1 else None
    return arguments[0]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _exit_code(value: object, *, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _arguments_summary(arguments: object) -> str:
    if not isinstance(arguments, dict):
        return "{}"
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"
