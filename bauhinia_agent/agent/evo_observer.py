"""Best-effort bridge from real AgentLoop activity to Evo facts.

This adapter observes the existing loop; it never executes tools, changes a
permission decision, or changes the response returned by the loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bauhinia_agent.agent.tool_execution import ToolExecutionEvent
from bauhinia_agent.evolution.compiler import ExperienceCompiler
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.identifiers import new_evo_id
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
    outcome_event_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


class AgentEvoObserver:
    """Observe a real AgentLoop turn and write only append-only Evo facts."""

    def __init__(self, *, session: AgentSession, provider: ChatProvider) -> None:
        self._session = session
        self._provider = provider
        self._store = EvoEventStore(session.store.root)
        self._evidence = EvidenceAdapter(self._store)
        self._outcomes = OutcomeClassifier(self._store)
        self._compiler = ExperienceCompiler(self._store)
        self._active_run_id: str | None = None
        self._evidence_count = 0
        self._diagnostics: list[str] = []
        self.last_result: AgentEvoRunResult | None = None

    def begin_turn(self) -> str:
        """Start one independent Evo Run for a new user turn."""

        self._active_run_id = new_evo_id("run")
        self._evidence_count = 0
        self._diagnostics = []
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
            verification_kind = _verification_kind(event)
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
        candidate_ids: tuple[str, ...] = ()
        if self._evidence_count:
            try:
                outcome = self._outcomes.classify(run_id)
                if outcome.outcome is not None:
                    outcome_event_id = outcome.outcome.event_id
                if outcome.diagnostic is not None:
                    self._diagnostics.append(f"outcome:{outcome.diagnostic.code}:{outcome.diagnostic.message}")
                candidate = self._compiler.compile(run_id, environment_summary=self._environment_summary())
                candidate_ids = tuple(item.candidate_id for item in candidate.candidates)
                if candidate.diagnostic is not None:
                    self._diagnostics.append(f"candidate:{candidate.diagnostic.code}:{candidate.diagnostic.message}")
            except Exception as error:  # noqa: BLE001 - compilation cannot affect the completed turn
                self._diagnostics.append(f"evo_completion_failed:{type(error).__name__}:{error}")
        result = AgentEvoRunResult(
            run_id=run_id,
            evidence_count=self._evidence_count,
            outcome_event_id=outcome_event_id,
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
        self._remember_recording_result(recorded.persisted, recorded.diagnostic)

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
        self._remember_recording_result(recorded.persisted, recorded.diagnostic)

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
        self._remember_recording_result(recorded.persisted, recorded.diagnostic)

    def _remember_recording_result(self, persisted: bool, diagnostic: object | None) -> None:
        if persisted:
            self._evidence_count += 1
        if diagnostic is not None:
            code = getattr(diagnostic, "code", "unknown")
            message = getattr(diagnostic, "message", str(diagnostic))
            self._diagnostics.append(f"evidence:{code}:{message}")

    def _environment_summary(self) -> str:
        return f"runtime=agent_loop; provider={self._provider.name}; model={self._provider.model}"


def _verification_kind(event: ToolExecutionEvent) -> str | None:
    result = event.result
    if result is None:
        return None
    data = result.data if isinstance(result.data, dict) else {}
    command = _text(data.get("command")) or _text(event.tool_call.arguments.get("command")) or ""
    text = command.lower()
    if any(marker in text for marker in ("pytest", "unittest", "python -m test", "npm test", "pnpm test", "yarn test")):
        return "test"
    if any(marker in text for marker in ("ruff", "flake8", "eslint", "pylint")):
        return "lint"
    if any(marker in text for marker in ("mypy", "pyright", "tsc", "typecheck")):
        return "type_check"
    if any(marker in text for marker in ("python -m build", "npm run build", "pnpm build", "cargo build")):
        return "build"
    return None


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
