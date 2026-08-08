"""Evidence recording adapters for verifiable Evo run facts.

The adapter is deliberately an application-facing boundary: callers translate
test, tool, and permission observations into immutable ``EvidenceRecorded``
events. It does not execute commands, decide permissions, or change the result
of the action that produced the observation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Protocol

from bauhinia_agent.evolution.events import EvidenceRecordedPayload, EvoEvent, EvoReferences
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError


class EvidenceError(ValueError):
    """Raised when a caller supplies an invalid evidence observation."""


_EVIDENCE_TYPES = frozenset({"test", "lint", "type_check", "build", "diff", "tool", "permission", "user_confirmation", "manual"})
_NAMED_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password|cookie)\b\s*([=:])\s*([^\s,;'\"]+)")
_BEARER_RE = re.compile(r"(?i)\b(authorization)\s*:\s*bearer\s+[^\s,;'\"]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")


class _EvidenceStore(Protocol):
    def append(self, event: EvoEvent[EvidenceRecordedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class EvidenceInput:
    """A normalized observation from one deterministic or user-facing source."""

    run_id: str
    evidence_type: str
    source: str
    summary: str
    locator: str | None = None
    verified: bool = False
    command: str | None = None
    input_summary: str | None = None
    cwd: str | None = None
    exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """A persisted, queryable evidence event represented without UI state."""

    event_id: str
    evidence_id: str
    run_id: str
    occurred_at: str
    payload: EvidenceRecordedPayload

    @property
    def evidence_type(self) -> str:
        return self.payload.evidence_type


@dataclass(frozen=True, slots=True)
class EvidenceDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EvidenceRecordResult:
    persisted: bool
    evidence: EvidenceRecord | None = None
    diagnostic: EvidenceDiagnostic | None = None


class EvidenceAdapter:
    """Translate evidence sources into append-only events and query them by Run."""

    def __init__(self, store: EvoEventStore | _EvidenceStore) -> None:
        self._store = store

    def record(self, observation: EvidenceInput) -> EvidenceRecordResult:
        """Persist an evidence observation without allowing recorder failure to escape."""

        event = self._event_from_input(observation)
        try:
            append_result = self._store.append(event)
        except EvoStoreError as error:
            return EvidenceRecordResult(
                persisted=False,
                diagnostic=EvidenceDiagnostic(code="evidence_recording_failed", message=str(error)),
            )
        except Exception as error:  # noqa: BLE001 - evidence recording must not alter the original action outcome
            return EvidenceRecordResult(
                persisted=False,
                diagnostic=EvidenceDiagnostic(code="evidence_recording_failed", message=f"unexpected evidence recorder failure: {error}"),
            )
        evidence = _record_from_event(append_result.event)
        diagnostic = None
        if append_result.diagnostic is not None:
            diagnostic = EvidenceDiagnostic(code=append_result.diagnostic.code, message=append_result.diagnostic.message)
        return EvidenceRecordResult(persisted=True, evidence=evidence, diagnostic=diagnostic)

    def record_tool(
        self,
        *,
        run_id: str,
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, object],
        ok: bool,
        summary: str,
    ) -> EvidenceRecordResult:
        """Record a completed tool observation using a deterministic argument summary."""

        if not isinstance(tool_name, str) or not tool_name.strip():
            raise EvidenceError("tool_name must be a non-blank string")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise EvidenceError("tool_call_id must be a non-blank string")
        return self.record(
            EvidenceInput(
                run_id=run_id,
                evidence_type="tool",
                source=tool_name,
                summary=summary,
                locator=tool_call_id,
                verified=False,
                input_summary=_json_summary(arguments),
                exit_code=0 if ok else 1,
            )
        )

    def record_permission(
        self,
        *,
        run_id: str,
        action: str,
        target: str,
        decision: str,
        reason: str = "",
    ) -> EvidenceRecordResult:
        """Record an authoritative permission decision without changing that decision."""

        if not isinstance(action, str) or not action.strip():
            raise EvidenceError("action must be a non-blank string")
        if not isinstance(decision, str) or not decision.strip():
            raise EvidenceError("decision must be a non-blank string")
        summary = reason or f"permission decision: {decision}"
        return self.record(
            EvidenceInput(
                run_id=run_id,
                evidence_type="permission",
                source="permission_manager",
                summary=summary,
                locator=action,
                input_summary=target,
                verified=True,
            )
        )

    def list_for_run(self, run_id: str) -> list[EvidenceRecord]:
        """Return canonical evidence facts for one Run in append order."""

        require_evo_id(run_id, field="run_id", kind="run")
        records: list[EvidenceRecord] = []
        for event in self._store.list_events():
            if event.event_type != "EvidenceRecorded" or event.refs.run_id != run_id:
                continue
            if not isinstance(event.payload, EvidenceRecordedPayload) or event.refs.evidence_id is None:
                continue
            records.append(_record_from_event(event))
        return records

    def _event_from_input(self, observation: EvidenceInput) -> EvoEvent[EvidenceRecordedPayload]:
        require_evo_id(observation.run_id, field="run_id", kind="run")
        if observation.evidence_type not in _EVIDENCE_TYPES:
            raise EvidenceError(f"unsupported evidence_type: {observation.evidence_type}")
        if isinstance(observation.exit_code, bool) or (observation.exit_code is not None and not isinstance(observation.exit_code, int)):
            raise EvidenceError("exit_code must be an integer or null")
        values = _redact_fields(
            source=observation.source,
            summary=observation.summary,
            locator=observation.locator,
            command=observation.command,
            input_summary=observation.input_summary,
            cwd=observation.cwd,
        )
        payload = EvidenceRecordedPayload(
            evidence_type=observation.evidence_type,
            source=_require_text(values["source"], field="source"),
            summary=_require_text(values["summary"], field="summary"),
            locator=values["locator"],
            verified=observation.verified,
            command=values["command"],
            input_summary=values["input_summary"],
            cwd=values["cwd"],
            exit_code=observation.exit_code,
            redacted=values["redacted"],
        )
        evidence_id = new_evo_id("evidence")
        return EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvidenceRecorded",
            refs=EvoReferences(run_id=observation.run_id, evidence_id=evidence_id),
            payload=payload,
        )


def _record_from_event(event: EvoEvent[EvidenceRecordedPayload]) -> EvidenceRecord:
    evidence_id = event.refs.evidence_id
    if evidence_id is None:
        raise EvidenceError("EvidenceRecorded event requires evidence_id")
    return EvidenceRecord(
        event_id=event.event_id,
        evidence_id=evidence_id,
        run_id=event.refs.run_id,
        occurred_at=event.occurred_at,
        payload=event.payload,
    )


def _json_summary(arguments: Mapping[str, object]) -> str:
    try:
        raw = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise EvidenceError(f"tool arguments must be JSON-compatible: {error}") from error
    return redact_text(raw)[0]


def _redact_fields(**fields: str | None) -> dict[str, str | bool | None]:
    result: dict[str, str | bool | None] = {}
    was_redacted = False
    for name, value in fields.items():
        if value is not None and not isinstance(value, str):
            raise EvidenceError(f"{name} must be a string or null")
        sanitized, changed = redact_text(value) if value is not None else (None, False)
        result[name] = sanitized
        was_redacted = was_redacted or changed
    result["redacted"] = was_redacted
    return result


def redact_text(value: str) -> tuple[str, bool]:
    """Redact common credential forms before a new Evo record is persisted."""

    if not isinstance(value, str):
        raise EvidenceError("value must be a string")
    redacted = _BEARER_RE.sub(lambda match: f"{match.group(1)}: Bearer [REDACTED]", value)
    redacted = _NAMED_SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted)
    redacted = _OPENAI_KEY_RE.sub("[REDACTED]", redacted)
    return redacted, redacted != value


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{field} must be a non-blank string")
    return value
