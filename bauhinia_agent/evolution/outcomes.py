"""Deterministic Outcome classification from append-only Evidence facts."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal, Protocol

from bauhinia_agent.evolution.evidence import (
    EvidenceAdapter,
    EvidenceIntegrityError,
    EvidenceRecord,
    resolve_evidence_records,
)
from bauhinia_agent.evolution.events import (
    EvidenceRecordedPayload,
    EvoEvent,
    EvoReferences,
    OutcomeClassifiedPayload,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

OutcomeValue = Literal["success", "failure", "cancelled", "timeout", "unknown"]
OutcomeCategory = Literal[
    "task_success",
    "task_failure",
    "verification_failure",
    "tool_failure",
    "environment_failure",
    "permission_denied",
    "cancelled",
    "timeout",
    "evaluation_infrastructure_failure",
    "unknown",
]


class OutcomeError(ValueError):
    """Raised when an outcome event cannot satisfy its domain contract."""


class OutcomeIntegrityError(OutcomeError):
    """Raised when a persisted Outcome does not match its prior Evidence facts."""


class _OutcomeStore(Protocol):
    def append(self, event: EvoEvent[OutcomeClassifiedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    event_id: str
    run_id: str
    occurred_at: str
    payload: OutcomeClassifiedPayload


@dataclass(frozen=True, slots=True)
class OutcomeDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OutcomeResult:
    persisted: bool
    outcome: OutcomeRecord | None = None
    diagnostic: OutcomeDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class _Classification:
    outcome: OutcomeValue
    category: OutcomeCategory
    confidence: float
    summary: str


class OutcomeClassifier:
    """Classify one Run using evidence precedence that is stable and inspectable.

    Precedence protects terminal safety states from being hidden by a later test
    result: permission rejection, cancellation, timeout, evaluation
    infrastructure, environment, tool, verification, task, then unknown.
    """

    def __init__(self, store: EvoEventStore | _OutcomeStore) -> None:
        self._store = store
        self._evidence = EvidenceAdapter(store)

    def classify(self, run_id: str) -> OutcomeResult:
        """Append one outcome fact for a Run without mutating evidence or execution."""

        require_evo_id(run_id, field="run_id", kind="run")
        evidence = self._evidence.list_for_run(run_id)
        classification = _classify(evidence)
        payload = OutcomeClassifiedPayload(
            outcome=classification.outcome,
            category=classification.category,
            summary=classification.summary,
            evidence_refs=tuple(record.evidence_id for record in evidence),
            confidence=classification.confidence,
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="OutcomeClassified",
            refs=EvoReferences(run_id=run_id, parent_event_id=evidence[-1].event_id if evidence else None),
            payload=payload,
        )
        try:
            append_result = self._store.append(event)
        except EvoStoreError as error:
            return OutcomeResult(False, diagnostic=OutcomeDiagnostic("outcome_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - classification must not change the original Run result
            return OutcomeResult(False, diagnostic=OutcomeDiagnostic("outcome_recording_failed", f"unexpected outcome recorder failure: {error}"))
        diagnostic = None
        if append_result.diagnostic is not None:
            diagnostic = OutcomeDiagnostic(append_result.diagnostic.code, append_result.diagnostic.message)
        return OutcomeResult(True, outcome=_record_from_event(append_result.event), diagnostic=diagnostic)

    def list_for_run(self, run_id: str) -> list[OutcomeRecord]:
        """Return all Outcome facts for a Run in append order."""

        require_evo_id(run_id, field="run_id", kind="run")
        records: list[OutcomeRecord] = []
        for event in self._store.list_events():
            if event.event_type == "OutcomeClassified" and event.refs.run_id == run_id and isinstance(event.payload, OutcomeClassifiedPayload):
                records.append(_record_from_event(event))
        return records


def attest_outcome_event(
    events: list[EvoEvent] | tuple[EvoEvent, ...],
    outcome_event: EvoEvent,
) -> tuple[EvidenceRecord, ...]:
    """Recompute a persisted Outcome from all prior Evidence in its Run.

    The Outcome is a derived fact, so consumers must not trust a caller-authored
    payload.  Requiring the exact prior Evidence inventory also prevents an
    Outcome from omitting a failure or acquiring Evidence appended later.
    """

    if outcome_event.event_type != "OutcomeClassified" or not isinstance(outcome_event.payload, OutcomeClassifiedPayload):
        raise OutcomeIntegrityError("source event is not OutcomeClassified")
    if outcome_event.sequence is None:
        raise OutcomeIntegrityError("OutcomeClassified must have a persisted sequence")
    run_id = require_evo_id(outcome_event.refs.run_id, field="run_id", kind="run")
    prior_evidence_ids = tuple(
        event.refs.evidence_id
        for event in events
        if event.event_type == "EvidenceRecorded"
        and isinstance(event.payload, EvidenceRecordedPayload)
        and event.refs.run_id == run_id
        and event.refs.evidence_id is not None
        and event.sequence is not None
        and event.sequence < outcome_event.sequence
    )
    if outcome_event.payload.evidence_refs != prior_evidence_ids:
        raise OutcomeIntegrityError("Outcome Evidence references must exactly match all prior Evidence in the Run")
    try:
        records = resolve_evidence_records(
            events,
            prior_evidence_ids,
            run_id=run_id,
            before_sequence=outcome_event.sequence,
        )
    except (EvidenceIntegrityError, ValueError) as error:
        raise OutcomeIntegrityError(f"Outcome Evidence is invalid: {error}") from error
    classification = _classify(list(records))
    payload = outcome_event.payload
    if payload.outcome != classification.outcome or payload.category != classification.category or payload.confidence != classification.confidence or payload.summary != classification.summary:
        raise OutcomeIntegrityError("Outcome payload does not match deterministic classification of prior Evidence")
    return records


def _classify(evidence: list[EvidenceRecord]) -> _Classification:
    if _matches(evidence, lambda record, text: record.evidence_type == "permission" and _has(text, "deny", "denied", "reject", "rejected")):
        return _result("failure", "permission_denied", 0.99, evidence)
    if _matches(evidence, lambda _record, text: _has(text, "cancelled", "canceled", "cancel requested")):
        return _result("cancelled", "cancelled", 0.98, evidence)
    if _matches(evidence, lambda _record, text: _has(text, "timed out", "timeout", "deadline exceeded")):
        return _result("timeout", "timeout", 0.98, evidence)
    if _matches(
        evidence,
        lambda record, text: ("eval" in record.payload.source.lower() or "evaluator" in record.payload.source.lower()) and _has(text, "infrastructure", "unavailable", "runner", "evaluator"),
    ):
        return _result("failure", "evaluation_infrastructure_failure", 0.95, evidence)
    if _matches(evidence, lambda _record, text: _has(text, "network unavailable", "dns", "connection refused", "environment unavailable")):
        return _result("failure", "environment_failure", 0.90, evidence)
    if _matches(evidence, lambda record, text: record.evidence_type == "tool" and (record.payload.exit_code not in {None, 0} or _has(text, "invalid tool argument", "unknown tool"))):
        return _result("failure", "tool_failure", 0.90, evidence)
    if _matches(
        evidence,
        lambda record, text: record.evidence_type in {"test", "lint", "type_check", "build", "diff"} and (record.payload.exit_code not in {None, 0} or _has(text, "failed", "failure")),
    ):
        return _result("failure", "verification_failure", 0.95, evidence)
    if _matches(evidence, lambda _record, text: _has(text, "task failed", "acceptance failed")):
        return _result("failure", "task_failure", 0.75, evidence)
    if _matches(
        evidence,
        lambda record, _text: record.evidence_type in {"test", "lint", "type_check", "build", "diff"} and record.payload.verified and record.payload.exit_code == 0,
    ):
        return _result("success", "task_success", 0.95, evidence)
    return _result("unknown", "unknown", 0.20 if evidence else 0.0, evidence)


def _matches(evidence: list[EvidenceRecord], predicate: Callable[[EvidenceRecord, str], bool]) -> bool:
    return any(predicate(record, _evidence_text(record)) for record in evidence)


def _evidence_text(record: EvidenceRecord) -> str:
    payload = record.payload
    return " ".join(value for value in (payload.source, payload.summary, payload.locator, payload.command, payload.input_summary) if value).lower()


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _result(outcome: OutcomeValue, category: OutcomeCategory, confidence: float, evidence: list[EvidenceRecord]) -> _Classification:
    return _Classification(
        outcome=outcome,
        category=category,
        confidence=confidence,
        summary=f"{category} classified from {len(evidence)} evidence record(s)",
    )


def _record_from_event(event: EvoEvent[OutcomeClassifiedPayload]) -> OutcomeRecord:
    return OutcomeRecord(event_id=event.event_id, run_id=event.refs.run_id, occurred_at=event.occurred_at, payload=event.payload)
