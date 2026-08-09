"""Evidence-bounded Experience Candidate compilation.

The compiler makes append-only candidate drafts from one classified Run.  It
never changes retrieval, memory, permissions, task execution, or candidate
lifecycle state; P6-002 and later phases own comparison and review workflows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from bauhinia_agent.evolution.diagnosis import DiagnosisService, DiagnosisSummary
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceRecord, redact_text
from bauhinia_agent.evolution.events import (
    DecisionRecordedPayload,
    EvoEvent,
    EvoReferences,
    ExperienceCandidateCreatedPayload,
    PlanCreatedPayload,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeClassifier, OutcomeRecord
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError


class ExperienceCompilerError(ValueError):
    """Raised when supplied compiler input is not a safe domain value."""


class _CompilerStore(Protocol):
    def append(self, event: EvoEvent[ExperienceCandidateCreatedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class ExperienceCandidate:
    """A persisted candidate that remains non-operative until a later review."""

    event_id: str
    candidate_id: str
    run_id: str
    occurred_at: str
    payload: ExperienceCandidateCreatedPayload


@dataclass(frozen=True, slots=True)
class ExperienceCompileDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExperienceCompileResult:
    persisted: bool
    candidates: tuple[ExperienceCandidate, ...] = ()
    diagnostic: ExperienceCompileDiagnostic | None = None


class ExperienceCompiler:
    """Compile a single Run into one conservative, evidence-linked draft."""

    def __init__(self, store: EvoEventStore | _CompilerStore) -> None:
        self._store = store
        self._evidence = EvidenceAdapter(store)
        self._outcomes = OutcomeClassifier(store)

    def compile(self, run_id: str, *, environment_summary: str) -> ExperienceCompileResult:
        """Persist one ``Candidate`` draft, or refuse when its evidence is insufficient."""

        require_evo_id(run_id, field="run_id", kind="run")
        environment = _environment_summary(environment_summary)
        evidence = self._evidence.list_for_run(run_id)
        if not evidence:
            return ExperienceCompileResult(
                persisted=False,
                diagnostic=ExperienceCompileDiagnostic(
                    "insufficient_evidence",
                    "Experience candidates require at least one persisted evidence record.",
                ),
            )
        outcomes = self._outcomes.list_for_run(run_id)
        if not outcomes:
            return ExperienceCompileResult(
                persisted=False,
                diagnostic=ExperienceCompileDiagnostic(
                    "missing_outcome",
                    "Experience candidates require a persisted Outcome classification.",
                ),
            )
        outcome = outcomes[-1]
        if outcome.payload.category == "unknown":
            return ExperienceCompileResult(
                persisted=False,
                diagnostic=ExperienceCompileDiagnostic(
                    "insufficient_evidence",
                    "The Outcome is unknown, so no candidate conclusion is supported.",
                ),
            )
        events = self._store.list_events()
        source_events = _source_events(events, run_id, evidence, outcome)
        diagnosis = DiagnosisService(self._store).diagnose(run_id)
        payload = _candidate_payload(run_id, source_events, evidence, outcome, diagnosis, environment)
        candidate_id = new_evo_id("candidate")
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ExperienceCandidateCreated",
            refs=EvoReferences(run_id=run_id, candidate_id=candidate_id, parent_event_id=outcome.event_id),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return ExperienceCompileResult(False, diagnostic=ExperienceCompileDiagnostic("candidate_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - candidate recording must not alter the source Run
            return ExperienceCompileResult(
                False,
                diagnostic=ExperienceCompileDiagnostic("candidate_recording_failed", f"unexpected candidate recorder failure: {error}"),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = ExperienceCompileDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return ExperienceCompileResult(True, (_candidate_from_event(appended.event),), diagnostic)

    def list_for_run(self, run_id: str) -> list[ExperienceCandidate]:
        """Return persisted candidate drafts for one Run in canonical event order."""

        require_evo_id(run_id, field="run_id", kind="run")
        return [
            _candidate_from_event(event)
            for event in self._store.list_events()
            if event.event_type == "ExperienceCandidateCreated" and event.refs.run_id == run_id and isinstance(event.payload, ExperienceCandidateCreatedPayload) and event.refs.candidate_id is not None
        ]


def _environment_summary(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceCompilerError("environment_summary must be a non-blank string")
    return redact_text(value)[0]


def _source_events(events: list[EvoEvent], run_id: str, evidence: list[EvidenceRecord], outcome: OutcomeRecord) -> tuple[EvoEvent, ...]:
    evidence_event_ids = {record.event_id for record in evidence}
    selected: list[EvoEvent] = []
    for event in events:
        if event.refs.run_id != run_id:
            continue
        if event.event_id in evidence_event_ids or event.event_id == outcome.event_id:
            selected.append(event)
        elif event.event_type in {"PlanCreated", "DecisionRecorded"}:
            selected.append(event)
    return tuple(selected)


def _candidate_payload(
    run_id: str,
    source_events: tuple[EvoEvent, ...],
    evidence: list[EvidenceRecord],
    outcome: OutcomeRecord,
    diagnosis: DiagnosisSummary,
    environment_summary: str,
) -> ExperienceCandidateCreatedPayload:
    goal = _latest_goal(source_events)
    action = _latest_action(source_events)
    kind, summary, applicability, counterexamples = _candidate_text(outcome, diagnosis, goal, action)
    return ExperienceCandidateCreatedPayload(
        kind=kind,
        summary=summary,
        scope="project",
        applicability=applicability,
        confidence=min(outcome.payload.confidence, 0.4),
        source_event_ids=tuple(event.event_id for event in source_events),
        evidence_refs=tuple(record.evidence_id for record in evidence),
        counterexamples=counterexamples,
        novelty=None,
        novelty_status="unassessed",
        source_run_ids=(run_id,),
        environment_summary=environment_summary,
        lifecycle_state="Candidate",
        extensions={
            "compiler_version": "p6-001",
            "outcome_category": outcome.payload.category,
            "diagnosis_confidence": diagnosis.confidence,
            "candidate_limit": "single-run draft; not eligible for retrieval or promotion",
            "task_signature": _task_signature(goal, action),
            "pattern_key": _pattern_key(action, goal),
        },
    )


def _candidate_text(outcome: OutcomeRecord, diagnosis: DiagnosisSummary, goal: str | None, action: str | None) -> tuple[str, str, str, tuple[str, ...]]:
    context = action or goal or "the recorded run"
    if outcome.payload.category == "task_success":
        return (
            "plan_template" if action or goal else "stable_fact",
            f"Candidate only: {context} was followed by verified success in this recorded run.",
            f"Runs with a comparable goal and verification method to {context}.",
            ("Do not generalize from one run; require independent evidence before reuse or promotion.",),
        )
    if outcome.payload.category == "verification_failure":
        next_step = diagnosis.next_verification or "Inspect and rerun the focused verification."
        return (
            "debug_hint",
            f"Hypothesis only: before retrying {context}, {next_step}",
            "Runs classified with deterministic verification failure evidence.",
            ("This does not establish a root cause; do not use it when verification evidence is absent or conflicting.",),
        )
    next_step = diagnosis.next_verification or "Collect additional deterministic evidence before retrying."
    return (
        "anti_pattern",
        f"Hypothesis only: do not repeat {context} unchanged after {outcome.payload.category}; {next_step}",
        f"Runs classified as {outcome.payload.category} with comparable evidence.",
        ("Do not apply this candidate when the recorded environment, permission, or tool conditions differ.",),
    )


def _latest_goal(events: tuple[EvoEvent, ...]) -> str | None:
    for event in reversed(events):
        if isinstance(event.payload, PlanCreatedPayload):
            return redact_text(event.payload.goal)[0]
    return None


def _latest_action(events: tuple[EvoEvent, ...]) -> str | None:
    for event in reversed(events):
        if isinstance(event.payload, DecisionRecordedPayload):
            return redact_text(event.payload.selected_action)[0]
    return None


def _task_signature(goal: str | None, action: str | None) -> str:
    if goal is None and action is None:
        return "unknown"
    return _fingerprint("task", goal or "", action or "")


def _pattern_key(action: str | None, goal: str | None) -> str:
    return _fingerprint("pattern", action or goal or "unspecified")


def _fingerprint(*parts: str) -> str:
    normalized = "\x1f".join(" ".join(part.lower().split()) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _candidate_from_event(event: EvoEvent[ExperienceCandidateCreatedPayload]) -> ExperienceCandidate:
    candidate_id = event.refs.candidate_id
    if candidate_id is None:
        raise ExperienceCompilerError("ExperienceCandidateCreated event requires candidate_id")
    return ExperienceCandidate(
        event_id=event.event_id,
        candidate_id=candidate_id,
        run_id=event.refs.run_id,
        occurred_at=event.occurred_at,
        payload=event.payload,
    )
