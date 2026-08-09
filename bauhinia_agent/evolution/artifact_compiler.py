"""Compile repeated, independent Experience Candidates into Artifact drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bauhinia_agent.evolution.candidate_artifacts import (
    CandidateArtifactDraft,
    CandidateArtifactError,
    CandidateArtifactKind,
    CandidateArtifactRecord,
    CandidateArtifactRegistry,
)
from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    CandidateReviewRecordedPayload,
    EvoEvent,
    ExperienceCandidateCreatedPayload,
)
from bauhinia_agent.evolution.identifiers import require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore

MIN_SUPPORT_CANDIDATES = 3
MIN_DISTINCT_TASKS = 3
_SUCCESS_ARTIFACTS = frozenset(
    {
        CandidateArtifactKind.PLAN_TEMPLATE,
        CandidateArtifactKind.SKILL_DRAFT,
        CandidateArtifactKind.MEMORY_RULE,
    }
)
_SUCCESS_SOURCE_KINDS = frozenset({"plan_template", "stable_fact"})
_FAILURE_SOURCE_KINDS = frozenset({"anti_pattern", "debug_hint"})


class _ArtifactCompilerStore(Protocol):
    def append(self, event: EvoEvent[CandidateArtifactCreatedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class ArtifactDerivationSpec:
    kind: CandidateArtifactKind
    name: str
    support_candidate_ids: tuple[str, ...]
    counterexample_candidate_ids: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    effects: tuple[str, ...]
    scope: str
    triggers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactDerivationDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ArtifactDerivationResult:
    persisted: bool
    artifact: CandidateArtifactRecord | None = None
    diagnostic: ArtifactDerivationDiagnostic | None = None


class CandidateArtifactCompiler:
    """Derive one reviewable Artifact only after independent evidence thresholds."""

    def __init__(self, store: EvoEventStore | _ArtifactCompilerStore) -> None:
        self._store = store

    def derive(self, spec: ArtifactDerivationSpec) -> ArtifactDerivationResult:
        events = self._store.list_events()
        support_ids = _unique_ids(spec.support_candidate_ids, field="support_candidate_ids")
        counterexample_ids = _unique_ids(spec.counterexample_candidate_ids, field="counterexample_candidate_ids")
        if set(support_ids) & set(counterexample_ids):
            return _refusal("overlapping_sources", "support and counterexample Candidates must not overlap")
        if len(support_ids) < MIN_SUPPORT_CANDIDATES:
            return _refusal(
                "insufficient_support",
                f"at least {MIN_SUPPORT_CANDIDATES} independent verified support Candidates are required",
            )
        candidates = _candidates(events)
        requested = (*support_ids, *counterexample_ids)
        missing = [candidate_id for candidate_id in requested if candidate_id not in candidates]
        if missing:
            return _refusal("unknown_sources", f"unknown Experience Candidates: {missing}")
        accepted = _accepted_candidate_ids(events)
        unaccepted = sorted(set(requested) - accepted)
        if unaccepted:
            return _refusal("unaccepted_sources", f"source Candidates require human acceptance: {unaccepted}")
        supports = tuple(candidates[candidate_id] for candidate_id in support_ids)
        counterexamples = tuple(candidates[candidate_id] for candidate_id in counterexample_ids)
        validation = _validate_supports(spec, supports, counterexamples)
        if validation is not None:
            return validation
        existing = _existing_derivation(events, spec.kind, spec.name, support_ids, counterexample_ids)
        if existing is not None:
            return ArtifactDerivationResult(
                False,
                existing,
                ArtifactDerivationDiagnostic("already_derived", "the same reviewed source set already produced this Artifact"),
            )

        source_events = (*supports, *counterexamples)
        draft = CandidateArtifactDraft(
            kind=spec.kind,
            name=spec.name,
            description=f"Reviewed {spec.kind.value} derived from {len(supports)} independent verified tasks.",
            instructions=_instructions(spec.kind, supports),
            inputs=spec.inputs,
            outputs=spec.outputs,
            dependencies=spec.dependencies,
            effects=spec.effects,
            scope=spec.scope,
            applicability=_applicability(supports),
            risks=_risks(source_events),
            source_candidate_ids=requested,
            support_candidate_ids=support_ids,
            counterexample_candidate_ids=counterexample_ids,
            confidence=_combined_confidence(supports),
            triggers=spec.triggers,
        )
        try:
            result = CandidateArtifactRegistry(self._store).create(draft)
        except CandidateArtifactError as error:
            return _refusal("invalid_artifact", str(error))
        if result.artifact is None:
            diagnostic = result.diagnostic
            return ArtifactDerivationResult(
                False,
                diagnostic=ArtifactDerivationDiagnostic(
                    diagnostic.code if diagnostic else "artifact_recording_failed",
                    diagnostic.message if diagnostic else "Artifact recording failed without a diagnostic.",
                ),
            )
        return ArtifactDerivationResult(True, result.artifact)


def _validate_supports(
    spec: ArtifactDerivationSpec,
    supports: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...],
    counterexamples: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...],
) -> ArtifactDerivationResult | None:
    all_events = (*supports, *counterexamples)
    if any(event.payload.scope != spec.scope for event in all_events):
        return _refusal("scope_mismatch", "all source Candidates must match the requested Artifact scope")
    if any(not event.payload.evidence_refs for event in all_events):
        return _refusal("missing_evidence", "every source Candidate must retain Evidence references")
    if any(_outcome_category(event) in {None, "unknown"} for event in all_events):
        return _refusal("unverified_sources", "every source Candidate must retain a classified non-unknown Outcome")
    pattern_keys = {_extension_text(event, "pattern_key") for event in all_events}
    if None in pattern_keys or len(pattern_keys) != 1:
        return _refusal("pattern_mismatch", "source Candidates must share one explicit non-empty pattern key")
    task_signatures = [_extension_text(event, "task_signature") for event in supports]
    if None in task_signatures or "unknown" in task_signatures or len(set(task_signatures)) < MIN_DISTINCT_TASKS:
        return _refusal(
            "insufficient_task_diversity",
            f"at least {MIN_DISTINCT_TASKS} distinct non-unknown task signatures are required",
        )
    if not _independent_runs(all_events):
        return _refusal("dependent_runs", "all source Candidates must come from disjoint Run sets")
    if not _independent_evidence(all_events):
        return _refusal("dependent_evidence", "all source Candidates must not reuse Evidence references")
    source_kinds = {event.payload.kind for event in supports}
    if spec.kind in _SUCCESS_ARTIFACTS:
        if not source_kinds <= _SUCCESS_SOURCE_KINDS or any(_outcome_category(event) != "task_success" for event in supports):
            return _refusal("incompatible_support", "this Artifact kind requires verified successful plan or stable-fact sources")
        if not counterexamples:
            return _refusal("missing_failure_counterexample", "successful Artifact patterns require a verified failure counterexample")
        if any(_outcome_category(event) == "task_success" for event in counterexamples):
            return _refusal("invalid_counterexample", "counterexamples for successful patterns must be classified failures")
    elif spec.kind is CandidateArtifactKind.TOOL_INVOCATION_POLICY:
        if not source_kinds <= _FAILURE_SOURCE_KINDS or any(_outcome_category(event) == "task_success" for event in supports):
            return _refusal("incompatible_support", "tool policies require repeated verified failure or caution sources")
    return None


def _candidates(events: list[EvoEvent]) -> dict[str, EvoEvent[ExperienceCandidateCreatedPayload]]:
    return {
        event.refs.candidate_id: event
        for event in events
        if event.event_type == "ExperienceCandidateCreated"
        and isinstance(event.payload, ExperienceCandidateCreatedPayload)
        and event.refs.candidate_id is not None
        and event.payload.lifecycle_state == "Candidate"
    }


def _accepted_candidate_ids(events: list[EvoEvent]) -> set[str]:
    latest: dict[str, CandidateReviewRecordedPayload] = {}
    for event in events:
        if event.event_type == "CandidateReviewRecorded" and isinstance(event.payload, CandidateReviewRecordedPayload):
            latest[event.payload.candidate_id] = event.payload
    return {candidate_id for candidate_id, review in latest.items() if review.decision == "accept"}


def _existing_derivation(
    events: list[EvoEvent],
    kind: CandidateArtifactKind,
    name: str,
    support_ids: tuple[str, ...],
    counterexample_ids: tuple[str, ...],
) -> CandidateArtifactRecord | None:
    for event in events:
        if event.event_type != "CandidateArtifactCreated" or not isinstance(event.payload, CandidateArtifactCreatedPayload):
            continue
        if (
            event.payload.kind == kind.value
            and event.payload.name == name
            and event.payload.support_candidate_ids == support_ids
            and event.payload.counterexample_candidate_ids == counterexample_ids
            and event.refs.artifact_id is not None
        ):
            return CandidateArtifactRecord(event.event_id, event.refs.artifact_id, event.refs.run_id, event.occurred_at, event.payload)
    return None


def _unique_ids(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise CandidateArtifactError(f"{field} must be a tuple")
    for value in values:
        require_evo_id(value, field=f"{field}[]", kind="candidate")
    if len(set(values)) != len(values):
        raise CandidateArtifactError(f"{field} must contain unique IDs")
    return values


def _extension_text(event: EvoEvent[ExperienceCandidateCreatedPayload], key: str) -> str | None:
    value = event.payload.extensions.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _outcome_category(event: EvoEvent[ExperienceCandidateCreatedPayload]) -> str | None:
    return _extension_text(event, "outcome_category")


def _independent_runs(events: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> bool:
    seen: set[str] = set()
    for event in events:
        run_ids = set((*event.payload.source_run_ids, event.refs.run_id))
        if seen & run_ids:
            return False
        seen.update(run_ids)
    return True


def _independent_evidence(events: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> bool:
    seen: set[str] = set()
    for event in events:
        evidence = set(event.payload.evidence_refs)
        if seen & evidence:
            return False
        seen.update(evidence)
    return True


def _instructions(kind: CandidateArtifactKind, supports: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> str:
    summaries = sorted({event.payload.summary for event in supports})
    heading = {
        CandidateArtifactKind.PLAN_TEMPLATE: "# Reviewed Plan Template",
        CandidateArtifactKind.SKILL_DRAFT: "# Reviewed Skill Draft",
        CandidateArtifactKind.TOOL_INVOCATION_POLICY: "# Reviewed Tool Invocation Policy",
        CandidateArtifactKind.MEMORY_RULE: "# Reviewed Memory Rule",
    }[kind]
    return "\n".join([heading, "", "Apply only within the declared scope and verify the result:", *(f"- {item}" for item in summaries)])


def _applicability(supports: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> str:
    values = sorted({event.payload.applicability for event in supports})
    return " Shared applicability: ".join(values)


def _risks(events: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> tuple[str, ...]:
    risks = {item for event in events for item in event.payload.counterexamples}
    risks.add("Candidate Artifact only; requires Shadow and held-out evaluation before promotion.")
    return tuple(sorted(risks))


def _combined_confidence(supports: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> float:
    average = sum(event.payload.confidence for event in supports) / len(supports)
    return min(0.7, average + 0.2)


def _refusal(code: str, message: str) -> ArtifactDerivationResult:
    return ArtifactDerivationResult(False, diagnostic=ArtifactDerivationDiagnostic(code, message))
