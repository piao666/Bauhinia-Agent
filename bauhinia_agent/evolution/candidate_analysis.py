"""Deterministic, non-mutating analysis of Experience Candidate drafts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from bauhinia_agent.evolution.evidence import redact_text
from bauhinia_agent.evolution.events import (
    CandidateConflictDetectedPayload,
    CandidateMergeProposedPayload,
    EvoEvent,
    EvoReferences,
    ExperienceCandidateCreatedPayload,
)
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

_DUPLICATE_SIMILARITY = 0.70
_CONFLICT_SIMILARITY = 0.40
_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
_STOP_WORDS = frozenset(
    {
        "after",
        "and",
        "before",
        "candidate",
        "comparable",
        "do",
        "in",
        "is",
        "method",
        "only",
        "or",
        "recorded",
        "run",
        "runs",
        "the",
        "this",
        "to",
        "verification",
        "with",
    }
)


class _CandidateAnalysisStore(Protocol):
    def append(self, event: EvoEvent) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class CandidateMergeProposal:
    event_id: str
    run_id: str
    occurred_at: str
    payload: CandidateMergeProposedPayload


@dataclass(frozen=True, slots=True)
class CandidateConflictGroup:
    event_id: str
    run_id: str
    occurred_at: str
    payload: CandidateConflictDetectedPayload


@dataclass(frozen=True, slots=True)
class CandidateAnalysisDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CandidateAnalysisResult:
    persisted: bool
    merge_proposals: tuple[CandidateMergeProposal, ...] = ()
    conflict_groups: tuple[CandidateConflictGroup, ...] = ()
    diagnostic: CandidateAnalysisDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class _AppendOutcome:
    record: CandidateMergeProposal | CandidateConflictGroup | None
    diagnostic: CandidateAnalysisDiagnostic | None = None


class CandidateAnalysisService:
    """Create append-only merge proposals and conflict groups for review.

    The service only compares ``Candidate`` drafts in the same scope. It never
    alters their payloads, lifecycle state, retrieval behavior, or permissions.
    """

    def __init__(self, store: EvoEventStore | _CandidateAnalysisStore) -> None:
        self._store = store

    def analyze(self) -> CandidateAnalysisResult:
        """Persist previously unseen duplicate and conflicting groups."""

        events = self._store.list_events()
        candidates = _candidate_events(events)
        existing_merges = _existing_ids(events, "CandidateMergeProposed", "cluster_id")
        existing_conflicts = _existing_ids(events, "CandidateConflictDetected", "conflict_group_id")
        merge_proposals: list[CandidateMergeProposal] = []
        conflict_groups: list[CandidateConflictGroup] = []
        diagnostic: CandidateAnalysisDiagnostic | None = None

        for group in _duplicate_groups(candidates):
            payload = _merge_payload(group)
            if payload.cluster_id in existing_merges:
                continue
            result = self._append("CandidateMergeProposed", group[0], payload)
            if result.record is None:
                return CandidateAnalysisResult(False, tuple(merge_proposals), tuple(conflict_groups), result.diagnostic)
            if not isinstance(result.record, CandidateMergeProposal):
                return CandidateAnalysisResult(
                    False, tuple(merge_proposals), tuple(conflict_groups), CandidateAnalysisDiagnostic("candidate_analysis_recording_failed", "merge proposal returned an invalid record type")
                )
            merge_proposals.append(result.record)
            diagnostic = result.diagnostic or diagnostic
            existing_merges.add(payload.cluster_id)

        for group in _conflict_groups(candidates):
            payload = _conflict_payload(group)
            if payload.conflict_group_id in existing_conflicts:
                continue
            result = self._append("CandidateConflictDetected", group[0], payload)
            if result.record is None:
                return CandidateAnalysisResult(False, tuple(merge_proposals), tuple(conflict_groups), result.diagnostic)
            if not isinstance(result.record, CandidateConflictGroup):
                return CandidateAnalysisResult(
                    False, tuple(merge_proposals), tuple(conflict_groups), CandidateAnalysisDiagnostic("candidate_analysis_recording_failed", "conflict group returned an invalid record type")
                )
            conflict_groups.append(result.record)
            diagnostic = result.diagnostic or diagnostic
            existing_conflicts.add(payload.conflict_group_id)

        return CandidateAnalysisResult(True, tuple(merge_proposals), tuple(conflict_groups), diagnostic)

    def _append(
        self,
        event_type: str,
        source: EvoEvent[ExperienceCandidateCreatedPayload],
        payload: CandidateMergeProposedPayload | CandidateConflictDetectedPayload,
    ) -> _AppendOutcome:
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type=event_type,
            refs=EvoReferences(run_id=source.refs.run_id, parent_event_id=source.event_id),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return _AppendOutcome(None, CandidateAnalysisDiagnostic("candidate_analysis_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - analysis cannot change any candidate or source Run
            return _AppendOutcome(None, CandidateAnalysisDiagnostic("candidate_analysis_recording_failed", f"unexpected candidate analysis failure: {error}"))
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = CandidateAnalysisDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        if isinstance(appended.event.payload, CandidateMergeProposedPayload):
            return _AppendOutcome(_merge_from_event(appended.event), diagnostic)
        return _AppendOutcome(_conflict_from_event(appended.event), diagnostic)


def _candidate_events(events: list[EvoEvent]) -> list[EvoEvent[ExperienceCandidateCreatedPayload]]:
    return [
        event
        for event in events
        if event.event_type == "ExperienceCandidateCreated"
        and isinstance(event.payload, ExperienceCandidateCreatedPayload)
        and event.refs.candidate_id is not None
        and event.payload.lifecycle_state == "Candidate"
    ]


def _existing_ids(events: list[EvoEvent], event_type: str, field: str) -> set[str]:
    result: set[str] = set()
    for event in events:
        if event.event_type != event_type:
            continue
        value = getattr(event.payload, field, None)
        if isinstance(value, str):
            result.add(value)
    return result


def _duplicate_groups(candidates: list[EvoEvent[ExperienceCandidateCreatedPayload]]) -> list[tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]]:
    return _connected_groups(
        candidates,
        lambda left, right: left.payload.scope == right.payload.scope and left.payload.kind == right.payload.kind and _similarity(left, right) >= _DUPLICATE_SIMILARITY,
    )


def _conflict_groups(candidates: list[EvoEvent[ExperienceCandidateCreatedPayload]]) -> list[tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]]:
    return _connected_groups(
        candidates,
        lambda left, right: left.payload.scope == right.payload.scope
        and _conclusion(left) != "unknown"
        and _conclusion(right) != "unknown"
        and _conclusion(left) != _conclusion(right)
        and _similarity(left, right) >= _CONFLICT_SIMILARITY,
    )


def _connected_groups(
    candidates: list[EvoEvent[ExperienceCandidateCreatedPayload]],
    predicate: Callable[[EvoEvent[ExperienceCandidateCreatedPayload], EvoEvent[ExperienceCandidateCreatedPayload]], bool],
) -> list[tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]]:
    adjacency: dict[str, set[str]] = {event.refs.candidate_id: set() for event in candidates if event.refs.candidate_id is not None}
    by_id = {event.refs.candidate_id: event for event in candidates if event.refs.candidate_id is not None}
    for index, left in enumerate(candidates):
        left_id = left.refs.candidate_id
        if left_id is None:
            continue
        for right in candidates[index + 1 :]:
            right_id = right.refs.candidate_id
            if right_id is None or not predicate(left, right):
                continue
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)
    groups: list[tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]] = []
    seen: set[str] = set()
    for candidate_id in sorted(adjacency):
        if candidate_id in seen or not adjacency[candidate_id]:
            continue
        pending = [candidate_id]
        group_ids: set[str] = set()
        while pending:
            current = pending.pop()
            if current in group_ids:
                continue
            group_ids.add(current)
            pending.extend(adjacency[current] - group_ids)
        seen.update(group_ids)
        groups.append(tuple(by_id[item] for item in sorted(group_ids)))
    return groups


def _merge_payload(group: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> CandidateMergeProposedPayload:
    first = group[0].payload
    candidate_ids = _candidate_ids(group)
    features = _shared_features(group)
    return CandidateMergeProposedPayload(
        cluster_id=_group_id("merge", first.scope, candidate_ids),
        scope=first.scope,
        kind=first.kind,
        candidate_ids=candidate_ids,
        source_run_ids=_source_run_ids(group),
        evidence_refs=_evidence_refs(group),
        task_features=features,
        similarity=_group_similarity(group),
        proposal_summary="Review-only proposal to merge near-duplicate Candidate drafts; source candidates remain unchanged.",
        extensions={"analysis_version": "p6-002", "action": "propose_merge"},
    )


def _conflict_payload(group: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> CandidateConflictDetectedPayload:
    first = group[0].payload
    candidate_ids = _candidate_ids(group)
    return CandidateConflictDetectedPayload(
        conflict_group_id=_group_id("conflict", first.scope, candidate_ids),
        scope=first.scope,
        candidate_ids=candidate_ids,
        conclusions=tuple(sorted({_conclusion(event) for event in group})),
        source_run_ids=_source_run_ids(group),
        evidence_refs=_evidence_refs(group),
        task_features=_shared_features(group),
        similarity=_group_similarity(group),
        summary="Conflicting Candidate conclusions were detected; no candidate was selected, merged, or changed.",
        extensions={"analysis_version": "p6-002", "requires_review": True},
    )


def _candidate_ids(group: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> tuple[str, ...]:
    return tuple(sorted(event.refs.candidate_id for event in group if event.refs.candidate_id is not None))


def _source_run_ids(group: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> tuple[str, ...]:
    return tuple(sorted({run_id for event in group for run_id in (*event.payload.source_run_ids, event.refs.run_id)}))


def _evidence_refs(group: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> tuple[str, ...]:
    return tuple(sorted({ref for event in group for ref in event.payload.evidence_refs}))


def _features(event: EvoEvent[ExperienceCandidateCreatedPayload]) -> frozenset[str]:
    text = redact_text(f"{event.payload.applicability} {event.payload.summary}")[0].lower()
    return frozenset(token for token in _TOKEN_RE.findall(text) if token not in _STOP_WORDS)


def _shared_features(group: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> tuple[str, ...]:
    if not group:
        return ()
    shared = set(_features(group[0]))
    for event in group[1:]:
        shared.intersection_update(_features(event))
    return tuple(sorted(shared))


def _similarity(left: EvoEvent[ExperienceCandidateCreatedPayload], right: EvoEvent[ExperienceCandidateCreatedPayload]) -> float:
    left_features = _features(left)
    right_features = _features(right)
    union = left_features | right_features
    return 0.0 if not union else len(left_features & right_features) / len(union)


def _group_similarity(group: tuple[EvoEvent[ExperienceCandidateCreatedPayload], ...]) -> float:
    pairs = [_similarity(left, right) for index, left in enumerate(group) for right in group[index + 1 :]]
    return min(pairs) if pairs else 1.0


def _conclusion(event: EvoEvent[ExperienceCandidateCreatedPayload]) -> str:
    if event.payload.kind in {"plan_template", "stable_fact"}:
        return "support"
    if event.payload.kind in {"anti_pattern", "debug_hint"}:
        return "caution"
    return "unknown"


def _group_id(prefix: str, scope: str, candidate_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\x1f".join((prefix, scope, *candidate_ids)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _merge_from_event(event: EvoEvent[CandidateMergeProposedPayload]) -> CandidateMergeProposal:
    return CandidateMergeProposal(event.event_id, event.refs.run_id, event.occurred_at, event.payload)


def _conflict_from_event(event: EvoEvent[CandidateConflictDetectedPayload]) -> CandidateConflictGroup:
    return CandidateConflictGroup(event.event_id, event.refs.run_id, event.occurred_at, event.payload)
