"""Evidence-governed collaboration contracts for P10.

This module is deliberately runtime-free.  It records parent/child facts,
detects resource and conclusion conflicts, and groups independent evidence.  It
does not create an AgentLoop, execute a tool, or make a permission decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Literal, Mapping, Sequence, cast

from bauhinia_agent.evolution.events import (
    CollaborationConflictDetectedPayload,
    CollaborationClaimPayload,
    CollaborationRunAggregatedPayload,
    CollaborationTaskDelegatedPayload,
    CollaborationTaskResultRecordedPayload,
    EvoEvent,
    EvoReferences,
    OutcomeClassifiedPayload,
    collaboration_claim_fingerprint,
)
from bauhinia_agent.evolution.evidence import EvidenceIntegrityError, redact_text, resolve_evidence_records
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeIntegrityError, attest_outcome_event
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError
from bauhinia_agent.planning.evo import PlanningRole, TaskContract

CollaborationStatus = Literal["success", "failure", "cancelled", "timeout", "permission_denied"]
ConflictKind = Literal["resource", "conclusion"]

_STATUSES = frozenset({"success", "failure", "cancelled", "timeout", "permission_denied"})
_ROLES = frozenset({"planner", "researcher", "executor", "verifier", "critic", "curator"})
_TRUSTED_EVIDENCE_TYPES = frozenset({"test", "lint", "type_check", "build", "diff"})


class CollaborationError(ValueError):
    """A collaboration hand-off or aggregate violates its domain contract."""


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollaborationError(f"{field} must be a non-blank string")
    return value


def _unique(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(_text(value, field=f"{field}[]") for value in values)
    if len(result) != len(set(result)):
        raise CollaborationError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class CollaborationClaim:
    """A reviewable conclusion with explicit evidence and provenance."""

    claim_key: str
    conclusion: str
    evidence_refs: tuple[str, ...]
    source_role: PlanningRole
    independence_key: str

    def __post_init__(self) -> None:
        _text(self.claim_key, field="claim_key")
        _text(self.conclusion, field="conclusion")
        _unique(self.evidence_refs, field="evidence_refs")
        _text(self.independence_key, field="independence_key")
        if self.source_role not in _ROLES:
            raise CollaborationError(f"unknown collaboration role: {self.source_role!r}")

    @property
    def fingerprint(self) -> str:
        return collaboration_claim_fingerprint(
            claim_key=self.claim_key,
            conclusion=self.conclusion,
            evidence_refs=self.evidence_refs,
            source_role=self.source_role,
            independence_key=self.independence_key,
        )


@dataclass(frozen=True, slots=True)
class CollaborationResult:
    result_id: str
    assignment_id: str
    role: PlanningRole
    status: CollaborationStatus
    summary: str
    evidence_refs: tuple[str, ...]
    confidence: float
    confidence_source: str = "legacy_unattributed"
    confidence_source_event_id: str | None = None
    child_run_id: str | None = None
    child_session_id: str | None = None
    claims: tuple[CollaborationClaim, ...] = ()
    files_changed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_evo_id(self.result_id, field="result_id")
        require_evo_id(self.assignment_id, field="assignment_id")
        if self.role not in _ROLES:
            raise CollaborationError(f"unknown collaboration role: {self.role!r}")
        if self.status not in _STATUSES:
            raise CollaborationError(f"unknown collaboration status: {self.status!r}")
        _text(self.summary, field="summary")
        _unique(self.evidence_refs, field="evidence_refs")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) or not 0 <= float(self.confidence) <= 1:
            raise CollaborationError("confidence must be between 0 and 1")
        _text(self.confidence_source, field="confidence_source")
        if self.confidence_source_event_id is not None:
            require_evo_id(self.confidence_source_event_id, field="confidence_source_event_id", kind="event")
        if self.child_run_id is not None:
            require_evo_id(self.child_run_id, field="child_run_id", kind="run")
        if self.child_session_id is not None:
            require_evo_id(self.child_session_id, field="child_session_id", kind="session")
        _unique(self.files_changed, field="files_changed")
        if any(claim.source_role != self.role for claim in self.claims):
            raise CollaborationError("claim source_role must match result role")

    def eligible_for_learning(self, contract: TaskContract, *, outcome_source_verified: bool = False) -> bool:
        return (
            self.status == "success"
            and bool(self.evidence_refs)
            and any(claim.evidence_refs for claim in self.claims)
            and self.confidence >= contract.minimum_confidence
            and self.role == contract.role
            and outcome_source_verified
        )


@dataclass(frozen=True, slots=True)
class RecordedCollaborationResult:
    contract: TaskContract
    result: CollaborationResult
    event_id: str | None
    eligible_for_learning: bool
    diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    claim_key: str
    conclusion: str
    group_key: str
    assignment_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    source_roles: tuple[PlanningRole, ...]


@dataclass(frozen=True, slots=True)
class CollaborationConflict:
    conflict_kind: ConflictKind
    assignment_ids: tuple[str, ...]
    branches: tuple[str, ...]
    resolution_state: str = "pending_verifier_or_curator"
    resource: str | None = None
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class CollaborationAggregate:
    collaboration_id: str
    child_run_ids: tuple[str, ...]
    result_event_ids: tuple[str, ...]
    eligible_result_ids: tuple[str, ...]
    evidence_groups: tuple[EvidenceGroup, ...]
    conflicts: tuple[CollaborationConflict, ...]
    event_id: str | None
    diagnostic: str | None = None

    @property
    def independent_support_count(self) -> int:
        return len(self.evidence_groups)


@dataclass(frozen=True, slots=True)
class DelegationRecord:
    collaboration_id: str
    assignment_id: str
    event_id: str | None
    diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class CollaborationProjectionDiagnostic:
    code: str
    message: str
    event_id: str | None = None


@dataclass(frozen=True, slots=True)
class CollaborationProjection:
    """Replay result derived solely from append-only collaboration facts."""

    collaboration_id: str
    contracts: Mapping[str, TaskContract]
    results: tuple[RecordedCollaborationResult, ...]
    evidence_groups: tuple[EvidenceGroup, ...]
    conflicts: tuple[CollaborationConflict, ...]
    diagnostics: tuple[CollaborationProjectionDiagnostic, ...] = ()


class CollaborationService:
    """Append collaboration facts and build deterministic evidence aggregates."""

    def __init__(self, *, store: EvoEventStore, parent_run_id: str, session_id: str | None = None) -> None:
        self._store = store
        self._parent_run_id = require_evo_id(parent_run_id, field="parent_run_id", kind="run")
        self._session_id = None if session_id is None else require_evo_id(session_id, field="session_id", kind="session")

    def delegate(
        self,
        *,
        collaboration_id: str,
        assignment_id: str,
        contract: TaskContract,
        runtime_role: str,
    ) -> DelegationRecord:
        require_evo_id(collaboration_id, field="collaboration_id")
        require_evo_id(assignment_id, field="assignment_id")
        _text(runtime_role, field="runtime_role")
        contract = _normalized_contract(contract)
        payload = CollaborationTaskDelegatedPayload(
            collaboration_id=collaboration_id,
            assignment_id=assignment_id,
            runtime_role=runtime_role,
            contract=_redacted_contract(contract),
        )
        event = self._event(
            "CollaborationTaskDelegated",
            payload,
            plan_id=contract.plan_id,
            node_id=contract.node_id,
            extensions={"collaboration_id": collaboration_id, "assignment_id": assignment_id},
        )
        try:
            persisted = self._store.append(event).event
        except EvoStoreError as error:
            return DelegationRecord(collaboration_id, assignment_id, None, f"collaboration_delegation_recording_failed: {error}")
        except Exception as error:  # noqa: BLE001 - recorder failure must remain a structured diagnostic
            return DelegationRecord(collaboration_id, assignment_id, None, f"collaboration_delegation_recording_failed: unexpected recorder failure: {error}")
        return DelegationRecord(collaboration_id, assignment_id, persisted.event_id)

    def record_result(
        self,
        *,
        collaboration_id: str,
        contract: TaskContract,
        result: CollaborationResult,
    ) -> RecordedCollaborationResult:
        require_evo_id(collaboration_id, field="collaboration_id")
        contract = _normalized_contract(contract)
        _validate_result_claims(result)
        try:
            events = self._store.list_events()
            source_verified, gate_diagnostic = _verify_outcome_source(
                result,
                events,
                contract=contract,
            )
            delegation_verified = _has_matching_delegation(
                events,
                collaboration_id=collaboration_id,
                assignment_id=result.assignment_id,
                contract=contract,
                parent_run_id=self._parent_run_id,
            )
            if not delegation_verified and source_verified:
                gate_diagnostic = "result has no matching prior delegation contract"
        except Exception as error:  # noqa: BLE001 - result recording must preserve the child execution result
            source_verified = False
            delegation_verified = False
            gate_diagnostic = f"child Outcome source could not be verified: {error}"
        eligible = result.eligible_for_learning(
            contract,
            outcome_source_verified=source_verified and delegation_verified,
        )
        persisted_claims = tuple(_redacted_claim(claim) for claim in result.claims)
        payload = CollaborationTaskResultRecordedPayload(
            collaboration_id=collaboration_id,
            assignment_id=result.assignment_id,
            status=result.status,
            summary=redact_text(result.summary)[0],
            evidence_refs=result.evidence_refs,
            confidence=float(result.confidence),
            eligible_for_learning=eligible,
            result_id=result.result_id,
            child_run_id=result.child_run_id,
            child_session_id=result.child_session_id,
            claims=persisted_claims,
            claim_fingerprints=tuple(claim.fingerprint for claim in persisted_claims),
            claim_format="v2_full",
            confidence_source=result.confidence_source,
            confidence_source_event_id=result.confidence_source_event_id,
            files_changed=result.files_changed,
        )
        event = self._event(
            "CollaborationTaskResultRecorded",
            payload,
            plan_id=contract.plan_id,
            node_id=contract.node_id,
            extensions={
                "collaboration_id": collaboration_id,
                "assignment_id": result.assignment_id,
                **({"child_run_id": result.child_run_id} if result.child_run_id else {}),
            },
        )
        try:
            persisted = self._store.append(event).event
        except EvoStoreError as error:
            return RecordedCollaborationResult(contract, result, None, False, f"collaboration_result_recording_failed: {error}")
        except Exception as error:  # noqa: BLE001 - preserve the child execution result
            return RecordedCollaborationResult(
                contract,
                result,
                None,
                False,
                f"collaboration_result_recording_failed: unexpected recorder failure: {error}",
            )
        diagnostic = None if eligible or gate_diagnostic is None else f"collaboration_learning_ineligible: {gate_diagnostic}"
        return RecordedCollaborationResult(contract, result, persisted.event_id, eligible, diagnostic)

    def child_outcome(self, child_run_id: str) -> EvoEvent[OutcomeClassifiedPayload] | None:
        """Return the latest canonical Outcome fact for a child Run."""

        require_evo_id(child_run_id, field="child_run_id", kind="run")
        events = self._store.list_events()
        for event in reversed(events):
            if event.event_type == "OutcomeClassified" and event.refs.run_id == child_run_id and isinstance(event.payload, OutcomeClassifiedPayload):
                try:
                    attest_outcome_event(events, event)
                except (OutcomeIntegrityError, ValueError):
                    continue
                return cast(EvoEvent[OutcomeClassifiedPayload], event)
        return None

    def verified_child_evidence(self, child_run_id: str, evidence_refs: Sequence[str]) -> tuple[str, ...]:
        """Filter an Outcome's refs to deterministic successful child evidence."""

        require_evo_id(child_run_id, field="child_run_id", kind="run")
        events = self._store.list_events()
        verified: list[str] = []
        for evidence_ref in evidence_refs:
            try:
                records = resolve_evidence_records(
                    events,
                    (evidence_ref,),
                    run_id=child_run_id,
                    require_verified=True,
                    deterministic_only=True,
                    require_exit_code=True,
                )
            except (EvidenceIntegrityError, ValueError):
                continue
            if records[0].payload.exit_code == 0:
                verified.append(evidence_ref)
        return tuple(verified)

    def rebuild(self, collaboration_id: str) -> CollaborationProjection:
        """Replay contracts, results, groups, and conflicts from the source."""

        return rebuild_collaboration(
            self._store.list_events(),
            collaboration_id=collaboration_id,
            parent_run_id=self._parent_run_id,
        )

    def resource_conflicts(
        self,
        *,
        collaboration_id: str,
        assignments: Mapping[str, TaskContract],
    ) -> tuple[CollaborationConflict, ...]:
        assignments = {assignment_id: _normalized_contract(contract) for assignment_id, contract in assignments.items()}
        conflicts: list[CollaborationConflict] = []
        items = list(assignments.items())
        for index, (left_id, left) in enumerate(items):
            for right_id, right in items[index + 1 :]:
                resource = conflicting_resource(left.resource_claims, right.resource_claims)
                if resource is None:
                    continue
                conflict = CollaborationConflict("resource", (left_id, right_id), (), resource=resource)
                conflicts.append(self._record_conflict(collaboration_id, conflict))
        return tuple(conflicts)

    def aggregate(
        self,
        *,
        collaboration_id: str,
        records: Sequence[RecordedCollaborationResult],
        resource_conflicts: Sequence[CollaborationConflict] = (),
    ) -> CollaborationAggregate:
        require_evo_id(collaboration_id, field="collaboration_id")
        events = self._store.list_events()
        projection = rebuild_collaboration(
            events,
            collaboration_id=collaboration_id,
            parent_run_id=self._parent_run_id,
        )
        canonical_by_event_id = {record.event_id: record for record in projection.results if record.event_id is not None}
        validated_records = tuple(
            canonical_by_event_id[event_id] for event_id in dict.fromkeys(record.event_id for record in records if record.event_id is not None) if event_id in canonical_by_event_id
        )
        groups = _evidence_groups(validated_records)
        conclusion_conflicts = tuple(self._record_conflict(collaboration_id, conflict) for conflict in _conclusion_conflicts(validated_records))
        persisted_resource_conflicts = tuple(
            conflict
            for conflict in resource_conflicts
            if conflict.event_id is not None
            and any(event.event_id == conflict.event_id and event.refs.run_id == self._parent_run_id and event.event_type == "CollaborationConflictDetected" for event in events)
        )
        conflicts = persisted_resource_conflicts + conclusion_conflicts
        result_event_ids = tuple(record.event_id for record in validated_records if record.event_id is not None)
        eligible_result_ids = tuple(record.result.result_id for record in validated_records if record.eligible_for_learning and record.event_id is not None)
        child_run_ids = tuple(dict.fromkeys(record.result.child_run_id for record in validated_records if record.result.child_run_id))
        conflict_event_ids = tuple(conflict.event_id for conflict in conflicts if conflict.event_id is not None)
        payload = CollaborationRunAggregatedPayload(
            collaboration_id=collaboration_id,
            child_run_ids=child_run_ids,
            result_event_ids=result_event_ids,
            conflict_event_ids=conflict_event_ids,
            evidence_group_count=len(groups),
            independent_support_count=len(groups),
            eligible_result_ids=eligible_result_ids,
        )
        event = self._event(
            "CollaborationRunAggregated",
            payload,
            extensions={"collaboration_id": collaboration_id},
        )
        existing = _matching_aggregate_event(
            events,
            payload,
            parent_run_id=self._parent_run_id,
        )
        if existing is not None:
            return CollaborationAggregate(
                collaboration_id,
                child_run_ids,
                result_event_ids,
                eligible_result_ids,
                groups,
                conflicts,
                existing.event_id,
            )
        try:
            persisted = self._store.append(event).event
        except EvoStoreError as error:
            return CollaborationAggregate(
                collaboration_id,
                child_run_ids,
                result_event_ids,
                eligible_result_ids,
                groups,
                conflicts,
                None,
                f"collaboration_aggregate_recording_failed: {error}",
            )
        except Exception as error:  # noqa: BLE001 - aggregation remains inspectable in memory
            return CollaborationAggregate(
                collaboration_id,
                child_run_ids,
                result_event_ids,
                eligible_result_ids,
                groups,
                conflicts,
                None,
                f"collaboration_aggregate_recording_failed: unexpected recorder failure: {error}",
            )
        return CollaborationAggregate(
            collaboration_id,
            child_run_ids,
            result_event_ids,
            eligible_result_ids,
            groups,
            conflicts,
            persisted.event_id,
        )

    def _record_conflict(self, collaboration_id: str, conflict: CollaborationConflict) -> CollaborationConflict:
        payload = CollaborationConflictDetectedPayload(
            collaboration_id=collaboration_id,
            conflict_kind=conflict.conflict_kind,
            assignment_ids=conflict.assignment_ids,
            branches=tuple(redact_text(branch)[0] for branch in conflict.branches),
            resolution_state=conflict.resolution_state,
            resource=conflict.resource,
        )
        event = self._event(
            "CollaborationConflictDetected",
            payload,
            extensions={"collaboration_id": collaboration_id},
        )
        try:
            existing = _matching_conflict_event(
                self._store.list_events(),
                payload,
                parent_run_id=self._parent_run_id,
            )
        except Exception:  # noqa: BLE001 - a replay failure must not hide the in-memory conflict
            existing = None
        if existing is not None:
            return replace(conflict, event_id=existing.event_id)
        try:
            persisted = self._store.append(event).event
        except Exception:  # noqa: BLE001 - conflict remains returned even if recorder is unavailable
            return conflict
        return replace(conflict, event_id=persisted.event_id)

    def _event(
        self,
        event_type: str,
        payload: object,
        *,
        plan_id: str | None = None,
        node_id: str | None = None,
        extensions: dict[str, object] | None = None,
    ) -> EvoEvent:
        return EvoEvent(
            event_id=new_evo_id("event"),
            event_type=event_type,
            refs=EvoReferences(
                run_id=self._parent_run_id,
                session_id=self._session_id,
                plan_id=plan_id,
                node_id=node_id,
                extensions=extensions or {},
            ),
            payload=payload,  # type: ignore[arg-type]
        )


def _validate_result_claims(result: CollaborationResult) -> None:
    result_evidence = set(result.evidence_refs)
    for claim in result.claims:
        if claim.source_role != result.role:
            raise CollaborationError("claim source_role must match result role")
        missing = set(claim.evidence_refs).difference(result_evidence)
        if missing:
            raise CollaborationError(f"claim evidence_refs must be a subset of result evidence_refs: {sorted(missing)[0]}")
        expected = collaboration_claim_fingerprint(
            claim_key=claim.claim_key,
            conclusion=claim.conclusion,
            evidence_refs=claim.evidence_refs,
            source_role=claim.source_role,
            independence_key=claim.independence_key,
        )
        if claim.fingerprint != expected:
            raise CollaborationError("claim fingerprint does not match the complete Claim")


def _redacted_claim(claim: CollaborationClaim) -> CollaborationClaimPayload:
    claim_key = redact_text(claim.claim_key)[0]
    conclusion = redact_text(claim.conclusion)[0]
    independence_key = redact_text(claim.independence_key)[0]
    fingerprint = collaboration_claim_fingerprint(
        claim_key=claim_key,
        conclusion=conclusion,
        evidence_refs=claim.evidence_refs,
        source_role=claim.source_role,
        independence_key=independence_key,
    )
    return CollaborationClaimPayload(
        claim_key=claim_key,
        conclusion=conclusion,
        evidence_refs=claim.evidence_refs,
        source_role=claim.source_role,
        independence_key=independence_key,
        fingerprint=fingerprint,
    )


def _verify_outcome_source(
    result: CollaborationResult,
    events: Sequence[EvoEvent],
    *,
    contract: TaskContract,
    before_sequence: int | None = None,
) -> tuple[bool, str | None]:
    if result.confidence_source != "outcome_event":
        return False, "confidence is not attributed to a child Outcome event"
    if result.confidence_source_event_id is None or result.child_run_id is None:
        return False, "confidence source requires both child_run_id and confidence_source_event_id"
    matches = [event for event in events if event.event_id == result.confidence_source_event_id]
    if len(matches) != 1:
        return False, "confidence source Outcome event is missing or ambiguous"
    event = matches[0]
    if before_sequence is not None and event.sequence is not None and event.sequence >= before_sequence:
        return False, "confidence source Outcome must precede the result fact"
    if event.event_type != "OutcomeClassified" or not isinstance(event.payload, OutcomeClassifiedPayload):
        return False, "confidence source is not an OutcomeClassified event"
    if event.refs.run_id != result.child_run_id:
        return False, "confidence source Outcome belongs to a different child Run"
    try:
        attest_outcome_event(tuple(events), event)
    except (OutcomeIntegrityError, ValueError) as error:
        return False, f"confidence source Outcome is not canonical: {error}"
    if event.payload.outcome != "success":
        return False, "child Outcome is not successful"
    if abs(event.payload.confidence - float(result.confidence)) > 1e-12:
        return False, "result confidence does not match the child Outcome"
    if not set(result.evidence_refs).issubset(event.payload.evidence_refs):
        return False, "result evidence_refs are not backed by the child Outcome"
    try:
        evidence = resolve_evidence_records(
            events,
            result.evidence_refs,
            run_id=result.child_run_id,
            require_verified=True,
            deterministic_only=True,
            require_exit_code=True,
            before_sequence=event.sequence,
        )
    except (EvidenceIntegrityError, ValueError) as error:
        return False, f"result verification Evidence is invalid: {error}"
    if any(record.payload.exit_code != 0 for record in evidence):
        return False, "successful collaboration results require zero-exit verification Evidence"
    actual_types = {record.evidence_type for record in evidence}
    missing_types = set(contract.expected_evidence).difference(actual_types)
    if missing_types:
        return (
            False,
            "result does not cover expected Evidence types: " + ", ".join(sorted(missing_types)),
        )
    return True, None


def _matching_result_event(
    events: Sequence[EvoEvent],
    *,
    event_id: str | None,
    collaboration_id: str,
    parent_run_id: str,
    result: CollaborationResult,
) -> EvoEvent[CollaborationTaskResultRecordedPayload] | None:
    if event_id is None:
        return None
    for event in events:
        payload = event.payload
        if (
            event.event_id == event_id
            and event.event_type == "CollaborationTaskResultRecorded"
            and isinstance(payload, CollaborationTaskResultRecordedPayload)
            and payload.collaboration_id == collaboration_id
            and event.refs.run_id == parent_run_id
            and payload.assignment_id == result.assignment_id
            and payload.claims_rebuildable
            and payload.result_id == result.result_id
        ):
            expected_claims = tuple(_redacted_claim(claim) for claim in result.claims)
            if payload.claims == expected_claims:
                return cast(EvoEvent[CollaborationTaskResultRecordedPayload], event)
    return None


def _has_matching_delegation(
    events: Sequence[EvoEvent],
    *,
    collaboration_id: str,
    assignment_id: str,
    contract: TaskContract,
    parent_run_id: str,
    before_sequence: int | None = None,
) -> bool:
    expected_contract = _redacted_contract(contract)
    return any(
        event.event_type == "CollaborationTaskDelegated"
        and isinstance(event.payload, CollaborationTaskDelegatedPayload)
        and event.payload.collaboration_id == collaboration_id
        and event.refs.run_id == parent_run_id
        and event.payload.assignment_id == assignment_id
        and event.payload.contract == expected_contract
        and (before_sequence is None or event.sequence is None or event.sequence < before_sequence)
        for event in events
    )


def _revalidate_record(
    record: RecordedCollaborationResult,
    *,
    collaboration_id: str,
    parent_run_id: str,
    events: Sequence[EvoEvent],
) -> RecordedCollaborationResult:
    _validate_result_claims(record.result)
    result_event = _matching_result_event(
        events,
        event_id=record.event_id,
        collaboration_id=collaboration_id,
        parent_run_id=parent_run_id,
        result=record.result,
    )
    if result_event is None:
        return replace(
            record,
            eligible_for_learning=False,
            diagnostic="collaboration_learning_ineligible: result fact is missing or not replayable",
        )
    if not _has_matching_delegation(
        events,
        collaboration_id=collaboration_id,
        assignment_id=record.result.assignment_id,
        contract=record.contract,
        parent_run_id=parent_run_id,
        before_sequence=result_event.sequence,
    ):
        return replace(
            record,
            eligible_for_learning=False,
            diagnostic="collaboration_learning_ineligible: matching delegation fact is missing",
        )
    source_verified, reason = _verify_outcome_source(
        record.result,
        events,
        contract=record.contract,
        before_sequence=result_event.sequence,
    )
    eligible = record.result.eligible_for_learning(
        record.contract,
        outcome_source_verified=source_verified,
    )
    return replace(
        record,
        eligible_for_learning=eligible,
        diagnostic=None if eligible else f"collaboration_learning_ineligible: {reason or 'result gate failed'}",
    )


def rebuild_collaboration(
    events: Sequence[EvoEvent],
    *,
    collaboration_id: str,
    parent_run_id: str,
) -> CollaborationProjection:
    """Deterministically replay collaboration facts without trusting aggregate summaries."""

    require_evo_id(collaboration_id, field="collaboration_id")
    require_evo_id(parent_run_id, field="parent_run_id", kind="run")
    all_ordered = tuple(sorted(events, key=lambda event: event.sequence or 0))
    ordered = tuple(event for event in all_ordered if event.refs.run_id == parent_run_id)
    contracts: dict[str, TaskContract] = {}
    delegation_sequences: dict[str, int] = {}
    diagnostics: list[CollaborationProjectionDiagnostic] = []
    for event in all_ordered:
        if event.refs.run_id != parent_run_id and getattr(event.payload, "collaboration_id", None) == collaboration_id:
            diagnostics.append(
                CollaborationProjectionDiagnostic(
                    "cross_run_fact_ignored",
                    "collaboration fact belongs to a different parent Run",
                    event.event_id,
                )
            )

    for event in ordered:
        payload = event.payload
        if event.event_type != "CollaborationTaskDelegated" or not isinstance(payload, CollaborationTaskDelegatedPayload) or payload.collaboration_id != collaboration_id:
            continue
        try:
            contract = _normalized_contract(TaskContract.from_dict(payload.contract))
        except Exception as error:  # noqa: BLE001 - malformed facts remain diagnosable
            diagnostics.append(
                CollaborationProjectionDiagnostic(
                    "invalid_delegation_contract",
                    f"cannot rebuild assignment {payload.assignment_id}: {error}",
                    event.event_id,
                )
            )
            continue
        existing = contracts.get(payload.assignment_id)
        if existing is not None and existing != contract:
            diagnostics.append(
                CollaborationProjectionDiagnostic(
                    "conflicting_delegation",
                    f"assignment {payload.assignment_id} has conflicting contracts; the first fact is retained",
                    event.event_id,
                )
            )
            continue
        contracts.setdefault(payload.assignment_id, contract)
        delegation_sequences.setdefault(payload.assignment_id, event.sequence or 0)

    results: list[RecordedCollaborationResult] = []
    for event in ordered:
        payload = event.payload
        if event.event_type != "CollaborationTaskResultRecorded" or not isinstance(payload, CollaborationTaskResultRecordedPayload) or payload.collaboration_id != collaboration_id:
            continue
        contract = contracts.get(payload.assignment_id)
        if contract is None or delegation_sequences.get(payload.assignment_id, 0) >= (event.sequence or 0):
            diagnostics.append(
                CollaborationProjectionDiagnostic(
                    "result_without_prior_contract",
                    f"result for assignment {payload.assignment_id} has no valid prior delegation",
                    event.event_id,
                )
            )
            continue
        if not payload.claims_rebuildable:
            diagnostics.append(
                CollaborationProjectionDiagnostic(
                    "legacy_fingerprint_only_claims",
                    f"result for assignment {payload.assignment_id} contains only v1 Claim fingerprints",
                    event.event_id,
                )
            )
        try:
            claims = tuple(
                CollaborationClaim(
                    claim_key=claim.claim_key,
                    conclusion=claim.conclusion,
                    evidence_refs=claim.evidence_refs,
                    source_role=cast(PlanningRole, claim.source_role),
                    independence_key=claim.independence_key,
                )
                for claim in payload.claims
            )
            result = CollaborationResult(
                result_id=payload.result_id or f"result_{event.event_id}",
                assignment_id=payload.assignment_id,
                role=contract.role,
                status=cast(CollaborationStatus, payload.status),
                summary=payload.summary,
                evidence_refs=payload.evidence_refs,
                confidence=payload.confidence,
                confidence_source=payload.confidence_source,
                confidence_source_event_id=payload.confidence_source_event_id,
                child_run_id=payload.child_run_id,
                child_session_id=payload.child_session_id,
                claims=claims,
                files_changed=payload.files_changed,
            )
            _validate_result_claims(result)
        except Exception as error:  # noqa: BLE001 - one bad result must not hide other assignments
            diagnostics.append(
                CollaborationProjectionDiagnostic(
                    "invalid_result_fact",
                    f"cannot rebuild assignment {payload.assignment_id} result: {error}",
                    event.event_id,
                )
            )
            continue
        source_verified, reason = _verify_outcome_source(
            result,
            all_ordered,
            contract=contract,
            before_sequence=event.sequence,
        )
        eligible = payload.claims_rebuildable and result.eligible_for_learning(contract, outcome_source_verified=source_verified)
        result_diagnostic = None
        if not eligible:
            result_diagnostic = f"collaboration_learning_ineligible: {reason or 'result gate failed'}"
        results.append(
            RecordedCollaborationResult(
                contract=contract,
                result=result,
                event_id=event.event_id,
                eligible_for_learning=eligible,
                diagnostic=result_diagnostic,
            )
        )

    conflicts: list[CollaborationConflict] = []
    for event in ordered:
        payload = event.payload
        if event.event_type == "CollaborationConflictDetected" and isinstance(payload, CollaborationConflictDetectedPayload) and payload.collaboration_id == collaboration_id:
            conflicts.append(
                CollaborationConflict(
                    conflict_kind=cast(ConflictKind, payload.conflict_kind),
                    assignment_ids=payload.assignment_ids,
                    branches=payload.branches,
                    resolution_state=payload.resolution_state,
                    resource=payload.resource,
                    event_id=event.event_id,
                )
            )

    replayed_results = tuple(results)
    return CollaborationProjection(
        collaboration_id=collaboration_id,
        contracts=dict(contracts),
        results=replayed_results,
        evidence_groups=_evidence_groups(replayed_results),
        conflicts=tuple(conflicts),
        diagnostics=tuple(diagnostics),
    )


def _matching_conflict_event(
    events: Sequence[EvoEvent],
    payload: CollaborationConflictDetectedPayload,
    *,
    parent_run_id: str,
) -> EvoEvent[CollaborationConflictDetectedPayload] | None:
    for event in events:
        if (
            event.refs.run_id == parent_run_id
            and event.event_type == "CollaborationConflictDetected"
            and isinstance(event.payload, CollaborationConflictDetectedPayload)
            and event.payload.to_dict() == payload.to_dict()
        ):
            return cast(EvoEvent[CollaborationConflictDetectedPayload], event)
    return None


def _matching_aggregate_event(
    events: Sequence[EvoEvent],
    payload: CollaborationRunAggregatedPayload,
    *,
    parent_run_id: str,
) -> EvoEvent[CollaborationRunAggregatedPayload] | None:
    for event in events:
        if (
            event.refs.run_id == parent_run_id
            and event.event_type == "CollaborationRunAggregated"
            and isinstance(event.payload, CollaborationRunAggregatedPayload)
            and event.payload.to_dict() == payload.to_dict()
        ):
            return cast(EvoEvent[CollaborationRunAggregatedPayload], event)
    return None


def _claim_resource(claim: str) -> tuple[str, str]:
    normalized = _normalize_resource_claim(claim)
    mode, _, resource = normalized.partition(":")
    return mode, resource


def _normalize_resource_claim(claim: str) -> str:
    value = _text(claim, field="resource_claim")
    mode, separator, raw_resource = value.partition(":")
    mode = mode.strip().lower()
    if separator != ":" or mode not in {"read", "write"}:
        raise CollaborationError("resource claim must use read:<relative-path> or write:<relative-path>")
    resource = raw_resource.replace("\\", "/")
    if not resource or resource.startswith("/") or "//" in resource or (len(resource) >= 2 and resource[1] == ":"):
        raise CollaborationError("resource path must be an unambiguous relative path")
    if any(segment in {"", ".", ".."} for segment in resource.split("/")):
        raise CollaborationError("resource path must not contain '.', '..', or empty segments")
    candidate = PurePosixPath(resource)
    canonical = candidate.as_posix().casefold()
    if canonical in {"", ".", ".."}:
        raise CollaborationError("resource path must identify a file or directory")
    return f"{mode}:{canonical}"


def _normalize_expected_evidence(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    aliases = (
        ("test", ("test", "pytest", "unittest", "tool result")),
        ("lint", ("lint", "ruff", "flake8", "eslint", "pylint")),
        ("type_check", ("type_check", "type-check", "typecheck", "mypy", "pyright", "tsc")),
        ("build", ("build",)),
        ("diff", ("diff", "patch")),
    )
    for value in values:
        text = _text(value, field="expected_evidence[]").casefold()
        matches = [evidence_type for evidence_type, markers in aliases if any(marker in text for marker in markers)]
        if not matches:
            raise CollaborationError(f"expected_evidence is not a trusted deterministic type: {value!r}")
        for evidence_type in matches:
            if evidence_type not in normalized:
                normalized.append(evidence_type)
    return tuple(normalized)


def _normalized_contract(contract: TaskContract) -> TaskContract:
    return replace(
        contract,
        expected_evidence=_normalize_expected_evidence(contract.expected_evidence),
        resource_claims=tuple(_normalize_resource_claim(claim) for claim in contract.resource_claims),
    )


def _redacted_contract(contract: TaskContract) -> dict[str, object]:
    contract = _normalized_contract(contract)
    result = contract.to_dict()
    result["goal"] = redact_text(contract.goal)[0]
    result["input_snapshot"] = redact_text(contract.input_snapshot)[0]
    return result


def _overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def conflicting_resource(left: Sequence[str], right: Sequence[str]) -> str | None:
    for left_claim in left:
        left_mode, left_resource = _claim_resource(left_claim)
        for right_claim in right:
            right_mode, right_resource = _claim_resource(right_claim)
            if "write" in {left_mode, right_mode} and _overlap(left_resource, right_resource):
                return left_resource if len(left_resource) <= len(right_resource) else right_resource
    return None


def _evidence_groups(records: Sequence[RecordedCollaborationResult]) -> tuple[EvidenceGroup, ...]:
    buckets: dict[tuple[str, str], list[tuple[RecordedCollaborationResult, CollaborationClaim]]] = {}
    for record in records:
        if not record.eligible_for_learning or record.event_id is None:
            continue
        for claim in record.result.claims:
            if not claim.evidence_refs:
                continue
            key = (claim.claim_key, claim.conclusion)
            buckets.setdefault(key, []).append((record, claim))
    groups: list[EvidenceGroup] = []
    for (claim_key, conclusion), members in sorted(buckets.items()):
        for cluster in _independent_clusters(members):
            material = "\0".join(sorted(claim.fingerprint for _, claim in cluster))
            groups.append(
                EvidenceGroup(
                    claim_key=claim_key,
                    conclusion=conclusion,
                    group_key=sha256(material.encode("utf-8")).hexdigest(),
                    assignment_ids=tuple(dict.fromkeys(record.result.assignment_id for record, _ in cluster)),
                    evidence_refs=tuple(dict.fromkeys(ref for _, claim in cluster for ref in claim.evidence_refs)),
                    source_roles=tuple(dict.fromkeys(claim.source_role for _, claim in cluster)),
                )
            )
    return tuple(groups)


def _independent_clusters(
    members: Sequence[tuple[RecordedCollaborationResult, CollaborationClaim]],
) -> tuple[tuple[tuple[RecordedCollaborationResult, CollaborationClaim], ...], ...]:
    """Collapse results that share provenance or any evidence reference."""

    parent = list(range(len(members)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, (left_record, left) in enumerate(members):
        left_refs = set(left.evidence_refs)
        for right_index in range(left_index + 1, len(members)):
            right_record, right = members[right_index]
            same_child_run = left_record.result.child_run_id is not None and left_record.result.child_run_id == right_record.result.child_run_id
            if same_child_run or left_refs.intersection(right.evidence_refs):
                union(left_index, right_index)
    clusters: dict[int, list[tuple[RecordedCollaborationResult, CollaborationClaim]]] = {}
    for index, member in enumerate(members):
        clusters.setdefault(find(index), []).append(member)
    return tuple(tuple(cluster) for _, cluster in sorted(clusters.items()))


def _conclusion_conflicts(records: Sequence[RecordedCollaborationResult]) -> tuple[CollaborationConflict, ...]:
    by_claim: dict[str, dict[str, list[str]]] = {}
    for record in records:
        if record.event_id is None:
            continue
        for claim in record.result.claims:
            by_claim.setdefault(claim.claim_key, {}).setdefault(claim.conclusion, []).append(record.result.assignment_id)
    conflicts: list[CollaborationConflict] = []
    for claim_key, branches in sorted(by_claim.items()):
        if len(branches) < 2:
            continue
        assignment_ids = tuple(dict.fromkeys(assignment for values in branches.values() for assignment in values))
        branch_labels = tuple(f"{claim_key}: {conclusion}" for conclusion in sorted(branches))
        conflicts.append(CollaborationConflict("conclusion", assignment_ids, branch_labels))
    return tuple(conflicts)
