"""Evidence-governed collaboration contracts for P10.

This module is deliberately runtime-free.  It records parent/child facts,
detects resource and conclusion conflicts, and groups independent evidence.  It
does not create an AgentLoop, execute a tool, or make a permission decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal, Mapping, Sequence

from bauhinia_agent.evolution.events import (
    CollaborationConflictDetectedPayload,
    CollaborationRunAggregatedPayload,
    CollaborationTaskDelegatedPayload,
    CollaborationTaskResultRecordedPayload,
    EvoEvent,
    EvoReferences,
)
from bauhinia_agent.evolution.evidence import redact_text
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError
from bauhinia_agent.planning.evo import PlanningRole, TaskContract

CollaborationStatus = Literal["success", "failure", "cancelled", "timeout", "permission_denied"]
ConflictKind = Literal["resource", "conclusion"]

_STATUSES = frozenset({"success", "failure", "cancelled", "timeout", "permission_denied"})
_ROLES = frozenset({"planner", "researcher", "executor", "verifier", "critic", "curator"})


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
        material = "\0".join((self.claim_key, self.conclusion, *sorted(self.evidence_refs)))
        return sha256(material.encode("utf-8")).hexdigest()

@dataclass(frozen=True, slots=True)
class CollaborationResult:
    result_id: str
    assignment_id: str
    role: PlanningRole
    status: CollaborationStatus
    summary: str
    evidence_refs: tuple[str, ...]
    confidence: float
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
        if self.child_run_id is not None:
            require_evo_id(self.child_run_id, field="child_run_id", kind="run")
        if self.child_session_id is not None:
            require_evo_id(self.child_session_id, field="child_session_id", kind="session")
        _unique(self.files_changed, field="files_changed")
        if any(claim.source_role != self.role for claim in self.claims):
            raise CollaborationError("claim source_role must match result role")

    def eligible_for_learning(self, contract: TaskContract) -> bool:
        return self.status == "success" and bool(self.evidence_refs) and self.confidence >= contract.minimum_confidence and self.role == contract.role


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
        eligible = result.eligible_for_learning(contract)
        payload = CollaborationTaskResultRecordedPayload(
            collaboration_id=collaboration_id,
            assignment_id=result.assignment_id,
            status=result.status,
            summary=redact_text(result.summary)[0],
            evidence_refs=result.evidence_refs,
            confidence=float(result.confidence),
            eligible_for_learning=eligible,
            child_run_id=result.child_run_id,
            child_session_id=result.child_session_id,
            claim_fingerprints=tuple(claim.fingerprint for claim in result.claims),
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
        return RecordedCollaborationResult(contract, result, persisted.event_id, eligible)

    def resource_conflicts(
        self,
        *,
        collaboration_id: str,
        assignments: Mapping[str, TaskContract],
    ) -> tuple[CollaborationConflict, ...]:
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
        groups = _evidence_groups(records)
        conclusion_conflicts = tuple(self._record_conflict(collaboration_id, conflict) for conflict in _conclusion_conflicts(records))
        conflicts = tuple(resource_conflicts) + conclusion_conflicts
        result_event_ids = tuple(record.event_id for record in records if record.event_id is not None)
        eligible_result_ids = tuple(record.result.result_id for record in records if record.eligible_for_learning and record.event_id is not None)
        child_run_ids = tuple(dict.fromkeys(record.result.child_run_id for record in records if record.result.child_run_id))
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


def _claim_resource(claim: str) -> tuple[str, str]:
    mode, _, resource = claim.partition(":")
    normalized = resource.replace("\\", "/").rstrip("/").casefold()
    return mode, normalized


def _redacted_contract(contract: TaskContract) -> dict[str, object]:
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

    for left_index, (_, left) in enumerate(members):
        left_refs = set(left.evidence_refs)
        for right_index in range(left_index + 1, len(members)):
            right = members[right_index][1]
            if left.independence_key == right.independence_key or left_refs.intersection(right.evidence_refs):
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
