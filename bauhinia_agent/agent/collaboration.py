"""Runtime adapter from P10 collaboration contracts to existing subagents.

The adapter intentionally delegates scheduling to ``BackgroundJobManager`` and
execution to ``SubagentRunner``.  It does not create another agent loop,
permission engine, tool registry, or thread pool.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import Lock, Timer
from typing import Mapping

from bauhinia_agent.agent.background import (
    STATUS_CANCELLED,
    STATUS_RUNNING,
    BackgroundCapacityError,
    BackgroundJobManager,
)
from bauhinia_agent.agent.subagent import SubagentRequest, SubagentResult, SubagentRole, SubagentRunner
from bauhinia_agent.evolution.collaboration import (
    CollaborationAggregate,
    CollaborationClaim,
    CollaborationConflict,
    CollaborationError,
    CollaborationResult,
    CollaborationService,
    RecordedCollaborationResult,
    conflicting_resource,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.planning.evo import PlanningRole, TaskContract
from bauhinia_agent.runtime.cancellation import current_cancellation_token
from bauhinia_agent.tools.types import ToolResult

ROLE_RUNTIME_PROFILES: dict[PlanningRole, SubagentRole] = {
    "planner": "researcher",
    "researcher": "researcher",
    "executor": "coder",
    "verifier": "tester",
    "critic": "reviewer",
    "curator": "reviewer",
}
_KNOWN_EFFECTS = frozenset({"read", "write", "execute", "network", "external"})


@dataclass(frozen=True, slots=True)
class RuntimeCollaborationOutcome:
    collaboration_id: str
    assignment_id: str
    child_run_id: str | None
    recorded: RecordedCollaborationResult
    job_id: str | None = None


@dataclass(frozen=True, slots=True)
class CollaborationDispatchBatch:
    collaboration_id: str
    job_ids: dict[str, str]
    blocked_assignment_ids: tuple[str, ...]
    resource_conflicts: tuple[CollaborationConflict, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeBatchState:
    assignment_ids: tuple[str, ...]
    resource_conflicts: tuple[CollaborationConflict, ...] = ()


class CollaborationRuntimeAdapter:
    """Safely adapt six domain roles onto the four existing runtime profiles."""

    def __init__(
        self,
        *,
        runner: SubagentRunner,
        background_manager: BackgroundJobManager,
        service: CollaborationService,
        parent_session_id: str,
        parent_run_id: str,
    ) -> None:
        self._runner = runner
        self._background = background_manager
        self._service = service
        self._parent_session_id = require_evo_id(parent_session_id, field="parent_session_id", kind="session")
        self._parent_run_id = require_evo_id(parent_run_id, field="parent_run_id", kind="run")
        self._lock = Lock()
        self._dispatch_lock = Lock()
        self._outcomes: dict[str, RuntimeCollaborationOutcome] = {}
        self._contracts: dict[str, tuple[str, TaskContract, str]] = {}
        self._job_ids: dict[str, str] = {}
        self._batches: dict[str, _RuntimeBatchState] = {}
        self._aggregating: set[str] = set()
        self._aggregates: dict[str, CollaborationAggregate] = {}
        self._deadline_timers: dict[str, Timer] = {}

    def dispatch_many(
        self,
        *,
        collaboration_id: str,
        assignments: Mapping[str, TaskContract],
    ) -> CollaborationDispatchBatch:
        # Domain reservations are serialized, while child execution remains
        # concurrent in the existing BackgroundJobManager.
        with self._dispatch_lock:
            return self._dispatch_many(
                collaboration_id=collaboration_id,
                assignments=assignments,
            )

    def _dispatch_many(
        self,
        *,
        collaboration_id: str,
        assignments: Mapping[str, TaskContract],
    ) -> CollaborationDispatchBatch:
        require_evo_id(collaboration_id, field="collaboration_id")
        for assignment_id, contract in assignments.items():
            require_evo_id(assignment_id, field="assignment_id")
            self._validate_contract(contract)
        if not assignments:
            raise CollaborationError("collaboration dispatch requires at least one assignment")
        assignment_ids = tuple(assignments)
        with self._lock:
            if collaboration_id in self._aggregates or collaboration_id in self._aggregating:
                raise CollaborationError(f"collaboration is already terminal: {collaboration_id}")
            previous_batch = self._batches.get(collaboration_id)
            duplicate_assignment = next(
                (assignment_id for assignment_id in assignment_ids if assignment_id in self._contracts or assignment_id in self._outcomes or assignment_id in self._job_ids),
                None,
            )
            if duplicate_assignment is not None:
                raise CollaborationError(f"assignment was already dispatched: {duplicate_assignment}")
            active_assignments = {assignment_id: entry[1] for assignment_id, entry in self._contracts.items() if entry[0] == collaboration_id and assignment_id not in self._outcomes}
            combined_assignment_ids = (*(previous_batch.assignment_ids if previous_batch is not None else ()), *assignment_ids)
            previous_conflicts = previous_batch.resource_conflicts if previous_batch is not None else ()
            self._batches[collaboration_id] = _RuntimeBatchState(combined_assignment_ids, previous_conflicts)
        try:
            resource_conflicts = self._service.resource_conflicts(
                collaboration_id=collaboration_id,
                assignments={**active_assignments, **assignments},
            )
        except Exception:
            with self._lock:
                if previous_batch is None:
                    self._batches.pop(collaboration_id, None)
                else:
                    self._batches[collaboration_id] = previous_batch
            self._maybe_aggregate(collaboration_id)
            raise
        with self._lock:
            self._batches[collaboration_id] = _RuntimeBatchState(
                combined_assignment_ids,
                (*previous_conflicts, *resource_conflicts),
            )
        delegations = {
            assignment_id: self._service.delegate(
                collaboration_id=collaboration_id,
                assignment_id=assignment_id,
                contract=contract,
                runtime_role=ROLE_RUNTIME_PROFILES[contract.role],
            )
            for assignment_id, contract in assignments.items()
        }
        admitted: list[tuple[str, TaskContract]] = []
        occupied_contracts = tuple(active_assignments.values())
        blocked: list[str] = []
        for assignment_id, contract in assignments.items():
            delegation = delegations[assignment_id]
            if delegation.event_id is None:
                blocked.append(assignment_id)
                self._store_immediate(
                    collaboration_id,
                    assignment_id,
                    contract,
                    status="failure",
                    summary=delegation.diagnostic or "Delegation could not be recorded.",
                )
                continue
            conflict = next(
                (
                    conflicting_resource(contract.resource_claims, admitted_contract.resource_claims)
                    for admitted_contract in (*occupied_contracts, *(item[1] for item in admitted))
                    if conflicting_resource(contract.resource_claims, admitted_contract.resource_claims) is not None
                ),
                None,
            )
            if conflict is not None:
                blocked.append(assignment_id)
                self._store_immediate(
                    collaboration_id,
                    assignment_id,
                    contract,
                    status="failure",
                    summary=f"Dispatch blocked by resource conflict: {conflict}",
                )
                continue
            admitted.append((assignment_id, contract))

        jobs: dict[str, str] = {}
        for assignment_id, contract in admitted:
            if _deadline_expired(contract.deadline_at):
                blocked.append(assignment_id)
                self._store_immediate(
                    collaboration_id,
                    assignment_id,
                    contract,
                    status="timeout",
                    summary="Task contract deadline elapsed before dispatch.",
                )
                continue
            child_run_id = new_evo_id("run")
            with self._lock:
                self._contracts[assignment_id] = (collaboration_id, contract, child_run_id)
            try:
                job = self._background.start(
                    lambda aid=assignment_id: self._execute(aid),
                    session_id=self._parent_session_id,
                    tool_name="delegate",
                    label=f"{contract.role}:{contract.node_id}",
                    task_id=assignment_id,
                    on_cancelled=(
                        lambda cancelled_job, aid=assignment_id: self._on_background_cancelled(
                            aid,
                            cancelled_job.id,
                        )
                    ),
                )
            except BackgroundCapacityError as error:
                blocked.append(assignment_id)
                with self._lock:
                    self._contracts.pop(assignment_id, None)
                self._store_immediate(
                    collaboration_id,
                    assignment_id,
                    contract,
                    status="failure",
                    summary=str(error),
                )
                continue
            with self._lock:
                self._job_ids[assignment_id] = job.id
                existing = self._outcomes.get(assignment_id)
                if existing is not None and existing.job_id is None:
                    self._outcomes[assignment_id] = replace(existing, job_id=job.id)
            jobs[assignment_id] = job.id
            self._schedule_deadline(assignment_id, contract.deadline_at)
        batch = CollaborationDispatchBatch(
            collaboration_id=collaboration_id,
            job_ids=jobs,
            blocked_assignment_ids=tuple(blocked),
            resource_conflicts=resource_conflicts,
        )
        self._maybe_aggregate(collaboration_id)
        return batch

    def cancel(self, assignment_id: str) -> bool:
        with self._lock:
            job_id = self._job_ids.get(assignment_id)
            contract_entry = self._contracts.get(assignment_id)
        if job_id is None:
            return False
        job = self._background.cancel(job_id, session_id=self._parent_session_id)
        if job is None:
            return False
        if job.status == STATUS_CANCELLED and contract_entry is not None:
            collaboration_id, contract, child_run_id = contract_entry
            self._store_immediate(
                collaboration_id,
                assignment_id,
                contract,
                child_run_id=child_run_id,
                status="cancelled",
                summary="Subagent task was cancelled before execution.",
                job_id=job_id,
            )
        return True

    def _on_background_cancelled(self, assignment_id: str, job_id: str) -> None:
        """Close a queued assignment even when cancellation bypasses the adapter."""

        with self._lock:
            contract_entry = self._contracts.get(assignment_id)
        if contract_entry is None:
            return
        collaboration_id, contract, child_run_id = contract_entry
        self._store_immediate(
            collaboration_id,
            assignment_id,
            contract,
            child_run_id=child_run_id,
            status="cancelled",
            summary="Subagent task was cancelled before execution.",
            job_id=job_id,
        )

    def wait(self, timeout: float | None = None) -> bool:
        return self._background.wait(timeout)

    def outcome(self, assignment_id: str) -> RuntimeCollaborationOutcome | None:
        with self._lock:
            outcome = self._outcomes.get(assignment_id)
            job_id = self._job_ids.get(assignment_id)
            contract_entry = self._contracts.get(assignment_id)
        if outcome is not None or job_id is None or contract_entry is None:
            return outcome
        job = self._background.get(job_id, session_id=self._parent_session_id)
        if job is None or job.status == STATUS_RUNNING:
            return None
        if job.status == STATUS_CANCELLED:
            collaboration_id, contract, child_run_id = contract_entry
            return self._store_immediate(
                collaboration_id,
                assignment_id,
                contract,
                child_run_id=child_run_id,
                status="cancelled",
                summary="Subagent task was cancelled before producing an accepted result.",
                job_id=job_id,
            )
        collaboration_id, contract, child_run_id = contract_entry
        return self._store_immediate(
            collaboration_id,
            assignment_id,
            contract,
            child_run_id=child_run_id,
            status="failure",
            summary=job.error or "Subagent job terminated without a recorded result.",
            job_id=job_id,
        )

    def outcomes(self) -> tuple[RuntimeCollaborationOutcome, ...]:
        with self._lock:
            assignment_ids = tuple(self._contracts)
            immediate_ids = tuple(key for key in self._outcomes if key not in self._contracts)
        values = [self.outcome(assignment_id) for assignment_id in (*assignment_ids, *immediate_ids)]
        return tuple(value for value in values if value is not None)

    def aggregate(self, collaboration_id: str) -> CollaborationAggregate | None:
        """Return the one automatically recorded terminal aggregate, if ready."""

        require_evo_id(collaboration_id, field="collaboration_id")
        self._maybe_aggregate(collaboration_id)
        with self._lock:
            return self._aggregates.get(collaboration_id)

    def _execute(self, assignment_id: str) -> ToolResult:
        with self._lock:
            collaboration_id, contract, child_run_id = self._contracts[assignment_id]
            job_id = self._job_ids.get(assignment_id)
        token = current_cancellation_token()
        if token is not None and token.is_cancelled:
            outcome = self._store_immediate(
                collaboration_id,
                assignment_id,
                contract,
                child_run_id=child_run_id,
                status="cancelled",
                summary="Subagent task was cancelled before execution.",
                job_id=job_id,
            )
            return _tool_result(outcome)
        request = SubagentRequest(
            role=ROLE_RUNTIME_PROFILES[contract.role],
            task=_task_prompt(contract),
            parent_session_id=self._parent_session_id,
            parent_task_hash=f"{contract.plan_id}:{contract.node_id}",
            parent_summary=f"Collaboration {collaboration_id}; domain role {contract.role}.",
            run_in_background=True,
            isolate_worktree=(contract.role == "executor" or (contract.role == "verifier" and "write" in contract.allowed_effects)),
            parent_run_id=self._parent_run_id,
            child_run_id=child_run_id,
            max_tool_rounds=contract.budget.max_tool_calls,
            max_output_tokens=contract.budget.max_tokens,
            allowed_tool_names=contract.capabilities,
            allowed_effects=contract.allowed_effects,
        )
        try:
            result = self._runner.run(request, cancellation_token=token)
        except Exception as error:  # noqa: BLE001 - a child failure must still close the assignment
            outcome = self._store_immediate(
                collaboration_id,
                assignment_id,
                contract,
                child_run_id=child_run_id,
                status="cancelled" if token is not None and token.is_cancelled else "failure",
                summary=("Subagent task was cancelled during execution." if token is not None and token.is_cancelled else f"Subagent runner failed: {type(error).__name__}: {error}"),
                job_id=job_id,
            )
            return _tool_result(outcome)
        invalid_result = _invalid_subagent_result(result, expected_role=request.role)
        if token is not None and token.is_cancelled:
            status = "cancelled"
        elif _deadline_expired(contract.deadline_at):
            status = "timeout"
        elif invalid_result is not None:
            outcome = self._store_immediate(
                collaboration_id,
                assignment_id,
                contract,
                child_run_id=child_run_id,
                status="failure",
                summary=f"Subagent returned an invalid result: {invalid_result}",
                job_id=job_id,
            )
            return _tool_result(outcome)
        else:
            status = _status_from_subagent(result)
        child_outcome = self._service.child_outcome(child_run_id)
        evidence = tuple(result.evidence)
        confidence = 0.0
        confidence_source_event_id: str | None = None
        if child_outcome is not None:
            evidence = self._service.verified_child_evidence(
                child_run_id,
                child_outcome.payload.evidence_refs,
            )
            confidence = child_outcome.payload.confidence
            confidence_source_event_id = child_outcome.event_id
            if status == "success" and child_outcome.payload.outcome != "success":
                status = _status_from_outcome(child_outcome.payload.outcome, child_outcome.payload.category)
        claims = ()
        if status == "success" and confidence_source_event_id is not None and evidence:
            claims = (
                CollaborationClaim(
                    claim_key=contract.node_id,
                    conclusion=result.summary,
                    evidence_refs=evidence,
                    source_role=contract.role,
                    independence_key=child_run_id,
                ),
            )
        result_kwargs: dict[str, object] = {}
        if confidence_source_event_id is not None:
            result_kwargs = {
                "confidence_source": "outcome_event",
                "confidence_source_event_id": confidence_source_event_id,
            }
        collaboration_result = CollaborationResult(
            result_id=new_evo_id("event"),
            assignment_id=assignment_id,
            role=contract.role,
            status=status,
            summary=result.summary,
            evidence_refs=evidence,
            confidence=confidence,
            child_run_id=child_run_id,
            child_session_id=result.child_session_id or None,
            claims=claims,
            files_changed=tuple(result.files_changed),
            **result_kwargs,  # type: ignore[arg-type]
        )
        recorded = self._service.record_result(
            collaboration_id=collaboration_id,
            contract=contract,
            result=collaboration_result,
        )
        outcome = RuntimeCollaborationOutcome(collaboration_id, assignment_id, child_run_id, recorded, job_id)
        with self._lock:
            self._outcomes[assignment_id] = outcome
        self._clear_deadline(assignment_id)
        self._maybe_aggregate(collaboration_id)
        return _tool_result(outcome)

    def _store_immediate(
        self,
        collaboration_id: str,
        assignment_id: str,
        contract: TaskContract,
        *,
        status: str,
        summary: str,
        child_run_id: str | None = None,
        job_id: str | None = None,
    ) -> RuntimeCollaborationOutcome:
        with self._lock:
            existing = self._outcomes.get(assignment_id)
            if existing is not None:
                return existing
            result = CollaborationResult(
                result_id=new_evo_id("event"),
                assignment_id=assignment_id,
                role=contract.role,
                status=status,  # type: ignore[arg-type]
                summary=summary,
                evidence_refs=(),
                confidence=0.0,
                child_run_id=child_run_id,
            )
            recorded = self._service.record_result(collaboration_id=collaboration_id, contract=contract, result=result)
            outcome = RuntimeCollaborationOutcome(collaboration_id, assignment_id, child_run_id, recorded, job_id)
            self._outcomes[assignment_id] = outcome
        self._clear_deadline(assignment_id)
        self._maybe_aggregate(collaboration_id)
        return outcome

    def _schedule_deadline(
        self,
        assignment_id: str,
        deadline_at: str | None,
    ) -> None:
        if deadline_at is None:
            return
        deadline = datetime.fromisoformat(deadline_at.replace("Z", "+00:00"))
        delay = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
        # CPython's Windows lock implementation cannot wait for arbitrarily
        # large timeout values.  Long-lived contracts therefore wake in bounded
        # slices and re-check the absolute UTC deadline instead of overflowing
        # a background Timer thread.
        timer = Timer(
            min(delay, _DEADLINE_TIMER_SLICE_SECONDS),
            self._on_deadline_timer,
            args=(assignment_id, deadline_at),
        )
        timer.daemon = True
        with self._lock:
            if assignment_id in self._outcomes:
                return
            previous = self._deadline_timers.pop(assignment_id, None)
            self._deadline_timers[assignment_id] = timer
        if previous is not None:
            previous.cancel()
        timer.start()

    def _on_deadline_timer(self, assignment_id: str, deadline_at: str) -> None:
        with self._lock:
            self._deadline_timers.pop(assignment_id, None)
            if assignment_id in self._outcomes:
                return
        if _deadline_expired(deadline_at):
            self.cancel(assignment_id)
            return
        self._schedule_deadline(assignment_id, deadline_at)

    def _clear_deadline(self, assignment_id: str) -> None:
        with self._lock:
            timer = self._deadline_timers.pop(assignment_id, None)
        if timer is not None:
            timer.cancel()

    def _maybe_aggregate(self, collaboration_id: str) -> None:
        """Record exactly one aggregate after every assignment is terminal."""

        with self._lock:
            if collaboration_id in self._aggregates or collaboration_id in self._aggregating:
                return
            batch = self._batches.get(collaboration_id)
            if batch is None:
                return
            records = tuple(self._outcomes[assignment_id].recorded for assignment_id in batch.assignment_ids if assignment_id in self._outcomes)
            if len(records) != len(batch.assignment_ids):
                return
            self._aggregating.add(collaboration_id)
        try:
            aggregate = self._service.aggregate(
                collaboration_id=collaboration_id,
                records=records,
                resource_conflicts=batch.resource_conflicts,
            )
        except Exception:  # noqa: BLE001 - aggregation must not change child execution results
            with self._lock:
                self._aggregating.discard(collaboration_id)
            return
        with self._lock:
            self._aggregates.setdefault(collaboration_id, aggregate)
            self._aggregating.discard(collaboration_id)

    def _validate_contract(self, contract: TaskContract) -> None:
        if contract.budget.max_attempts != 1:
            raise CollaborationError("formal collaboration currently requires max_attempts=1; retries need independent child Runs")
        if contract.cancellation_mode not in {"cooperative", "terminate"}:
            raise CollaborationError("cancellation_mode must be cooperative or terminate")
        runtime_role = ROLE_RUNTIME_PROFILES[contract.role]
        profile = self._runner.profile(runtime_role)
        if profile is None:
            raise CollaborationError(f"runtime profile is unavailable: {runtime_role}")
        unavailable = set(contract.capabilities).difference(profile.allowed_tool_names)
        if unavailable:
            raise CollaborationError(f"contract requests unavailable capability: {sorted(unavailable)[0]}")
        unknown_effects = set(contract.allowed_effects).difference(_KNOWN_EFFECTS)
        if unknown_effects:
            raise CollaborationError(f"contract declares unknown high-risk effect: {sorted(unknown_effects)[0]}")
        mutation_roles = {"executor", "verifier"}
        if contract.role not in mutation_roles and any(effect in {"write", "delete", "external"} for effect in contract.allowed_effects):
            raise CollaborationError(f"role {contract.role} cannot receive mutation effects")
        if contract.role == "verifier" and "external" in contract.allowed_effects:
            raise CollaborationError("role verifier cannot receive external effects")
        try:
            self._runner.validate_contract_scope(
                role=runtime_role,
                allowed_tool_names=contract.capabilities,
                allowed_effects=contract.allowed_effects,
            )
        except ValueError as error:
            raise CollaborationError(f"contract capability/effect mismatch: {error}") from error


def _deadline_expired(deadline_at: str | None) -> bool:
    if deadline_at is None:
        return False
    deadline = datetime.fromisoformat(deadline_at.replace("Z", "+00:00"))
    return datetime.now(UTC) >= deadline


def _status_from_subagent(result: SubagentResult) -> str:
    if result.ok:
        return "success"
    error = (result.error or "").casefold()
    if "permission" in error or "denied" in error:
        return "permission_denied"
    if "cancel" in error or "interrupt" in error:
        return "cancelled"
    if "timeout" in error or "deadline" in error:
        return "timeout"
    return "failure"


def _status_from_outcome(outcome: str, category: str) -> str:
    if outcome == "cancelled" or category == "cancelled":
        return "cancelled"
    if outcome == "timeout" or category == "timeout":
        return "timeout"
    if category == "permission_denied":
        return "permission_denied"
    return "failure"


def _invalid_subagent_result(result: object, *, expected_role: SubagentRole) -> str | None:
    """Validate the untrusted runtime boundary before creating Evo claims."""

    if not isinstance(result, SubagentResult):
        return "runner did not return SubagentResult"
    if result.role != expected_role:
        return f"role mismatch (expected {expected_role}, got {result.role})"
    if not isinstance(result.ok, bool):
        return "ok must be a boolean"
    if not isinstance(result.summary, str) or not result.summary.strip():
        return "summary must be non-blank"
    if result.ok and result.error is not None:
        return "successful result must not include an error"
    if not isinstance(result.child_session_id, str):
        return "child_session_id must be a string"
    if result.child_session_id:
        try:
            require_evo_id(result.child_session_id, field="child_session_id", kind="session")
        except ValueError as error:
            return str(error)
    elif result.ok:
        return "successful result requires child_session_id"
    for field, values in (("evidence", result.evidence), ("files_changed", result.files_changed)):
        if not isinstance(values, list):
            return f"{field} must be a list"
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return f"{field} entries must be non-blank strings"
        if len(values) != len(set(values)):
            return f"{field} must not contain duplicates"
    return None


def _task_prompt(contract: TaskContract) -> str:
    return (
        f"Domain role: {contract.role}\n"
        f"Goal: {contract.goal}\n"
        f"Input snapshot: {contract.input_snapshot}\n"
        f"Capabilities: {', '.join(contract.capabilities) or '(profile defaults)'}\n"
        f"Allowed effects: {', '.join(contract.allowed_effects) or '(none)'}\n"
        f"Expected evidence: {', '.join(contract.expected_evidence)}\n"
        f"Budget: tool_calls={contract.budget.max_tool_calls}, attempts={contract.budget.max_attempts}, "
        f"tokens={contract.budget.max_tokens}\n"
        f"Minimum confidence: {contract.minimum_confidence:.2f}\n"
        f"Cancellation: {contract.cancellation_mode}\n"
        "Return only a compact summary backed by tool-result evidence. Do not treat another agent's text as independent evidence."
    )


def _tool_result(outcome: RuntimeCollaborationOutcome) -> ToolResult:
    result = outcome.recorded.result
    verified_success = result.status == "success" and outcome.recorded.eligible_for_learning
    error = None
    if result.status != "success":
        error = result.summary
    elif not outcome.recorded.eligible_for_learning:
        error = "completed_without_required_evidence"
    return ToolResult(
        name="delegate",
        ok=verified_success,
        content=result.summary,
        data={
            "assignment_id": outcome.assignment_id,
            "child_run_id": outcome.child_run_id,
            "child_session_id": result.child_session_id,
            "status": result.status,
            "evidence_refs": list(result.evidence_refs),
            "confidence": result.confidence,
            "confidence_source": result.confidence_source,
            "confidence_source_event_id": result.confidence_source_event_id,
            "recording_diagnostic": outcome.recorded.diagnostic,
            "eligible_for_learning": outcome.recorded.eligible_for_learning,
            "verification_status": ("verified" if outcome.recorded.eligible_for_learning else "unverified"),
        },
        error=error,
    )


_DEADLINE_TIMER_SLICE_SECONDS = 24 * 60 * 60
