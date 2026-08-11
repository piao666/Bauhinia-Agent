"""Runtime adapter from P10 collaboration contracts to existing subagents.

The adapter intentionally delegates scheduling to ``BackgroundJobManager`` and
execution to ``SubagentRunner``.  It does not create another agent loop,
permission engine, tool registry, or thread pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Mapping

from bauhinia_agent.agent.background import (
    STATUS_CANCELLED,
    STATUS_RUNNING,
    BackgroundCapacityError,
    BackgroundJobManager,
)
from bauhinia_agent.agent.subagent import SubagentRequest, SubagentResult, SubagentRole, SubagentRunner
from bauhinia_agent.evolution.collaboration import (
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
        self._outcomes: dict[str, RuntimeCollaborationOutcome] = {}
        self._contracts: dict[str, tuple[str, TaskContract, str]] = {}
        self._job_ids: dict[str, str] = {}

    def dispatch_many(
        self,
        *,
        collaboration_id: str,
        assignments: Mapping[str, TaskContract],
    ) -> CollaborationDispatchBatch:
        require_evo_id(collaboration_id, field="collaboration_id")
        for assignment_id, contract in assignments.items():
            require_evo_id(assignment_id, field="assignment_id")
            self._validate_contract(contract)
        resource_conflicts = self._service.resource_conflicts(
            collaboration_id=collaboration_id,
            assignments=assignments,
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
                    for _, admitted_contract in admitted
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
            jobs[assignment_id] = job.id
        return CollaborationDispatchBatch(
            collaboration_id=collaboration_id,
            job_ids=jobs,
            blocked_assignment_ids=tuple(blocked),
            resource_conflicts=resource_conflicts,
        )

    def cancel(self, assignment_id: str) -> bool:
        with self._lock:
            job_id = self._job_ids.get(assignment_id)
        if job_id is None:
            return False
        return self._background.cancel(job_id, session_id=self._parent_session_id) is not None

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
        return None

    def outcomes(self) -> tuple[RuntimeCollaborationOutcome, ...]:
        with self._lock:
            assignment_ids = tuple(self._contracts)
            immediate_ids = tuple(key for key in self._outcomes if key not in self._contracts)
        values = [self.outcome(assignment_id) for assignment_id in (*assignment_ids, *immediate_ids)]
        return tuple(value for value in values if value is not None)

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
            isolate_worktree=contract.role == "executor",
            parent_run_id=self._parent_run_id,
            child_run_id=child_run_id,
            max_tool_rounds=contract.budget.max_tool_calls,
            max_output_tokens=contract.budget.max_tokens,
        )
        result = self._runner.run(request)
        if token is not None and token.is_cancelled:
            status = "cancelled"
        elif _deadline_expired(contract.deadline_at):
            status = "timeout"
        else:
            status = _status_from_subagent(result)
        evidence = tuple(dict.fromkeys(result.evidence))
        claims = ()
        if evidence:
            claims = (
                CollaborationClaim(
                    claim_key=contract.node_id,
                    conclusion=result.summary,
                    evidence_refs=evidence,
                    source_role=contract.role,
                    independence_key=child_run_id,
                ),
            )
        collaboration_result = CollaborationResult(
            result_id=new_evo_id("event"),
            assignment_id=assignment_id,
            role=contract.role,
            status=status,
            summary=result.summary,
            evidence_refs=evidence,
            confidence=result.confidence,
            child_run_id=child_run_id,
            child_session_id=result.child_session_id or None,
            claims=claims,
            files_changed=tuple(dict.fromkeys(result.files_changed)),
        )
        recorded = self._service.record_result(
            collaboration_id=collaboration_id,
            contract=contract,
            result=collaboration_result,
        )
        outcome = RuntimeCollaborationOutcome(collaboration_id, assignment_id, child_run_id, recorded, job_id)
        with self._lock:
            self._outcomes[assignment_id] = outcome
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
            return outcome

    def _validate_contract(self, contract: TaskContract) -> None:
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
        if contract.role != "executor" and any(effect in {"write", "delete", "external"} for effect in contract.allowed_effects):
            raise CollaborationError(f"role {contract.role} cannot receive mutation effects")


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
    return ToolResult(
        name="delegate",
        ok=result.status == "success",
        content=result.summary,
        data={
            "assignment_id": outcome.assignment_id,
            "child_run_id": outcome.child_run_id,
            "child_session_id": result.child_session_id,
            "status": result.status,
            "evidence_refs": list(result.evidence_refs),
            "recording_diagnostic": outcome.recorded.diagnostic,
        },
        error=None if result.status == "success" else result.summary,
    )
