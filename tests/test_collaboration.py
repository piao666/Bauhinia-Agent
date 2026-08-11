from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from bauhinia_agent.agent.background import BackgroundJobManager
from bauhinia_agent.agent.collaboration import CollaborationRuntimeAdapter, ROLE_RUNTIME_PROFILES
from bauhinia_agent.agent.subagent import SUBAGENT_PROFILES, SubagentRequest, SubagentResult
from bauhinia_agent.evolution.collaboration import (
    CollaborationClaim,
    CollaborationError,
    CollaborationResult,
    CollaborationService,
)
from bauhinia_agent.evolution.compiler import ExperienceCompiler
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import (
    CollaborationRunAggregatedPayload,
    CollaborationTaskDelegatedPayload,
    EvoEvent,
)
from bauhinia_agent.evolution.store import EvoEventStore
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.planning.evo import PlanBudget, PlanGraphError, TaskContract
from bauhinia_agent.runtime.cancellation import current_cancellation_token


def _contract(
    role: str,
    node: str,
    *,
    resources: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = (),
    effects: tuple[str, ...] = ("read",),
) -> TaskContract:
    return TaskContract(
        role=role,  # type: ignore[arg-type]
        plan_id="plan_p10",
        node_id=node,
        goal=f"Complete {node}",
        input_snapshot="tree@abc123",
        allowed_effects=effects,
        expected_evidence=("tool result event",),
        budget=PlanBudget(max_tool_calls=5, max_attempts=1, max_tokens=2000),
        capabilities=capabilities,
        resource_claims=resources,
        minimum_confidence=0.7,
        cancellation_mode="cooperative",
        deadline_at="2099-01-01T00:00:00Z",
    )


def _result(
    assignment: str,
    role: str,
    conclusion: str,
    evidence: tuple[str, ...],
    independence: str,
    *,
    run: str,
) -> CollaborationResult:
    claim = CollaborationClaim(
        claim_key="claim_api",
        conclusion=conclusion,
        evidence_refs=evidence,
        source_role=role,  # type: ignore[arg-type]
        independence_key=independence,
    )
    return CollaborationResult(
        result_id=f"result_{assignment}",
        assignment_id=assignment,
        role=role,  # type: ignore[arg-type]
        status="success",
        summary=conclusion,
        evidence_refs=evidence,
        confidence=0.9,
        child_run_id=run,
        child_session_id=f"session_{assignment}",
        claims=(claim,),
    )


def test_task_contract_supports_all_roles_and_round_trips() -> None:
    for role in ROLE_RUNTIME_PROFILES:
        contract = _contract(role, f"node_{role}")
        assert TaskContract.from_dict(contract.to_dict()) == contract

    with pytest.raises(PlanGraphError, match="resource_claims"):
        _contract("researcher", "node_bad", resources=("repo/path",))


def test_collaboration_event_payload_is_explicit_and_forward_compatible() -> None:
    raw = {
        "event_id": "event_delegate",
        "event_type": "CollaborationTaskDelegated",
        "schema_version": "v1",
        "occurred_at": "2026-08-09T00:00:00Z",
        "sequence": 1,
        "refs": {"run_id": "run_parent"},
        "payload": {
            "collaboration_id": "collab_1",
            "assignment_id": "assignment_1",
            "runtime_role": "researcher",
            "contract": _contract("researcher", "node_1").to_dict(),
            "future_field": {"kept": True},
        },
    }
    event = EvoEvent.from_dict(raw)
    assert isinstance(event.payload, CollaborationTaskDelegatedPayload)
    assert event.payload.extensions == {"future_field": {"kept": True}}
    assert EvoEvent.from_json(event.to_json()).to_dict() == event.to_dict()


def test_results_without_evidence_or_confidence_are_not_learning_inputs(tmp_path: Path) -> None:
    service = CollaborationService(store=EvoEventStore(tmp_path), parent_run_id="run_parent")
    contract = _contract("researcher", "node_gate")
    result = CollaborationResult(
        result_id="result_empty",
        assignment_id="assignment_empty",
        role="researcher",
        status="success",
        summary="Looks correct but has no independent evidence.",
        evidence_refs=(),
        confidence=0.99,
        child_run_id="run_child",
    )
    record = service.record_result(collaboration_id="collab_gate", contract=contract, result=result)
    assert record.event_id is not None
    assert record.eligible_for_learning is False


def test_delegation_redacts_sensitive_input_snapshot_at_persistence_boundary(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract("researcher", "node_secret")
    secret_contract = TaskContract.from_dict({**contract.to_dict(), "input_snapshot": "TOKEN=super-secret"})

    delegated = service.delegate(
        collaboration_id="collab_secret",
        assignment_id="assignment_secret",
        contract=secret_contract,
        runtime_role="researcher",
    )

    assert delegated.event_id is not None
    assert "super-secret" not in store.events_path.read_text(encoding="utf-8")

    low_confidence = CollaborationResult(
        result_id="result_low",
        assignment_id="assignment_low",
        role="researcher",
        status="success",
        summary="Evidence exists but confidence is too low.",
        evidence_refs=("evidence_1",),
        confidence=0.2,
        child_run_id="run_child_low",
    )
    record = service.record_result(collaboration_id="collab_gate", contract=contract, result=low_confidence)
    assert record.eligible_for_learning is False


def test_resource_conflicts_are_recorded_and_parent_aggregates_child_runs(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contracts = {
        "assignment_writer": _contract("executor", "node_write", resources=("write:src",), effects=("write",)),
        "assignment_reader": _contract("researcher", "node_read", resources=("read:src/api.py",)),
        "assignment_docs": _contract("researcher", "node_docs", resources=("read:docs",)),
    }
    conflicts = service.resource_conflicts(collaboration_id="collab_resources", assignments=contracts)
    assert len(conflicts) == 1
    assert conflicts[0].resource == "src"
    assert conflicts[0].event_id is not None

    records = []
    for assignment, role, run, evidence in (
        ("assignment_writer", "executor", "run_writer", ("evidence_writer",)),
        ("assignment_docs", "researcher", "run_docs", ("evidence_docs",)),
    ):
        result = _result(assignment, role, "API is stable", evidence, run, run=run)
        records.append(service.record_result(collaboration_id="collab_resources", contract=contracts[assignment], result=result))
    aggregate = service.aggregate(collaboration_id="collab_resources", records=records, resource_conflicts=conflicts)
    assert aggregate.event_id is not None
    assert aggregate.child_run_ids == ("run_writer", "run_docs")
    aggregate_event = store.list_events()[-1]
    assert isinstance(aggregate_event.payload, CollaborationRunAggregatedPayload)
    assert aggregate_event.payload.child_run_ids == aggregate.child_run_ids


def test_copied_agent_results_count_once_but_independent_evidence_counts_again(tmp_path: Path) -> None:
    service = CollaborationService(store=EvoEventStore(tmp_path), parent_run_id="run_parent")
    records = []
    specs = (
        ("assignment_1", "researcher", ("evidence_shared",), "lineage_a", "run_1"),
        ("assignment_2", "critic", ("evidence_shared",), "lineage_b", "run_2"),
        ("assignment_3", "verifier", ("evidence_other",), "lineage_a", "run_3"),
        ("assignment_4", "verifier", ("evidence_independent",), "lineage_c", "run_4"),
    )
    for assignment, role, evidence, independence, run in specs:
        contract = _contract(role, f"node_{assignment}")
        result = _result(assignment, role, "API is stable", evidence, independence, run=run)
        records.append(service.record_result(collaboration_id="collab_copy", contract=contract, result=result))

    aggregate = service.aggregate(collaboration_id="collab_copy", records=records)
    assert aggregate.independent_support_count == 2
    grouped_assignments = {assignment for group in aggregate.evidence_groups for assignment in group.assignment_ids}
    assert grouped_assignments == {item[0] for item in specs}


def test_conflicting_conclusions_keep_branches_for_verifier_or_curator(tmp_path: Path) -> None:
    service = CollaborationService(store=EvoEventStore(tmp_path), parent_run_id="run_parent")
    left_contract = _contract("researcher", "node_left")
    right_contract = _contract("critic", "node_right")
    left = service.record_result(
        collaboration_id="collab_conflict",
        contract=left_contract,
        result=_result("assignment_left", "researcher", "API is stable", ("evidence_left",), "left", run="run_left"),
    )
    right = service.record_result(
        collaboration_id="collab_conflict",
        contract=right_contract,
        result=_result("assignment_right", "critic", "API is breaking", ("evidence_right",), "right", run="run_right"),
    )
    aggregate = service.aggregate(collaboration_id="collab_conflict", records=(left, right))
    assert len(aggregate.conflicts) == 1
    conflict = aggregate.conflicts[0]
    assert conflict.conflict_kind == "conclusion"
    assert conflict.resolution_state == "pending_verifier_or_curator"
    assert set(conflict.assignment_ids) == {"assignment_left", "assignment_right"}
    assert len(conflict.branches) == 2


def test_compiler_links_parent_aggregate_without_counting_child_runs_as_independent_runs(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract("verifier", "node_verifier")
    record = service.record_result(
        collaboration_id="collab_compile",
        contract=contract,
        result=_result(
            "assignment_verifier",
            "verifier",
            "API is stable",
            ("evidence_child",),
            "lineage_child",
            run="run_child",
        ),
    )
    aggregate = service.aggregate(collaboration_id="collab_compile", records=(record,))
    assert aggregate.event_id is not None
    EvidenceAdapter(store).record(
        EvidenceInput(
            run_id="run_parent",
            evidence_type="test",
            source="pytest",
            summary="1 passed",
            exit_code=0,
            verified=True,
        )
    )
    OutcomeClassifier(store).classify("run_parent")

    compiled = ExperienceCompiler(store).compile("run_parent", environment_summary="Windows")

    assert compiled.persisted is True
    candidate = compiled.candidates[0]
    assert aggregate.event_id in candidate.payload.source_event_ids
    assert candidate.payload.source_run_ids == ("run_parent",)


class _ConcurrentRunner:
    def __init__(self, barrier: threading.Barrier | None = None, *, slow: bool = False) -> None:
        self.barrier = barrier
        self.slow = slow
        self.requests: list[SubagentRequest] = []
        self._lock = threading.Lock()

    def profile(self, role: str):
        return SUBAGENT_PROFILES.get(role)

    def run(self, request: SubagentRequest) -> SubagentResult:
        with self._lock:
            self.requests.append(request)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        if self.slow:
            token = current_cancellation_token()
            for _ in range(100):
                if token is not None and token.is_cancelled:
                    return SubagentResult(False, request.role, "session_cancelled", "cancelled", error="cancelled")
                time.sleep(0.005)
        return SubagentResult(
            True,
            request.role,
            f"session_{request.child_run_id}",
            "Verified result",
            evidence=[f"evidence_{request.child_run_id}"],
            confidence=0.9,
        )


def _adapter(tmp_path: Path, runner: _ConcurrentRunner, *, workers: int = 2):
    manager = BackgroundJobManager(max_jobs=4, max_workers=workers)
    service = CollaborationService(store=EvoEventStore(tmp_path), parent_run_id="run_parent", session_id="session_parent")
    adapter = CollaborationRuntimeAdapter(
        runner=runner,  # type: ignore[arg-type]
        background_manager=manager,
        service=service,
        parent_session_id="session_parent",
        parent_run_id="run_parent",
    )
    return adapter, manager, service


def test_runtime_adapter_reuses_background_manager_for_parallel_work(tmp_path: Path) -> None:
    runner = _ConcurrentRunner(threading.Barrier(2))
    adapter, manager, service = _adapter(tmp_path, runner)
    try:
        batch = adapter.dispatch_many(
            collaboration_id="collab_parallel",
            assignments={
                "assignment_a": _contract("researcher", "node_a", resources=("read:src/a",)),
                "assignment_b": _contract("verifier", "node_b", resources=("read:src/b",)),
            },
        )
        assert set(batch.job_ids) == {"assignment_a", "assignment_b"}
        assert adapter.wait(3)
        outcomes = adapter.outcomes()
        assert len(outcomes) == 2
        assert all(outcome.recorded.eligible_for_learning for outcome in outcomes)
        aggregate = service.aggregate(
            collaboration_id="collab_parallel",
            records=tuple(outcome.recorded for outcome in outcomes),
        )
        assert len(aggregate.child_run_ids) == 2
        assert {request.parent_run_id for request in runner.requests} == {"run_parent"}
        assert {request.max_tool_rounds for request in runner.requests} == {5}
        assert {request.max_output_tokens for request in runner.requests} == {2000}
    finally:
        manager.shutdown()


def test_runtime_adapter_blocks_write_conflict_without_duplicate_execution(tmp_path: Path) -> None:
    runner = _ConcurrentRunner()
    adapter, manager, _ = _adapter(tmp_path, runner, workers=1)
    try:
        batch = adapter.dispatch_many(
            collaboration_id="collab_write",
            assignments={
                "assignment_first": _contract("executor", "node_first", resources=("write:src/api.py",), effects=("write",)),
                "assignment_second": _contract("executor", "node_second", resources=("write:src",), effects=("write",)),
            },
        )
        assert batch.blocked_assignment_ids == ("assignment_second",)
        assert adapter.wait(3)
        assert len(runner.requests) == 1
        assert runner.requests[0].isolate_worktree is True
    finally:
        manager.shutdown()


def test_runtime_rejects_unknown_effect_before_recording_or_execution(tmp_path: Path) -> None:
    runner = _ConcurrentRunner()
    adapter, manager, _ = _adapter(tmp_path, runner, workers=1)
    contract = _contract("researcher", "node_unknown")
    unsafe = TaskContract.from_dict({**contract.to_dict(), "allowed_effects": ["teleport"]})
    try:
        with pytest.raises(CollaborationError, match="unknown high-risk effect"):
            adapter.dispatch_many(collaboration_id="collab_unknown", assignments={"assignment_unknown": unsafe})
        assert runner.requests == []
    finally:
        manager.shutdown()


def test_runtime_cancellation_records_cancelled_result_and_reclaims_job(tmp_path: Path) -> None:
    runner = _ConcurrentRunner(slow=True)
    adapter, manager, _ = _adapter(tmp_path, runner, workers=1)
    try:
        batch = adapter.dispatch_many(
            collaboration_id="collab_cancel",
            assignments={"assignment_slow": _contract("researcher", "node_slow")},
        )
        assert "assignment_slow" in batch.job_ids
        deadline = time.monotonic() + 2
        while not runner.requests and time.monotonic() < deadline:
            time.sleep(0.005)
        assert adapter.cancel("assignment_slow") is True
        assert adapter.wait(3)
        outcome = adapter.outcome("assignment_slow")
        assert outcome is not None
        assert outcome.recorded.result.status == "cancelled"
        assert outcome.recorded.eligible_for_learning is False
    finally:
        manager.shutdown()
