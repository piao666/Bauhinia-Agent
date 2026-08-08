from __future__ import annotations

import pytest

from bauhinia_agent.evolution import EvoEventStore
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.context.writer import SessionEventWriter
from bauhinia_agent.planning.evo import (
    DecisionRecord,
    EvoPlanningService,
    PlanBudget,
    PlanGraph,
    PlanGraphError,
    PlanNode,
    PlanningExecutionGate,
    ReplanRequest,
    TaskContract,
)
from bauhinia_agent.planning.service import TaskPlanService
from bauhinia_agent.planning.task_plan_bridge import TaskPlanEvoBridge
from bauhinia_agent.tools.task_plan_support import execute_task_plan_mutation


def _node(
    suffix: str,
    *,
    depends_on: tuple[str, ...] = (),
    status: str = "pending",
    attempts: int = 0,
    max_attempts: int = 1,
) -> PlanNode:
    return PlanNode(
        node_id=f"node_{suffix}",
        goal=f"Goal {suffix}",
        depends_on=depends_on,
        preconditions=("repository is available",),
        risks=("changes need verification",),
        budget=PlanBudget(max_tool_calls=2, max_attempts=max_attempts),
        verification_conditions=("pytest passes",),
        status=status,  # type: ignore[arg-type]
        attempts=attempts,
    )


def _graph(*, version: int = 0, second_status: str = "pending") -> PlanGraph:
    first = _node("inspect", max_attempts=2)
    second = _node("implement", depends_on=(first.node_id,), status=second_status)
    return PlanGraph(plan_id="plan_release", goal="Ship P2", version=version, nodes=(first, second))


def _service(tmp_path) -> EvoPlanningService:
    return EvoPlanningService(store=EvoEventStore(tmp_path), run_id="run_p2", session_id="session_p2")


def test_plan_graph_round_trips_and_requires_a_valid_dag() -> None:
    graph = _graph()

    assert PlanGraph.from_dict(graph.to_dict()) == graph
    with pytest.raises(PlanGraphError, match="cycle"):
        PlanGraph(
            plan_id="plan_cycle",
            goal="cycle",
            version=0,
            nodes=(_node("a", depends_on=("node_b",)), _node("b", depends_on=("node_a",))),
        )
    with pytest.raises(PlanGraphError, match="not ready"):
        _graph(second_status="in_progress")


def test_plan_graph_state_transitions_include_failure_cancellation_and_recovery() -> None:
    graph = _graph()
    failed_first = _node("inspect", status="failed", attempts=1, max_attempts=2)
    failed = graph.replace_nodes((failed_first, graph.node("node_implement")), expected_version=0)
    recovered_first = _node("inspect", status="pending", attempts=1, max_attempts=2)
    recovered = failed.replace_nodes((recovered_first, failed.node("node_implement")), expected_version=1)
    cancelled_first = _node("inspect", status="cancelled", attempts=1, max_attempts=2)
    cancelled = recovered.replace_nodes((cancelled_first, recovered.node("node_implement")), expected_version=2)
    resumed = cancelled.replace_nodes((recovered_first, cancelled.node("node_implement")), expected_version=3)

    assert (failed.version, recovered.version, cancelled.version, resumed.version) == (1, 2, 3, 4)
    with pytest.raises(PlanGraphError, match="cannot transition"):
        graph.replace_nodes((_node("inspect", status="completed", max_attempts=2), graph.node("node_implement")), expected_version=0).replace_nodes(
            (_node("inspect", status="in_progress", max_attempts=2), graph.node("node_implement")), expected_version=1
        )


def test_decision_record_is_structured_and_rejects_private_reasoning_shape() -> None:
    decision = DecisionRecord(
        decision_id="decision_1",
        plan_id="plan_release",
        node_id="node_inspect",
        subgoal="inspect the current implementation",
        evidence_refs=("evidence_tests",),
        assumptions=("tests are deterministic",),
        options_considered=("reuse", "replace"),
        selected_action="reuse",
        rationale_summary="Existing task-plan APIs already own session state.",
        confidence=0.8,
        expected_observation="a P2 graph can be persisted",
        verification_method="pytest",
    )

    assert decision.to_payload().to_dict()["decision_id"] == "decision_1"
    assert "private_reasoning" not in DecisionRecord.__dataclass_fields__


@pytest.mark.parametrize(
    "trigger",
    [
        "verification_failed",
        "tool_error",
        "permission_denied",
        "precondition_invalid",
        "context_conflict",
        "budget_exhausted",
        "user_changed_goal",
        "cancelled",
    ],
)
def test_replan_persists_trigger_version_and_evidence(tmp_path, trigger: str) -> None:
    service = _service(tmp_path)
    original = service.create(_graph())
    replacement = PlanNode(
        node_id="node_implement_retry",
        goal="Implement with a narrower change",
        depends_on=("node_inspect",),
        preconditions=("inspection completed",),
        risks=("avoid retrying the failed action",),
        budget=PlanBudget(max_attempts=1),
        verification_conditions=("focused pytest passes",),
        parent_node_id="node_implement",
    )
    retired = PlanNode.from_dict({**original.node("node_implement").to_dict(), "status": "superseded"})
    next_graph = original.replace_nodes((original.node("node_inspect"), retired, replacement), expected_version=0)

    result = service.replan(
        ReplanRequest(
            plan_id=original.plan_id,
            expected_version=0,
            trigger=trigger,  # type: ignore[arg-type]
            outcome="replace",
            evidence_refs=("evidence_failure",),
            rationale_summary="Observed evidence invalidated the previous node.",
            next_graph=next_graph,
        )
    )

    assert result.version == 1
    restored = service.get(original.plan_id)
    assert restored == result
    events = EvoEventStore(tmp_path).list_events()
    assert [event.event_type for event in events] == ["PlanCreated", "DecisionRecorded", "PlanNodeUpdated", "PlanNodeUpdated"]
    assert events[-1].payload.extensions["replan"] == {
        "trigger": trigger,
        "outcome": "replace",
        "evidence_refs": ["evidence_failure"],
    }


def test_service_records_node_update_and_rebuilds_versioned_graph(tmp_path) -> None:
    service = _service(tmp_path)
    graph = service.create(_graph())
    started_first = PlanNode.from_dict({**graph.node("node_inspect").to_dict(), "status": "in_progress"})

    started = service.update_node(
        plan_id=graph.plan_id,
        expected_version=0,
        node=started_first,
        change_summary="inspection started",
    )
    updated_first = PlanNode.from_dict({**started.node("node_inspect").to_dict(), "status": "completed"})
    result = service.update_node(
        plan_id=graph.plan_id,
        expected_version=1,
        node=updated_first,
        change_summary="inspection evidence accepted",
    )

    assert result.version == 2
    assert service.get(graph.plan_id) == result
    with pytest.raises(PlanGraphError, match="version conflict"):
        service.update_node(plan_id=graph.plan_id, expected_version=0, node=updated_first, change_summary="stale")


def _contract(role: str) -> TaskContract:
    return TaskContract(
        role=role,  # type: ignore[arg-type]
        plan_id="plan_release",
        node_id="node_inspect",
        goal="Inspect",
        input_snapshot="session snapshot hash",
        allowed_effects=("read",),
        expected_evidence=("pytest",),
        budget=PlanBudget(max_tool_calls=1, max_attempts=1),
    )


def test_execution_gate_requires_existing_permission_and_independent_verifier() -> None:
    calls: list[str] = []
    gate = PlanningExecutionGate(
        permission_preflight=lambda contract: calls.append(f"permission:{contract.node_id}") is None,
        execute=lambda contract: calls.append(f"execute:{contract.node_id}") or {"result": "ok"},
        verify=lambda contract, result: (True, ("evidence_test",)),
    )

    receipt = gate.execute(_contract("executor"))
    verified, evidence = gate.verify(receipt, _contract("verifier"))

    assert receipt.permission_checked is True
    assert verified is True
    assert evidence == ("evidence_test",)
    assert calls == ["permission:node_inspect", "execute:node_inspect"]
    denied = PlanningExecutionGate(permission_preflight=lambda contract: False, execute=lambda contract: None, verify=lambda contract, result: (True, ()))
    with pytest.raises(PlanGraphError, match="permission"):
        denied.execute(_contract("executor"))
    with pytest.raises(PlanGraphError, match="verifier"):
        gate.verify(receipt, _contract("executor"))


def test_session_task_plan_is_mirrored_without_becoming_a_second_control_plane(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    writer = SessionEventWriter(store=store, session_id="session_p2")
    writer.append_session_created()
    bridge = TaskPlanEvoBridge(root=str(tmp_path), session_id="session_p2")
    task_service = TaskPlanService(store=store, writer=writer, observe_evo_plan=bridge.observe)

    created = task_service.create(
        mode="linear",
        expected_revision=0,
        tasks=[{"id": "inspect", "content": "Inspect"}, {"id": "verify", "content": "Verify"}],
    )
    completed = task_service.update(expected_revision=1, updates=[{"id": "inspect", "status": "completed"}])

    graph_events = EvoEventStore(tmp_path).list_events()
    graph = PlanGraph.from_dict(graph_events[0].payload.extensions["graph"])
    assert created.evo_diagnostic is None
    assert completed.evo_diagnostic is None
    assert graph.nodes[0].status == created.plan.tasks[0].status
    restored = EvoPlanningService(store=EvoEventStore(tmp_path), run_id="run_session_p2", session_id="session_p2").get(graph.plan_id)
    assert restored is not None
    assert restored.node(graph.nodes[0].node_id).status == completed.plan.tasks[0].status


def test_evo_recording_failure_is_discoverable_without_rejecting_task_plan(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    writer = SessionEventWriter(store=store, session_id="session_p2")
    writer.append_session_created()
    service = TaskPlanService(store=store, writer=writer, observe_evo_plan=lambda plan, operation: "evo_plan_recording_failed: offline")

    result = execute_task_plan_mutation(
        "task_create",
        lambda: service.create(mode="linear", expected_revision=0, tasks=[{"id": "inspect", "content": "Inspect"}]),
    )

    assert result.ok is True
    assert result.data["evo_diagnostic"] == "evo_plan_recording_failed: offline"
    assert service.current() is not None
