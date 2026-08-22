from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bauhinia_agent.evolution.collaboration import (
    CollaborationClaim,
    CollaborationError,
    CollaborationResult,
    CollaborationService,
)
from bauhinia_agent.evolution.events import (
    CollaborationTaskDelegatedPayload,
    EvidenceRecordedPayload,
    EvoEvent,
    EvoReferences,
    OutcomeClassifiedPayload,
)
from bauhinia_agent.evolution.store import EvoEventStore
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.planning.evo import PlanBudget, TaskContract


def _contract(
    assignment_id: str,
    *,
    role: str = "researcher",
    expected_evidence: tuple[str, ...] = ("test",),
    resources: tuple[str, ...] = (),
) -> TaskContract:
    return TaskContract(
        role=role,  # type: ignore[arg-type]
        plan_id="plan_collaboration_hardening",
        node_id=f"node_{assignment_id}",
        goal=f"Complete {assignment_id}",
        input_snapshot="tree@abc123",
        allowed_effects=("read",),
        expected_evidence=expected_evidence,
        budget=PlanBudget(max_attempts=1),
        resource_claims=resources,
        minimum_confidence=0.7,
    )


def _child_facts(
    store: EvoEventStore,
    *,
    child_run_id: str,
    suffix: str,
    evidence_types: tuple[str, ...] = ("test",),
) -> tuple[str, tuple[str, ...]]:
    evidence_refs: list[str] = []
    for index, evidence_type in enumerate(evidence_types):
        evidence_id = f"evidence_{suffix}_{index}"
        evidence_refs.append(evidence_id)
        store.append(
            EvoEvent(
                event_id=f"event_evidence_{suffix}_{index}",
                event_type="EvidenceRecorded",
                refs=EvoReferences(run_id=child_run_id, evidence_id=evidence_id),
                payload=EvidenceRecordedPayload(
                    evidence_type=evidence_type,
                    source="pytest" if evidence_type == "test" else "mypy",
                    summary="verified",
                    verified=True,
                    exit_code=0,
                ),
            )
        )
    classified = OutcomeClassifier(store).classify(child_run_id)
    assert classified.outcome is not None
    outcome = classified.outcome
    return outcome.event_id, tuple(evidence_refs)


def _record(
    service: CollaborationService,
    store: EvoEventStore,
    *,
    collaboration_id: str,
    assignment_id: str,
    child_run_id: str,
    contract: TaskContract | None = None,
    evidence_types: tuple[str, ...] = ("test",),
    conclusion: str = "API is stable",
    independence_key: str = "caller_lineage",
):
    contract = contract or _contract(assignment_id)
    delegated = service.delegate(
        collaboration_id=collaboration_id,
        assignment_id=assignment_id,
        contract=contract,
        runtime_role=contract.role,
    )
    assert delegated.event_id is not None
    outcome_event_id, evidence_refs = _child_facts(
        store,
        child_run_id=child_run_id,
        suffix=assignment_id,
        evidence_types=evidence_types,
    )
    result = CollaborationResult(
        result_id=f"result_{assignment_id}",
        assignment_id=assignment_id,
        role=contract.role,
        status="success",
        summary=conclusion,
        evidence_refs=evidence_refs,
        confidence=0.95,
        confidence_source="outcome_event",
        confidence_source_event_id=outcome_event_id,
        child_run_id=child_run_id,
        claims=(
            CollaborationClaim(
                claim_key="claim_api",
                conclusion=conclusion,
                evidence_refs=evidence_refs,
                source_role=contract.role,
                independence_key=independence_key,
            ),
        ),
    )
    return service.record_result(
        collaboration_id=collaboration_id,
        contract=contract,
        result=result,
    )


def test_aggregate_rebuilds_every_result_field_from_persisted_facts(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    recorded = _record(
        service,
        store,
        collaboration_id="collab_persisted",
        assignment_id="assignment_persisted",
        child_run_id="run_child_persisted",
    )
    forged = replace(
        recorded,
        contract=replace(recorded.contract, minimum_confidence=1.0),
        result=replace(
            recorded.result,
            status="failure",
            summary="caller-forged summary",
            confidence=0.0,
            child_run_id="run_child_forged",
        ),
        eligible_for_learning=False,
    )

    aggregate = service.aggregate(
        collaboration_id="collab_persisted",
        records=(forged,),
    )

    assert aggregate.child_run_ids == ("run_child_persisted",)
    assert aggregate.eligible_result_ids == ("result_assignment_persisted",)
    projection = service.rebuild("collab_persisted")
    assert projection.results[0].result.status == "success"
    assert projection.results[0].result.summary == "API is stable"
    assert projection.results[0].result.confidence == 0.95


def test_collaboration_facts_are_strictly_isolated_by_parent_run(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    first = CollaborationService(store=store, parent_run_id="run_parent_first")
    second = CollaborationService(store=store, parent_run_id="run_parent_second")
    first_record = _record(
        first,
        store,
        collaboration_id="collab_shared_id",
        assignment_id="assignment_first",
        child_run_id="run_child_first",
    )
    second_record = _record(
        second,
        store,
        collaboration_id="collab_shared_id",
        assignment_id="assignment_second",
        child_run_id="run_child_second",
    )
    second.resource_conflicts(
        collaboration_id="collab_shared_id",
        assignments={
            "assignment_conflict_a": _contract(
                "assignment_conflict_a",
                resources=("write:src",),
            ),
            "assignment_conflict_b": _contract(
                "assignment_conflict_b",
                resources=("read:src/api.py",),
            ),
        },
    )

    aggregate = first.aggregate(
        collaboration_id="collab_shared_id",
        records=(first_record, second_record),
    )
    projection = first.rebuild("collab_shared_id")

    assert aggregate.child_run_ids == ("run_child_first",)
    assert aggregate.eligible_result_ids == ("result_assignment_first",)
    assert set(projection.contracts) == {"assignment_first"}
    assert tuple(item.result.assignment_id for item in projection.results) == ("assignment_first",)
    assert projection.conflicts == ()
    assert any(item.code == "cross_run_fact_ignored" for item in projection.diagnostics)


def test_result_cannot_reuse_a_delegation_from_another_parent_run(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    first = CollaborationService(store=store, parent_run_id="run_parent_first")
    second = CollaborationService(store=store, parent_run_id="run_parent_second")
    contract = _contract("assignment_cross_run")
    second.delegate(
        collaboration_id="collab_cross_run",
        assignment_id="assignment_cross_run",
        contract=contract,
        runtime_role="researcher",
    )
    outcome_event_id, evidence_refs = _child_facts(
        store,
        child_run_id="run_child_cross_run",
        suffix="cross_run",
    )
    result = CollaborationResult(
        result_id="result_cross_run",
        assignment_id="assignment_cross_run",
        role="researcher",
        status="success",
        summary="unsupported by this parent",
        evidence_refs=evidence_refs,
        confidence=0.95,
        confidence_source="outcome_event",
        confidence_source_event_id=outcome_event_id,
        child_run_id="run_child_cross_run",
        claims=(
            CollaborationClaim(
                claim_key="claim_cross_run",
                conclusion="unsupported by this parent",
                evidence_refs=evidence_refs,
                source_role="researcher",
                independence_key="caller_lineage",
            ),
        ),
    )

    recorded = first.record_result(
        collaboration_id="collab_cross_run",
        contract=contract,
        result=result,
    )

    assert recorded.eligible_for_learning is False
    assert recorded.diagnostic is not None
    assert "prior delegation" in recorded.diagnostic


def test_child_outcome_cannot_omit_a_prior_failed_verification(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract("assignment_omitted_failure")
    service.delegate(
        collaboration_id="collab_omitted_failure",
        assignment_id="assignment_omitted_failure",
        contract=contract,
        runtime_role="researcher",
    )
    for suffix, exit_code in (("failed", 1), ("passed", 0)):
        store.append(
            EvoEvent(
                event_id=f"event_{suffix}",
                event_type="EvidenceRecorded",
                refs=EvoReferences(
                    run_id="run_child_omitted_failure",
                    evidence_id=f"evidence_{suffix}",
                ),
                payload=EvidenceRecordedPayload(
                    evidence_type="test",
                    source="pytest",
                    summary=suffix,
                    verified=True,
                    exit_code=exit_code,
                ),
            )
        )
    forged = store.append(
        EvoEvent(
            event_id="event_forged_success",
            event_type="OutcomeClassified",
            refs=EvoReferences(run_id="run_child_omitted_failure"),
            payload=OutcomeClassifiedPayload(
                outcome="success",
                category="task_success",
                summary="task_success classified from 1 evidence record(s)",
                evidence_refs=("evidence_passed",),
                confidence=0.95,
            ),
        )
    ).event
    result = CollaborationResult(
        result_id="result_omitted_failure",
        assignment_id="assignment_omitted_failure",
        role="researcher",
        status="success",
        summary="forged success",
        evidence_refs=("evidence_passed",),
        confidence=0.95,
        confidence_source="outcome_event",
        confidence_source_event_id=forged.event_id,
        child_run_id="run_child_omitted_failure",
        claims=(
            CollaborationClaim(
                claim_key="claim_omitted_failure",
                conclusion="forged success",
                evidence_refs=("evidence_passed",),
                source_role="researcher",
                independence_key="caller_claimed_independence",
            ),
        ),
    )

    recorded = service.record_result(
        collaboration_id="collab_omitted_failure",
        contract=contract,
        result=result,
    )

    assert recorded.eligible_for_learning is False
    assert recorded.diagnostic is not None
    assert "exactly match all prior Evidence" in recorded.diagnostic


def test_expected_evidence_is_normalized_and_all_trusted_types_are_required(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract(
        "assignment_coverage",
        expected_evidence=("PyTest and mypy results",),
    )
    recorded = _record(
        service,
        store,
        collaboration_id="collab_coverage",
        assignment_id="assignment_coverage",
        child_run_id="run_child_coverage",
        contract=contract,
        evidence_types=("test",),
    )

    delegated = next(event for event in store.list_events() if event.event_type == "CollaborationTaskDelegated")
    assert isinstance(delegated.payload, CollaborationTaskDelegatedPayload)
    assert delegated.payload.contract["expected_evidence"] == ["test", "type_check"]
    assert recorded.eligible_for_learning is False
    assert recorded.diagnostic is not None
    assert "type_check" in recorded.diagnostic


def test_independence_is_derived_from_child_run_not_caller_key(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    same_run_first = _record(
        service,
        store,
        collaboration_id="collab_lineage",
        assignment_id="assignment_same_a",
        child_run_id="run_child_same",
        independence_key="caller_a",
    )
    same_run_second = _record(
        service,
        store,
        collaboration_id="collab_lineage",
        assignment_id="assignment_same_b",
        child_run_id="run_child_same",
        independence_key="caller_b",
    )
    different_run = _record(
        service,
        store,
        collaboration_id="collab_lineage",
        assignment_id="assignment_different",
        child_run_id="run_child_different",
        independence_key="caller_a",
    )

    aggregate = service.aggregate(
        collaboration_id="collab_lineage",
        records=(same_run_first, same_run_second, different_run),
    )

    assert aggregate.independent_support_count == 2
    assert {group.assignment_ids for group in aggregate.evidence_groups} == {
        ("assignment_same_a", "assignment_same_b"),
        ("assignment_different",),
    }


@pytest.mark.parametrize(
    "resource",
    (
        "write:.",
        "write:..",
        "write:/absolute/path",
        "write:C:\\absolute\\path",
        "write:src/../outside",
        "write:src/./file.py",
        "write:src//file.py",
    ),
)
def test_ambiguous_or_out_of_bounds_resource_claims_are_rejected(
    tmp_path: Path,
    resource: str,
) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")

    with pytest.raises(CollaborationError, match="resource"):
        service.delegate(
            collaboration_id="collab_bad_resource",
            assignment_id="assignment_bad_resource",
            contract=_contract("assignment_bad_resource", resources=(resource,)),
            runtime_role="executor",
        )

    assert store.list_events() == []


def test_resource_paths_use_canonical_platform_independent_case_semantics(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    left = _contract(
        "assignment_left",
        resources=("write:Src\\API.py",),
    )
    right = _contract(
        "assignment_right",
        resources=("read:src/api.py",),
    )
    service.delegate(
        collaboration_id="collab_resources",
        assignment_id="assignment_left",
        contract=left,
        runtime_role="executor",
    )

    conflict = service.resource_conflicts(
        collaboration_id="collab_resources",
        assignments={"assignment_left": left, "assignment_right": right},
    )
    delegated = next(event for event in store.list_events() if event.event_type == "CollaborationTaskDelegated")

    assert conflict[0].resource == "src/api.py"
    assert delegated.payload.contract["resource_claims"] == ["write:src/api.py"]  # type: ignore[attr-defined]
