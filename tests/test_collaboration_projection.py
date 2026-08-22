from __future__ import annotations

from pathlib import Path

import pytest

from bauhinia_agent.evolution.collaboration import (
    CollaborationClaim,
    CollaborationError,
    CollaborationResult,
    CollaborationService,
)
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import (
    CollaborationTaskResultRecordedPayload,
    EvidenceRecordedPayload,
    EvoEvent,
    EvoEventError,
    EvoReferences,
    OutcomeClassifiedPayload,
)
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.evolution.store import EvoEventStore
from bauhinia_agent.planning.evo import PlanBudget, TaskContract


def _contract(role: str, node_id: str) -> TaskContract:
    return TaskContract(
        role=role,  # type: ignore[arg-type]
        plan_id="plan_projection",
        node_id=node_id,
        goal=f"Complete {node_id}",
        input_snapshot="tree@abc123",
        allowed_effects=("read",),
        expected_evidence=("verified test",),
        budget=PlanBudget(max_attempts=1),
        minimum_confidence=0.7,
    )


def _child_outcome(store: EvoEventStore, run_id: str) -> tuple[str, tuple[str, ...], float]:
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="1 passed",
            exit_code=0,
            verified=True,
        )
    )
    assert evidence.evidence is not None
    outcome = OutcomeClassifier(store).classify(run_id)
    assert outcome.outcome is not None
    return outcome.outcome.event_id, (evidence.evidence.evidence_id,), outcome.outcome.payload.confidence


def _result(
    *,
    assignment_id: str,
    role: str,
    run_id: str,
    evidence_refs: tuple[str, ...],
    outcome_event_id: str,
    confidence: float,
    conclusion: str = "API is stable",
    independence_key: str = "lineage_a",
) -> CollaborationResult:
    return CollaborationResult(
        result_id=f"result_{assignment_id}",
        assignment_id=assignment_id,
        role=role,  # type: ignore[arg-type]
        status="success",
        summary=conclusion,
        evidence_refs=evidence_refs,
        confidence=confidence,
        confidence_source="outcome_event",
        confidence_source_event_id=outcome_event_id,
        child_run_id=run_id,
        claims=(
            CollaborationClaim(
                claim_key="claim_api",
                conclusion=conclusion,
                evidence_refs=evidence_refs,
                source_role=role,  # type: ignore[arg-type]
                independence_key=independence_key,
            ),
        ),
    )


def test_result_fact_persists_complete_redacted_claim_and_outcome_provenance(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract("researcher", "node_secret")
    service.delegate(
        collaboration_id="collab_secret",
        assignment_id="assignment_secret",
        contract=contract,
        runtime_role="researcher",
    )
    source_event_id, evidence_refs, confidence = _child_outcome(store, "run_child_secret")
    result = _result(
        assignment_id="assignment_secret",
        role="researcher",
        run_id="run_child_secret",
        evidence_refs=evidence_refs,
        outcome_event_id=source_event_id,
        confidence=confidence,
        conclusion="TOKEN=super-secret",
        independence_key="API_KEY=also-secret",
    )

    recorded = service.record_result(collaboration_id="collab_secret", contract=contract, result=result)

    assert recorded.eligible_for_learning is True
    event = store.list_events()[-1]
    assert isinstance(event.payload, CollaborationTaskResultRecordedPayload)
    assert event.payload.claim_format == "v2_full"
    assert event.payload.claims_rebuildable is True
    assert event.payload.claims[0].source_role == "researcher"
    assert event.payload.claims[0].evidence_refs == evidence_refs
    assert event.payload.confidence_source == "outcome_event"
    assert event.payload.confidence_source_event_id == source_event_id
    assert "super-secret" not in store.events_path.read_text(encoding="utf-8")
    assert "also-secret" not in store.events_path.read_text(encoding="utf-8")
    assert EvoEvent.from_json(event.to_json()).to_dict() == event.to_dict()


def test_result_rejects_claim_evidence_outside_result_evidence(tmp_path: Path) -> None:
    service = CollaborationService(store=EvoEventStore(tmp_path), parent_run_id="run_parent")
    contract = _contract("researcher", "node_subset")
    result = CollaborationResult(
        result_id="result_subset",
        assignment_id="assignment_subset",
        role="researcher",
        status="success",
        summary="summary",
        evidence_refs=("evidence_result",),
        confidence=0.9,
        child_run_id="run_child_subset",
        claims=(
            CollaborationClaim(
                claim_key="claim_subset",
                conclusion="conclusion",
                evidence_refs=("evidence_other",),
                source_role="researcher",
                independence_key="lineage_subset",
            ),
        ),
    )

    with pytest.raises(CollaborationError, match="subset"):
        service.record_result(collaboration_id="collab_subset", contract=contract, result=result)


def test_forged_or_cross_run_outcome_source_fails_closed(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract("researcher", "node_source")
    source_event_id, evidence_refs, confidence = _child_outcome(store, "run_other")
    result = _result(
        assignment_id="assignment_source",
        role="researcher",
        run_id="run_claimed_child",
        evidence_refs=evidence_refs,
        outcome_event_id=source_event_id,
        confidence=confidence,
    )

    recorded = service.record_result(collaboration_id="collab_source", contract=contract, result=result)

    assert recorded.eligible_for_learning is False
    assert recorded.diagnostic is not None
    assert "different child Run" in recorded.diagnostic


def test_outcome_cannot_be_retroactively_supported_by_future_evidence(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract("researcher", "node_future_evidence")
    service.delegate(
        collaboration_id="collab_future_evidence",
        assignment_id="assignment_future_evidence",
        contract=contract,
        runtime_role="researcher",
    )
    outcome = store.append(
        EvoEvent(
            event_id="event_outcome_before_evidence",
            event_type="OutcomeClassified",
            refs=EvoReferences(run_id="run_future_evidence"),
            payload=OutcomeClassifiedPayload(
                outcome="success",
                category="task_success",
                summary="forged future support",
                evidence_refs=("evidence_future_support",),
                confidence=0.95,
            ),
        )
    ).event
    store.append(
        EvoEvent(
            event_id="event_future_evidence",
            event_type="EvidenceRecorded",
            refs=EvoReferences(
                run_id="run_future_evidence",
                evidence_id="evidence_future_support",
            ),
            payload=EvidenceRecordedPayload(
                evidence_type="test",
                source="pytest",
                summary="late pass",
                verified=True,
                exit_code=0,
            ),
        )
    )
    result = _result(
        assignment_id="assignment_future_evidence",
        role="researcher",
        run_id="run_future_evidence",
        evidence_refs=("evidence_future_support",),
        outcome_event_id=outcome.event_id,
        confidence=0.95,
    )

    recorded = service.record_result(
        collaboration_id="collab_future_evidence",
        contract=contract,
        result=result,
    )

    assert recorded.eligible_for_learning is False
    assert recorded.diagnostic is not None
    assert "not canonical" in recorded.diagnostic


def test_legacy_fingerprint_only_result_is_readable_but_downgraded(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    contract = _contract("researcher", "node_legacy")
    service.delegate(
        collaboration_id="collab_legacy",
        assignment_id="assignment_legacy",
        contract=contract,
        runtime_role="researcher",
    )
    legacy = CollaborationTaskResultRecordedPayload.from_dict(
        {
            "collaboration_id": "collab_legacy",
            "assignment_id": "assignment_legacy",
            "status": "success",
            "summary": "legacy result",
            "evidence_refs": ["evidence_legacy"],
            "confidence": 0.9,
            "eligible_for_learning": True,
            "child_run_id": "run_child_legacy",
            "claim_fingerprints": ["legacy-fingerprint"],
            "files_changed": [],
        }
    )
    assert legacy.claim_format == "v1_fingerprint_only"
    assert legacy.claims_rebuildable is False
    store.append(
        EvoEvent(
            event_id="event_legacy_result",
            event_type="CollaborationTaskResultRecorded",
            refs=EvoReferences(run_id="run_parent"),
            payload=legacy,
        )
    )

    projection = service.rebuild("collab_legacy")

    assert len(projection.results) == 1
    assert projection.results[0].eligible_for_learning is False
    assert any(item.code == "legacy_fingerprint_only_claims" for item in projection.diagnostics)


def test_projection_rebuilds_groups_and_conflicts_and_aggregate_is_idempotent(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path)
    service = CollaborationService(store=store, parent_run_id="run_parent")
    specs = (
        ("assignment_left", "researcher", "run_left", "API is stable", "lineage_left"),
        ("assignment_right", "critic", "run_right", "API is breaking", "lineage_right"),
    )
    records = []
    for assignment_id, role, run_id, conclusion, independence_key in specs:
        contract = _contract(role, f"node_{assignment_id}")
        service.delegate(
            collaboration_id="collab_replay",
            assignment_id=assignment_id,
            contract=contract,
            runtime_role=role,
        )
        source_event_id, evidence_refs, confidence = _child_outcome(store, run_id)
        result = _result(
            assignment_id=assignment_id,
            role=role,
            run_id=run_id,
            evidence_refs=evidence_refs,
            outcome_event_id=source_event_id,
            confidence=confidence,
            conclusion=conclusion,
            independence_key=independence_key,
        )
        records.append(service.record_result(collaboration_id="collab_replay", contract=contract, result=result))

    first = service.aggregate(collaboration_id="collab_replay", records=records)
    event_count = len(store.list_events())
    second = service.aggregate(collaboration_id="collab_replay", records=records)
    projection = service.rebuild("collab_replay")

    assert second.event_id == first.event_id
    assert len(store.list_events()) == event_count
    assert set(projection.contracts) == {"assignment_left", "assignment_right"}
    assert len(projection.results) == 2
    assert len(projection.evidence_groups) == 2
    assert len(projection.conflicts) == 1
    assert projection.conflicts[0].conflict_kind == "conclusion"


def test_complete_claim_fingerprint_tampering_is_rejected() -> None:
    with pytest.raises(EvoEventError, match="fingerprint"):
        CollaborationTaskResultRecordedPayload.from_dict(
            {
                "collaboration_id": "collab_tamper",
                "assignment_id": "assignment_tamper",
                "status": "success",
                "summary": "summary",
                "evidence_refs": ["evidence_1"],
                "confidence": 0.9,
                "eligible_for_learning": True,
                "claims": [
                    {
                        "claim_key": "claim_api",
                        "conclusion": "stable",
                        "evidence_refs": ["evidence_1"],
                        "source_role": "researcher",
                        "independence_key": "lineage_a",
                        "fingerprint": "tampered",
                    }
                ],
                "claim_format": "v2_full",
                "claim_fingerprints": ["tampered"],
                "confidence_source": "outcome_event",
            }
        )
