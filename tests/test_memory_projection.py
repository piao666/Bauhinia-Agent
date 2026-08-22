from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bauhinia_agent.evolution.events import (
    EvidenceRecordedPayload,
    EvoEvent,
    EvoReferences,
    MemoryCreatedPayload,
    MemoryLifecycleChangedPayload,
)
from bauhinia_agent.memory.models import MemoryProvenance, MemoryRecord, MemoryScope
from bauhinia_agent.memory.projection import (
    build_memory_projection as _build_memory_projection,
)

CREATED_AT = datetime(2026, 8, 11, tzinfo=UTC)
PROJECT_ID = "project_bauhinia"


def _record(
    memory_id: str,
    *,
    status: str = "active",
    layer: str = "semantic",
    project_id: str = PROJECT_ID,
    origin: str = "verified_evidence",
    user_id: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        layer=layer,  # type: ignore[arg-type]
        content=f"content for {memory_id}",
        scope=MemoryScope(
            project_id=project_id,
            session_id="session_1" if layer == "task" else None,
            user_id=user_id,
        ),
        provenance=MemoryProvenance(
            origin=origin,  # type: ignore[arg-type]
            source_run_ids=("run_1",),
            source_event_ids=("event_source",),
            evidence_refs=("evidence_source",),
        ),
        confidence=0.8,
        created_at=CREATED_AT,
        expires_at=CREATED_AT + timedelta(days=1),
        status=status,  # type: ignore[arg-type]
    )


def _created(record: MemoryRecord, event_id: str) -> EvoEvent[MemoryCreatedPayload]:
    return EvoEvent(
        event_id=event_id,
        event_type="MemoryCreated",
        refs=EvoReferences(run_id="run_1", memory_id=record.memory_id),
        payload=MemoryCreatedPayload(
            memory_type=record.layer,
            content=record.content,
            scope="session" if record.scope.session_id is not None else "project",
            confidence=record.confidence,
            source_event_ids=record.provenance.source_event_ids,
            extensions={"memory_record": record.to_dict()},
        ),
    )


def _evidence(
    event_id: str,
    evidence_id: str,
    *,
    run_id: str = "run_1",
    evidence_type: str = "test",
    verified: bool = True,
) -> EvoEvent[EvidenceRecordedPayload]:
    is_confirmation = evidence_type == "user_confirmation"
    return EvoEvent(
        event_id=event_id,
        event_type="EvidenceRecorded",
        refs=EvoReferences(run_id=run_id, evidence_id=evidence_id),
        payload=EvidenceRecordedPayload(
            evidence_type=evidence_type,
            source=("user_input_boundary" if is_confirmation else "memory projection test"),
            summary="verified lifecycle evidence",
            locator="confirmation_1" if is_confirmation else None,
            verified=verified,
            command="pytest -q" if evidence_type == "test" else None,
            input_summary=('{"session_id":"session_1","user_id":"user_1"}' if is_confirmation else None),
            exit_code=0 if evidence_type in {"test", "diff"} else None,
        ),
    )


def build_memory_projection(
    events: tuple[EvoEvent, ...] | list[EvoEvent],
    *,
    project_id: str,
):
    """All ordinary fixtures share one real, prior provenance Evidence fact."""

    return _build_memory_projection(
        (_evidence("event_source", "evidence_source"), *tuple(events)),
        project_id=project_id,
    )


def _lifecycle(
    event_id: str,
    action: str,
    memory_ids: tuple[str, ...],
    basis_event_ids: tuple[str, ...],
    *,
    project_id: str = PROJECT_ID,
    replacement_memory_id: str | None = None,
    proposal_memory_id: str | None = None,
    confirmed_by_user_id: str | None = None,
    evidence_refs: tuple[str, ...] = ("evidence_lifecycle",),
    run_id: str = "run_1",
    refs_memory_id: str | None = None,
) -> EvoEvent[MemoryLifecycleChangedPayload]:
    target_id = replacement_memory_id or proposal_memory_id or memory_ids[0]
    return EvoEvent(
        event_id=event_id,
        event_type="MemoryLifecycleChanged",
        refs=EvoReferences(run_id=run_id, memory_id=refs_memory_id or target_id),
        payload=MemoryLifecycleChangedPayload(
            lifecycle_schema_version="v1",
            change_id=f"memory_change_{event_id}",
            project_id=project_id,
            action=action,
            memory_ids=memory_ids,
            reason="verified lifecycle decision",
            evidence_refs=evidence_refs,
            actor_kind="user" if action == "confirm" else "system",
            actor_id="user_1" if action == "confirm" else "memory_service",
            basis_event_ids=basis_event_ids,
            replacement_memory_id=replacement_memory_id,
            proposal_memory_id=proposal_memory_id,
            confirmed_by_user_id=confirmed_by_user_id,
            extensions={"confirmation_session_id": ("session_1" if action == "confirm" else None)},
        ),
    )


def test_projection_reduces_supersede_and_keeps_inference_record_immutable() -> None:
    old = _record("memory_old")
    replacement = _record("memory_replacement", status="proposed")
    inference = _record("memory_inference", status="proposed", layer="task", origin="inference")
    events = (
        _created(old, "event_create_old"),
        _created(replacement, "event_create_replacement"),
        _created(inference, "event_create_inference"),
        _evidence("event_evidence_lifecycle", "evidence_lifecycle"),
        _lifecycle(
            "event_supersede",
            "supersede",
            (old.memory_id,),
            ("event_create_old", "event_create_replacement"),
            replacement_memory_id=replacement.memory_id,
        ),
        _evidence(
            "event_evidence_user_confirmation",
            "evidence_user_confirmation",
            evidence_type="user_confirmation",
        ),
        _lifecycle(
            "event_confirm_inference",
            "confirm",
            (inference.memory_id,),
            ("event_create_inference",),
            confirmed_by_user_id="user_1",
            evidence_refs=("evidence_user_confirmation",),
        ),
    )

    projection = build_memory_projection(events, project_id=PROJECT_ID)

    assert projection.diagnostics == ()
    assert tuple(entry.record.memory_id for entry in projection.entries) == (
        "memory_old",
        "memory_replacement",
        "memory_inference",
    )
    old_entry = projection.get("memory_old")
    replacement_entry = projection.get("memory_replacement")
    inference_entry = projection.get("memory_inference")
    assert old_entry is not None and replacement_entry is not None and inference_entry is not None
    assert old_entry.effective_status == "superseded"
    assert old_entry.replacement_memory_id == replacement.memory_id
    assert old_entry.lifecycle_event_ids == ("event_supersede",)
    assert not old_entry.retrieval_eligible
    assert replacement_entry.record.status == "proposed"
    assert replacement_entry.effective_status == "active"
    assert replacement_entry.retrieval_eligible
    assert inference_entry.record.status == "proposed"
    assert inference_entry.effective_status == "active"
    assert inference_entry.confirmed_by_user_id == "user_1"
    assert inference_entry.retrieval_eligible


def test_created_memory_with_missing_provenance_fails_closed() -> None:
    forged = _record("memory_forged_source")
    projection = _build_memory_projection(
        (_created(forged, "event_create_forged_source"),),
        project_id=PROJECT_ID,
    )

    assert projection.get(forged.memory_id) is None
    assert [item.code for item in projection.diagnostics] == ["invalid_memory_provenance"]


def test_stale_basis_is_diagnosed_without_partial_state_change() -> None:
    record = _record("memory_current")
    projection = build_memory_projection(
        (
            _created(record, "event_create_current"),
            _evidence("event_evidence_lifecycle", "evidence_lifecycle"),
            _lifecycle(
                "event_stale_invalidate",
                "invalidate",
                (record.memory_id,),
                ("event_not_current",),
            ),
        ),
        project_id=PROJECT_ID,
    )

    entry = projection.get(record.memory_id)
    assert entry is not None
    assert entry.effective_status == "active"
    assert entry.latest_state_event_id == "event_create_current"
    assert entry.lifecycle_event_ids == ()
    assert not entry.retrieval_eligible
    assert [item.code for item in projection.diagnostics] == ["stale_lifecycle_basis"]
    assert entry.diagnostics == projection.diagnostics


def test_lifecycle_rejects_future_unverified_cross_run_evidence_and_wrong_target_ref() -> None:
    record = _record("memory_evidence_guard")
    created = _created(record, "event_create_evidence_guard")
    lifecycle = _lifecycle(
        "event_invalidate_evidence_guard",
        "invalidate",
        (record.memory_id,),
        ("event_create_evidence_guard",),
    )
    cases = (
        (
            (created, lifecycle, _evidence("event_evidence_future", "evidence_lifecycle")),
            "invalid_lifecycle_evidence",
        ),
        (
            (
                created,
                _evidence("event_evidence_unverified", "evidence_lifecycle", verified=False),
                lifecycle,
            ),
            "invalid_lifecycle_evidence",
        ),
        (
            (
                created,
                _evidence("event_evidence_other_run", "evidence_lifecycle", run_id="run_other"),
                lifecycle,
            ),
            "invalid_lifecycle_evidence",
        ),
        (
            (
                created,
                _evidence("event_evidence_lifecycle", "evidence_lifecycle"),
                _lifecycle(
                    "event_wrong_target_ref",
                    "invalidate",
                    (record.memory_id,),
                    ("event_create_evidence_guard",),
                    refs_memory_id="memory_wrong_target",
                ),
            ),
            "lifecycle_memory_reference_mismatch",
        ),
    )

    for events, expected_code in cases:
        projection = build_memory_projection(events, project_id=PROJECT_ID)
        entry = projection.get(record.memory_id)
        assert entry is not None
        assert entry.effective_status == "active"
        assert entry.latest_state_event_id == "event_create_evidence_guard"
        assert [item.code for item in projection.diagnostics] == [expected_code]
        assert not entry.retrieval_eligible


def test_confirm_requires_prior_verified_user_confirmation_evidence() -> None:
    proposal = _record("memory_confirmation_guard", status="proposed")
    projection = build_memory_projection(
        (
            _created(proposal, "event_create_confirmation_guard"),
            _evidence("event_evidence_not_confirmation", "evidence_lifecycle"),
            _lifecycle(
                "event_confirm_without_user_evidence",
                "confirm",
                (proposal.memory_id,),
                ("event_create_confirmation_guard",),
                confirmed_by_user_id="user_1",
            ),
        ),
        project_id=PROJECT_ID,
    )

    entry = projection.get(proposal.memory_id)
    assert entry is not None
    assert entry.record.status == "proposed"
    assert entry.effective_status == "proposed"
    assert entry.confirmed_by_user_id is None
    assert [item.code for item in projection.diagnostics] == ["missing_user_confirmation_evidence"]


def test_dangling_duplicate_direct_state_and_cross_project_facts_fail_closed() -> None:
    current = _record("memory_current")
    direct_invalid = _record("memory_direct_invalid", status="invalidated")
    other_project = _record("memory_other_project", project_id="project_other")
    projection = build_memory_projection(
        (
            _created(current, "event_create_current"),
            _created(current, "event_duplicate_current"),
            _created(direct_invalid, "event_create_direct_invalid"),
            _created(other_project, "event_create_other_project"),
            _evidence("event_evidence_lifecycle", "evidence_lifecycle"),
            _lifecycle(
                "event_dangling",
                "invalidate",
                ("memory_missing",),
                ("event_missing",),
            ),
        ),
        project_id=PROJECT_ID,
    )

    assert tuple(entry.record.memory_id for entry in projection.entries) == (
        "memory_current",
        "memory_direct_invalid",
    )
    assert [item.code for item in projection.diagnostics] == [
        "duplicate_memory_create",
        "invalid_initial_status",
        "cross_project_memory",
        "dangling_memory_reference",
    ]
    assert projection.get("memory_current") is not None
    assert not projection.get("memory_current").retrieval_eligible  # type: ignore[union-attr]
    assert projection.get("memory_direct_invalid") is not None
    assert not projection.get("memory_direct_invalid").retrieval_eligible  # type: ignore[union-attr]
    assert projection.get("memory_other_project") is None


def test_projection_replay_is_deterministic() -> None:
    record = _record("memory_replay")
    events = (
        _created(record, "event_create_replay"),
        _evidence("event_evidence_lifecycle", "evidence_lifecycle"),
        _lifecycle(
            "event_invalidate_replay",
            "invalidate",
            (record.memory_id,),
            ("event_create_replay",),
        ),
    )

    assert build_memory_projection(events, project_id=PROJECT_ID) == build_memory_projection(
        iter(events),
        project_id=PROJECT_ID,
    )


def test_confirm_pending_merge_supersedes_sources_and_activates_target() -> None:
    source_a = _record("memory_source_a")
    source_b = _record("memory_source_b")
    proposal = _record("memory_merge_proposal", status="proposed")
    events = (
        _created(source_a, "event_create_source_a"),
        _created(source_b, "event_create_source_b"),
        _created(proposal, "event_create_merge_proposal"),
        _evidence("event_evidence_lifecycle", "evidence_lifecycle"),
        _lifecycle(
            "event_propose_merge",
            "propose_merge",
            (source_a.memory_id, source_b.memory_id),
            ("event_create_source_a", "event_create_source_b", "event_create_merge_proposal"),
            proposal_memory_id=proposal.memory_id,
        ),
        _evidence(
            "event_evidence_user_confirmation",
            "evidence_user_confirmation",
            evidence_type="user_confirmation",
        ),
        _lifecycle(
            "event_confirm_merge",
            "confirm",
            (proposal.memory_id,),
            ("event_propose_merge", "event_propose_merge", "event_propose_merge"),
            confirmed_by_user_id="user_1",
            evidence_refs=("evidence_user_confirmation",),
        ),
    )

    projection = build_memory_projection(events, project_id=PROJECT_ID)

    assert projection.diagnostics == ()
    source_a_entry = projection.get(source_a.memory_id)
    source_b_entry = projection.get(source_b.memory_id)
    proposal_entry = projection.get(proposal.memory_id)
    assert source_a_entry is not None and source_b_entry is not None and proposal_entry is not None
    assert source_a_entry.record.status == "active"
    assert source_a_entry.effective_status == "superseded"
    assert source_a_entry.replacement_memory_id == proposal.memory_id
    assert source_b_entry.effective_status == "superseded"
    assert source_b_entry.replacement_memory_id == proposal.memory_id
    assert proposal_entry.record.status == "proposed"
    assert proposal_entry.effective_status == "active"
    assert proposal_entry.merge_source_memory_ids == (source_a.memory_id, source_b.memory_id)
    assert proposal_entry.confirmed_by_user_id == "user_1"
    assert proposal_entry.lifecycle_event_ids == ("event_propose_merge", "event_confirm_merge")
    assert source_a_entry.lifecycle_event_ids == ("event_propose_merge", "event_confirm_merge")
    assert proposal_entry.retrieval_eligible


def test_pending_merge_confirmation_rejects_a_stale_source_basis() -> None:
    source_a = _record("memory_stale_source_a")
    source_b = _record("memory_stale_source_b")
    proposal = _record("memory_stale_proposal", status="proposed")
    events = (
        _created(source_a, "event_create_stale_source_a"),
        _created(source_b, "event_create_stale_source_b"),
        _created(proposal, "event_create_stale_proposal"),
        _evidence("event_evidence_lifecycle", "evidence_lifecycle"),
        _lifecycle(
            "event_propose_stale_merge",
            "propose_merge",
            (source_a.memory_id, source_b.memory_id),
            (
                "event_create_stale_source_a",
                "event_create_stale_source_b",
                "event_create_stale_proposal",
            ),
            proposal_memory_id=proposal.memory_id,
        ),
        _lifecycle(
            "event_change_merge_source",
            "invalidate",
            (source_a.memory_id,),
            ("event_propose_stale_merge",),
        ),
        _evidence(
            "event_evidence_user_confirmation",
            "evidence_user_confirmation",
            evidence_type="user_confirmation",
        ),
        _lifecycle(
            "event_confirm_stale_merge",
            "confirm",
            (proposal.memory_id,),
            (
                "event_propose_stale_merge",
                "event_propose_stale_merge",
                "event_propose_stale_merge",
            ),
            confirmed_by_user_id="user_1",
            evidence_refs=("evidence_user_confirmation",),
        ),
    )

    projection = build_memory_projection(events, project_id=PROJECT_ID)

    assert [item.code for item in projection.diagnostics] == ["stale_lifecycle_basis"]
    source_a_entry = projection.get(source_a.memory_id)
    source_b_entry = projection.get(source_b.memory_id)
    proposal_entry = projection.get(proposal.memory_id)
    assert source_a_entry is not None and source_b_entry is not None and proposal_entry is not None
    assert source_a_entry.effective_status == "invalidated"
    assert source_b_entry.effective_status == "active"
    assert proposal_entry.effective_status == "proposed"
    assert proposal_entry.confirmed_by_user_id is None
    assert "event_confirm_stale_merge" not in proposal_entry.lifecycle_event_ids
