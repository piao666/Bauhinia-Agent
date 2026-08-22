from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bauhinia_agent.memory import (
    MEMORY_LAYER_RULES,
    MemoryModelError,
    MemoryLifecycleChange,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
)

CREATED_AT = datetime(2026, 8, 8, 0, 0, tzinfo=UTC)


def _scope(layer: str = "semantic") -> MemoryScope:
    return MemoryScope(project_id="project_bauhinia", session_id="session_1" if layer == "task" else None)


def _provenance(*, origin: str = "verified_evidence") -> MemoryProvenance:
    if origin == "user_confirmation":
        return MemoryProvenance(origin=origin, source_run_ids=("run_1",), source_event_ids=("event_user_confirmed",))  # type: ignore[arg-type]
    return MemoryProvenance(
        origin=origin,  # type: ignore[arg-type]
        source_run_ids=("run_1",),
        source_event_ids=("event_1",),
        evidence_refs=("evidence_1",),
    )


def _record(layer: str = "semantic", **overrides: object) -> MemoryRecord:
    values: dict[str, object] = {
        "memory_id": "memory_1",
        "layer": layer,
        "content": "The project uses an append-only Evo event store.",
        "scope": _scope(layer),
        "provenance": _provenance(),
        "confidence": 0.9,
        "created_at": CREATED_AT,
        "expires_at": CREATED_AT + timedelta(days=1),
    }
    values.update(overrides)
    return MemoryRecord(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("layer", ["task", "episodic", "semantic", "procedural", "meta"])
def test_persistable_memory_layers_have_explicit_rules(layer: str) -> None:
    record = _record(layer)

    assert record.rule is MEMORY_LAYER_RULES[layer]  # type: ignore[index]
    assert record.is_readable_by(project_id="project_bauhinia", session_id="session_1")


def test_memory_record_round_trips_with_utc_timestamps_and_scope() -> None:
    record = _record()

    assert record.to_dict() == {
        "memory_id": "memory_1",
        "layer": "semantic",
        "content": "The project uses an append-only Evo event store.",
        "scope": {"project_id": "project_bauhinia", "session_id": None, "user_id": None},
        "provenance": {
            "origin": "verified_evidence",
            "source_run_ids": ["run_1"],
            "source_event_ids": ["event_1"],
            "evidence_refs": ["evidence_1"],
        },
        "confidence": 0.9,
        "created_at": "2026-08-08T00:00:00Z",
        "expires_at": "2026-08-09T00:00:00Z",
        "conflict_group": None,
        "status": "active",
        "sensitivity": "internal",
    }
    assert MemoryRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: MemoryProvenance(origin="verified_evidence", source_run_ids=("run_1",), source_event_ids=("event_1",)), "requires evidence_refs"),
        (lambda: MemoryProvenance(origin="user_confirmation", source_run_ids=("run_1",), evidence_refs=("evidence_1",)), "requires source_event_ids"),
        (lambda: MemoryProvenance(origin="temporary_dialogue", source_run_ids=("run_1",), source_event_ids=("event_1",)), "temporary dialogue"),
    ],
)
def test_provenance_rejects_missing_evidence_or_temporary_dialogue(factory: object, match: str) -> None:
    with pytest.raises(MemoryModelError, match=match):
        factory()  # type: ignore[operator]


def test_memory_requires_source_scope_and_confidence() -> None:
    with pytest.raises(MemoryModelError, match="source event or evidence"):
        MemoryProvenance(origin="verified_evidence", source_run_ids=("run_1",))
    with pytest.raises(MemoryModelError, match="project_id"):
        MemoryScope(project_id="")
    with pytest.raises(MemoryModelError, match="confidence"):
        _record(confidence=None)


def test_task_memory_requires_session_scope_and_other_layers_reject_it() -> None:
    with pytest.raises(MemoryModelError, match="task memory requires session scope"):
        _record("task", scope=MemoryScope(project_id="project_bauhinia"))
    with pytest.raises(MemoryModelError, match="semantic memory must not declare session scope"):
        _record("semantic", scope=MemoryScope(project_id="project_bauhinia", session_id="session_1"))


def test_inference_stays_proposed_and_cannot_upgrade_to_semantic_fact() -> None:
    inference = _provenance(origin="inference")

    with pytest.raises(MemoryModelError, match="inference memory must remain proposed"):
        _record("task", provenance=inference)
    with pytest.raises(MemoryModelError, match="cannot write semantic"):
        _record("semantic", provenance=inference, status="proposed")


def test_lifetime_and_sensitivity_are_conservative() -> None:
    with pytest.raises(MemoryModelError, match="maximum lifetime"):
        _record(expires_at=CREATED_AT + timedelta(days=366))
    with pytest.raises(MemoryModelError, match="secrets must not be persisted"):
        _record(sensitivity="secret")


def test_memory_reads_are_project_isolated() -> None:
    task_memory = _record("task")
    semantic_memory = _record("semantic")

    assert task_memory.is_readable_by(project_id="project_bauhinia", session_id="session_1")
    assert not task_memory.is_readable_by(project_id="project_bauhinia", session_id="session_2")
    assert semantic_memory.is_readable_by(project_id="project_bauhinia")
    assert not semantic_memory.is_readable_by(project_id="another_project")


def test_user_scope_conflict_group_and_expiry_are_explicit() -> None:
    record = _record(scope=MemoryScope(project_id="project_bauhinia", user_id="user_1"), conflict_group="conflict_provider")

    assert record.is_readable_by(project_id="project_bauhinia", user_id="user_1")
    assert not record.is_readable_by(project_id="project_bauhinia", user_id="user_2")
    assert record.freshness_weight(at=CREATED_AT) == 1.0
    assert record.freshness_weight(at=CREATED_AT + timedelta(days=2)) == 0.0


@pytest.mark.parametrize(
    ("kind", "memory_ids", "kwargs", "match"),
    [
        ("supersede", ("memory_old",), {"replacement_memory_id": "memory_new"}, None),
        ("invalidate", ("memory_old",), {}, None),
        ("propose_merge", ("memory_a", "memory_b"), {"proposal_memory_id": "memory_proposal"}, None),
        ("confirm", ("memory_proposal",), {"confirmed_by_user_id": "user_1"}, None),
        ("propose_merge", ("memory_a",), {}, "propose_merge requires"),
        ("confirm", ("memory_proposal",), {}, "confirm requires"),
    ],
)
def test_lifecycle_changes_are_append_only_and_auditable(kind: str, memory_ids: tuple[str, ...], kwargs: dict[str, str], match: str | None) -> None:
    values = {
        "change_id": "event_memory_change",
        "kind": kind,
        "memory_ids": memory_ids,
        "occurred_at": CREATED_AT,
        "reason": "new verified evidence",
        "evidence_refs": ("evidence_1",),
        **kwargs,
    }
    if match is not None:
        with pytest.raises(MemoryModelError, match=match):
            MemoryLifecycleChange(**values)  # type: ignore[arg-type]
    else:
        change = MemoryLifecycleChange(**values)  # type: ignore[arg-type]
        assert change.memory_ids == memory_ids
        assert MemoryLifecycleChange.from_dict(change.to_dict()) == change
