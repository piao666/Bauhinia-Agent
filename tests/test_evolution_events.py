from __future__ import annotations

import json

import pytest

from bauhinia_agent.evolution.events import (
    DecisionRecordedPayload,
    EvoEvent,
    EvoEventError,
    EvoReferences,
    PlanCreatedPayload,
    UnknownEvoPayload,
)
from bauhinia_agent.evolution.identifiers import EvoIdentifierError, new_evo_id, require_evo_id


def _refs(**overrides: object) -> EvoReferences:
    values = {
        "run_id": "run_1",
        "session_id": "sess_1",
        "plan_id": "plan_1",
        "node_id": "node_1",
    }
    values.update(overrides)
    return EvoReferences(**values)


def test_event_json_is_stable_and_preserves_parent_relationship() -> None:
    event = EvoEvent(
        event_id="event_1",
        event_type="DecisionRecorded",
        refs=_refs(parent_event_id="event_0"),
        payload=DecisionRecordedPayload(
            subgoal="define event contract",
            evidence_refs=("evidence_1",),
            assumptions=("existing session events remain unchanged",),
            options_considered=("dataclasses", "provider-specific models"),
            selected_action="dataclasses",
            rationale_summary="The domain contract must not depend on a provider.",
            confidence=0.9,
            expected_observation="known events round-trip deterministically",
            verification_method="pytest",
        ),
        schema_version="v1",
        occurred_at="2026-08-01T00:00:00Z",
        sequence=2,
    )

    assert event.to_dict() == {
        "event_id": "event_1",
        "event_type": "DecisionRecorded",
        "schema_version": "v1",
        "occurred_at": "2026-08-01T00:00:00Z",
        "sequence": 2,
        "refs": {
            "run_id": "run_1",
            "session_id": "sess_1",
            "plan_id": "plan_1",
            "node_id": "node_1",
            "parent_event_id": "event_0",
        },
        "payload": {
            "subgoal": "define event contract",
            "evidence_refs": ["evidence_1"],
            "assumptions": ["existing session events remain unchanged"],
            "options_considered": ["dataclasses", "provider-specific models"],
            "selected_action": "dataclasses",
            "rationale_summary": "The domain contract must not depend on a provider.",
            "confidence": 0.9,
            "expected_observation": "known events round-trip deterministically",
            "verification_method": "pytest",
            "outcome": None,
            "next_decision": None,
        },
    }
    assert event.to_json() == json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert EvoEvent.from_json(event.to_json()) == event


def test_known_event_payloads_are_explicit_and_round_trip() -> None:
    payloads = {
        "PlanCreated": {"goal": "ship", "node_ids": ["node_1"]},
        "PlanNodeUpdated": {"status": "completed", "change_summary": "verified", "attempt": 1},
        "DecisionRecorded": {
            "subgoal": "choose",
            "evidence_refs": [],
            "assumptions": [],
            "options_considered": ["a"],
            "selected_action": "a",
            "rationale_summary": "evidence",
            "confidence": 0.5,
            "expected_observation": "ok",
            "verification_method": "test",
        },
        "EvidenceRecorded": {"evidence_type": "test", "source": "pytest", "summary": "passed"},
        "OutcomeClassified": {"outcome": "success", "category": "task", "summary": "done"},
        "MemoryCreated": {"memory_type": "semantic", "content": "fact", "scope": "project", "confidence": 0.8, "source_event_ids": ["event_1"]},
        "MemoryUsed": {"reason": "matched project scope"},
        "ExperienceCandidateCreated": {
            "kind": "plan_template",
            "scope": "project",
            "applicability": "provider tasks",
            "confidence": 0.7,
            "source_event_ids": ["event_1"],
            "evidence_refs": ["evidence_1"],
        },
        "CandidateMergeProposed": {
            "cluster_id": "merge_1",
            "scope": "project",
            "kind": "plan_template",
            "candidate_ids": ["candidate_1", "candidate_2"],
            "source_run_ids": ["run_1", "run_2"],
            "evidence_refs": ["evidence_1", "evidence_2"],
            "task_features": ["provider"],
            "similarity": 0.8,
            "proposal_summary": "review merge",
        },
        "CandidateConflictDetected": {
            "conflict_group_id": "conflict_1",
            "scope": "project",
            "candidate_ids": ["candidate_1", "candidate_2"],
            "conclusions": ["caution", "support"],
            "source_run_ids": ["run_1", "run_2"],
            "evidence_refs": ["evidence_1", "evidence_2"],
            "task_features": ["provider"],
            "similarity": 0.8,
            "summary": "review conflict",
        },
        "CandidateReviewRecorded": {
            "candidate_id": "candidate_1",
            "decision": "edit",
            "reviewer": "curator_1",
            "reason": "narrow scope",
            "scope": "branch",
            "ttl_seconds": 3600,
            "sensitivity": "internal",
        },
        "EvaluationCompleted": {"dataset": "held-out", "evaluator_version": "v1", "passed": True, "sample_count": 2},
        "PromotionChanged": {"from_state": "Candidate", "to_state": "Shadow", "reason": "start shadow"},
        "SelfModelUpdated": {
            "dimension": "python",
            "scope": "project",
            "sample_count": 2,
            "window_start": "2026-08-01T00:00:00Z",
            "window_end": "2026-08-01T01:00:00Z",
            "confidence": 0.4,
            "metrics": {"success_rate": 1.0},
        },
    }

    for event_type, payload in payloads.items():
        event = EvoEvent(
            event_id=f"event_{event_type}",
            event_type=event_type,
            refs=_refs(),
            payload=EvoEvent.from_dict(
                {
                    "event_id": f"event_{event_type}",
                    "event_type": event_type,
                    "schema_version": "v1",
                    "occurred_at": "2026-08-01T00:00:00Z",
                    "sequence": 1,
                    "refs": {"run_id": "run_1"},
                    "payload": payload,
                }
            ).payload,
            occurred_at="2026-08-01T00:00:00Z",
            sequence=1,
        )
        restored = EvoEvent.from_dict(event.to_dict())
        assert restored.event_type == event_type
        assert restored.is_known_type
        assert restored.payload.to_dict() == event.payload.to_dict()


def test_unknown_event_and_fields_are_retained() -> None:
    raw = {
        "event_id": "event_future",
        "event_type": "FutureEvent",
        "schema_version": "v9",
        "occurred_at": "2026-08-01T00:00:00Z",
        "sequence": 3,
        "refs": {"run_id": "run_1", "future_ref": {"key": "value"}},
        "payload": {"future_value": [1, {"safe": True}]},
        "future_envelope_field": "preserve me",
    }

    event = EvoEvent.from_dict(raw)

    assert isinstance(event.payload, UnknownEvoPayload)
    assert not event.is_known_type
    assert event.to_dict() == raw


def test_legacy_event_shape_is_read_without_losing_extensions() -> None:
    event = EvoEvent.from_dict(
        {
            "id": "event_legacy",
            "type": "PlanCreated",
            "created_at": "2026-08-01T00:00:00Z",
            "sequence": 1,
            "run_id": "run_1",
            "payload": {"goal": "read old event", "legacy_field": "keep"},
        }
    )

    assert event.schema_version == "v0"
    assert event.refs.run_id == "run_1"
    assert isinstance(event.payload, PlanCreatedPayload)
    assert event.payload.to_dict()["legacy_field"] == "keep"


def test_persisted_event_requires_sequence() -> None:
    event = EvoEvent(
        event_id="event_draft",
        event_type="PlanCreated",
        refs=_refs(),
        payload=PlanCreatedPayload(goal="draft"),
        occurred_at="2026-08-01T00:00:00Z",
    )

    with pytest.raises(EvoEventError, match="sequence"):
        event.validate_persisted()


@pytest.mark.parametrize(
    ("field", "value"),
    [("event_id", ""), ("event_id", "bad id"), ("schema_version", ""), ("occurred_at", "2026-08-01T00:00:00")],
)
def test_event_rejects_invalid_envelope_values(field: str, value: str) -> None:
    kwargs = {
        "event_id": "event_1",
        "event_type": "PlanCreated",
        "refs": _refs(),
        "payload": PlanCreatedPayload(goal="goal"),
        "occurred_at": "2026-08-01T00:00:00Z",
        "sequence": 1,
    }
    kwargs[field] = value

    with pytest.raises((EvoEventError, EvoIdentifierError)):
        EvoEvent(**kwargs)


def test_identifier_generation_and_validation() -> None:
    identifier = new_evo_id("run")

    assert identifier.startswith("run_")
    assert require_evo_id(identifier, field="run_id", kind="run") == identifier
    with pytest.raises(EvoIdentifierError):
        require_evo_id("bad id", field="run_id")
