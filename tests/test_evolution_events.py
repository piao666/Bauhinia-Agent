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
        "CandidateArtifactCreated": {
            "artifact_schema_version": "v1",
            "lineage_id": "artifact_1",
            "artifact_version": 1,
            "kind": "plan_template",
            "name": "verify-provider-change",
            "description": "Verify provider changes.",
            "instructions": "Run focused tests.",
            "inputs": ["changed files"],
            "outputs": ["verification evidence"],
            "dependencies": ["pytest"],
            "effects": ["execute"],
            "triggers": ["provider change"],
            "scope": "project",
            "applicability": "Provider adapter changes.",
            "risks": ["Tests may be incomplete."],
            "source_candidate_ids": ["candidate_1"],
            "support_candidate_ids": ["candidate_1"],
            "counterexample_candidate_ids": [],
            "source_run_ids": ["run_1"],
            "evidence_refs": ["evidence_1"],
            "counterexamples": ["Do not use without tests."],
            "confidence": 0.6,
            "content_hash": "abc123",
        },
        "CandidateShadowTrialRecorded": {
            "trial_id": "shadow_trial_1",
            "artifact_id": "artifact_1",
            "artifact_version": 1,
            "mode": "shadow",
            "task_input_hash": "a" * 16,
            "workspace_baseline_hash": "b" * 16,
            "environment_hash": "c" * 16,
            "baseline_summary": "Baseline suggestion.",
            "candidate_summary": "Candidate suggestion.",
            "evidence_refs": ["evidence_1"],
            "passed": True,
            "real_effects_applied": False,
        },
        "CandidateArtifactControlChanged": {
            "artifact_id": "artifact_1",
            "artifact_version": 2,
            "action": "rollback_shadow",
            "reviewer": "curator",
            "reason": "Shadow regression.",
            "evidence_refs": ["evidence_1"],
            "target_artifact_id": "artifact_0",
            "target_artifact_version": 1,
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
        "EvaluationTrialRecorded": {
            "evaluation_schema_version": "v1",
            "trial_id": "eval_trial_1",
            "trial_key": "a" * 64,
            "attempt": 1,
            "case_id": "eval_case_1",
            "corpus_id": "corpus_1",
            "corpus_version": "v1",
            "split": "held_out",
            "variant_id": "eval_variant_1",
            "variant_kind": "baseline",
            "artifact_id": None,
            "artifact_version": None,
            "evaluator_version": "v1",
            "seed": 7,
            "task_input_hash": "b" * 64,
            "workspace_baseline_hash": "c" * 64,
            "environment_hash": "d" * 64,
            "model_config_hash": "e" * 64,
            "variant_hash": "f" * 64,
            "task_outcome": "task_success",
            "evaluation_status": "completed",
            "success": True,
            "verification_quality": 1.0,
            "cost": 1.0,
            "latency_ms": 10.0,
            "risk_events": [],
            "evidence_refs": ["evidence_1"],
            "verification_commands": ["pytest -q"],
            "verification_skipped": False,
            "verification_coverage": 1.0,
            "claimed_success": True,
            "evidence_success": True,
            "output_truncated": False,
            "accessed_resource_hashes": [],
            "invalid_reasons": [],
        },
        "EvaluationCorpusRegistered": {
            "corpus_schema_version": "v1",
            "corpus_id": "corpus_1",
            "corpus_version": "v1",
            "license_spdx": "MIT",
            "provenance": "Repository-authored fixtures.",
            "case_ids": ["eval_case_1"],
            "case_splits": ["held_out"],
            "task_input_hashes": ["a" * 64],
            "workspace_baseline_hashes": ["b" * 64],
            "environment_hashes": ["c" * 64],
            "private_reference_hashes": ["d" * 64],
            "case_manifest_hashes": ["e" * 64],
            "manifest_hash": "f" * 64,
        },
        "EvaluationComparisonCompleted": {
            "report_id": "evaluation_1",
            "artifact_id": "artifact_1",
            "artifact_version": 1,
            "corpus_id": "corpus_1",
            "corpus_version": "v1",
            "evaluator_version": "v1+heldout-audit-v1",
            "baseline_variant_id": "eval_variant_baseline",
            "candidate_variant_id": "eval_variant_candidate",
            "case_ids": ["eval_case_1"],
            "trial_event_ids": ["event_trial_1", "event_trial_2"],
            "baseline_sample_count": 2,
            "candidate_sample_count": 2,
            "invalid_trial_count": 0,
            "minimum_repeats": 2,
            "baseline_success_rate": 0.0,
            "candidate_success_rate": 1.0,
            "baseline_verification_quality": 0.9,
            "candidate_verification_quality": 1.0,
            "baseline_cost": 10.0,
            "candidate_cost": 11.0,
            "baseline_latency_ms": 100.0,
            "candidate_latency_ms": 110.0,
            "baseline_risk_event_count": 0,
            "candidate_risk_event_count": 0,
            "uncertainty": 0.0,
            "eligible": True,
            "blocking_reasons": [],
            "integrity_violations": [],
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
