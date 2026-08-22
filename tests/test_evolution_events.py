from __future__ import annotations

import json

import pytest

from bauhinia_agent.evolution.events import (
    ContextPackRecordedPayload,
    DecisionRecordedPayload,
    EvoEvent,
    EvoEventError,
    EvoReferences,
    MemoryLifecycleChangedPayload,
    MemoryUsedPayload,
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


def _memory_lifecycle_payload(action: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "lifecycle_schema_version": "v1",
        "change_id": f"change_{action}",
        "project_id": "project_1",
        "action": action,
        "memory_ids": ["memory_1"],
        "reason": f"Review requested {action}.",
        "evidence_refs": ["evidence_1"],
        "actor_kind": "system",
        "actor_id": "memory_service",
        "basis_event_ids": ["event_memory_1"],
    }
    if action == "supersede":
        payload["replacement_memory_id"] = "memory_2"
    elif action == "propose_merge":
        payload["memory_ids"] = ["memory_1", "memory_2"]
        payload["proposal_memory_id"] = "memory_3"
        payload["basis_event_ids"] = ["event_memory_1", "event_memory_2"]
    elif action == "confirm":
        payload["actor_kind"] = "user"
        payload["actor_id"] = "user_1"
        payload["confirmed_by_user_id"] = "user_1"
    return payload


def _context_pack_payload() -> dict[str, object]:
    return {
        "context_pack_schema_version": "v1",
        "context_pack_id": "context_pack_1",
        "query_signature_hash": "a" * 64,
        "token_budget": 128,
        "used_tokens": 96,
        "estimator_id": "deterministic_chars_v1",
        "selected_memory_ids": ["memory_1", "memory_2"],
        "selected_ranks": [1, 2],
        "selected_original_token_costs": [64, 80],
        "selected_packed_token_costs": [64, 32],
        "selected_truncated": [False, True],
        "selected_start_offsets": [0, 0],
        "selected_end_offsets": [64, 32],
        "omitted_memory_ids": ["memory_3"],
        "omitted_reasons": ["token_budget_exhausted"],
    }


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
        "MemoryLifecycleChanged": _memory_lifecycle_payload("invalidate"),
        "ContextPackRecorded": _context_pack_payload(),
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
        "SelfModelObservationRecorded": {
            "project_id": "project_1",
            "model_config_hash": "a" * 64,
            "evaluator_version": "eval-v1",
            "environment_hash": "b" * 64,
            "language": "python",
            "repository_scale": "medium",
            "task_type": "bugfix",
            "tool_category": "test",
            "risk_level": "low",
            "verification_level": "strong",
            "source_event_id": "event_source_1",
            "success": True,
            "outcome_category": "task_success",
            "verification_quality": 1.0,
            "cost": 1.0,
            "latency_ms": 10.0,
            "risk_event_count": 0,
            "evidence_refs": ["evidence_1"],
        },
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


@pytest.mark.parametrize("action", ["supersede", "invalidate", "propose_merge", "confirm"])
def test_memory_lifecycle_actions_round_trip(action: str) -> None:
    raw = {
        "event_id": f"event_{action}",
        "event_type": "MemoryLifecycleChanged",
        "schema_version": "v1",
        "occurred_at": "2026-08-01T00:00:00Z",
        "sequence": 1,
        "refs": {"run_id": "run_1", "memory_id": "memory_1"},
        "payload": _memory_lifecycle_payload(action),
    }

    event = EvoEvent.from_dict(raw)

    assert isinstance(event.payload, MemoryLifecycleChangedPayload)
    assert event.payload.action == action
    assert EvoEvent.from_json(event.to_json()) == event


def test_memory_lifecycle_unknown_payload_fields_are_retained() -> None:
    raw = _memory_lifecycle_payload("invalidate")
    raw["future_review_policy"] = {"minimum_approvals": 2}

    payload = MemoryLifecycleChangedPayload.from_dict(raw)

    assert payload.to_dict()["future_review_policy"] == {"minimum_approvals": 2}


def test_memory_lifecycle_basis_can_repeat_one_atomic_prior_event() -> None:
    raw = _memory_lifecycle_payload("confirm")
    raw["basis_event_ids"] = ["event_merge", "event_merge", "event_merge"]

    payload = MemoryLifecycleChangedPayload.from_dict(raw)

    assert payload.basis_event_ids == (
        "event_merge",
        "event_merge",
        "event_merge",
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("unknown_action", "unsupported memory lifecycle action"),
        ("supersede_missing_replacement", "supersede requires"),
        ("invalidate_with_replacement", "invalidate does not accept"),
        ("merge_with_one_source", "propose_merge requires"),
        ("confirm_without_user", "confirm requires"),
        ("system_confirm", "user or maintainer"),
        ("duplicate_sources", "must not contain duplicates"),
    ],
)
def test_memory_lifecycle_rejects_invalid_actions_and_relationships(case: str, match: str) -> None:
    raw = _memory_lifecycle_payload("invalidate")
    if case == "unknown_action":
        raw["action"] = "retire"
    elif case == "supersede_missing_replacement":
        raw["action"] = "supersede"
    elif case == "invalidate_with_replacement":
        raw["replacement_memory_id"] = "memory_2"
    elif case == "merge_with_one_source":
        raw["action"] = "propose_merge"
        raw["proposal_memory_id"] = "memory_2"
    elif case == "confirm_without_user":
        raw["action"] = "confirm"
        raw["actor_kind"] = "user"
        raw["actor_id"] = "user_1"
    elif case == "system_confirm":
        raw = _memory_lifecycle_payload("confirm")
        raw["actor_kind"] = "system"
    else:
        raw = _memory_lifecycle_payload("propose_merge")
        raw["memory_ids"] = ["memory_1", "memory_1"]

    with pytest.raises(EvoEventError, match=match):
        MemoryLifecycleChangedPayload.from_dict(raw)


def test_context_pack_event_and_reference_round_trip_preserve_unknown_fields() -> None:
    payload = _context_pack_payload()
    payload["future_packing_policy"] = {"preserve_headers": True}
    raw = {
        "event_id": "event_context_pack_1",
        "event_type": "ContextPackRecorded",
        "schema_version": "v1",
        "occurred_at": "2026-08-01T00:00:00Z",
        "sequence": 1,
        "refs": {"run_id": "run_1", "context_pack_id": "context_pack_1"},
        "payload": payload,
    }

    event = EvoEvent.from_dict(raw)

    assert isinstance(event.payload, ContextPackRecordedPayload)
    assert event.refs.context_pack_id == "context_pack_1"
    assert event.payload.extensions == {"future_packing_policy": {"preserve_headers": True}}
    assert EvoEvent.from_json(event.to_json()).to_dict() == raw


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("selected_lengths", "selected context-pack arrays"),
        ("omitted_lengths", "omitted_memory_ids and omitted_reasons"),
        ("over_budget", "must not exceed"),
        ("negative_cost", "non-negative integer"),
        ("invalid_boolean", "boolean"),
        ("invalid_memory_id", "whitespace"),
    ],
)
def test_context_pack_rejects_invalid_budget_and_parallel_arrays(case: str, match: str) -> None:
    raw = _context_pack_payload()
    if case == "selected_lengths":
        raw["selected_end_offsets"] = [64]
    elif case == "omitted_lengths":
        raw["omitted_memory_ids"] = ["memory_3", "memory_4"]
    elif case == "over_budget":
        raw["used_tokens"] = 129
    elif case == "negative_cost":
        raw["selected_packed_token_costs"] = [64, -1]
    elif case == "invalid_boolean":
        raw["selected_truncated"] = [False, 0]
    else:
        raw["selected_memory_ids"] = ["memory_1", "bad memory id"]

    with pytest.raises((EvoEventError, EvoIdentifierError), match=match):
        ContextPackRecordedPayload.from_dict(raw)


def test_memory_used_round_trip_binds_context_pack_and_verified_feedback() -> None:
    raw = {
        "reason": "Selected for the current plan node.",
        "retrieval_rank": 0,
        "helpfulness": "helpful",
        "context_pack_id": "context_pack_1",
        "usage_status": "used",
        "packed_token_cost": 32,
        "truncated": True,
        "outcome_event_id": "event_outcome_1",
        "verification_evidence_refs": ["evidence_verify_1"],
        "feedback_status": "helpful",
    }

    payload = MemoryUsedPayload.from_dict(raw)

    assert payload.to_dict() == raw
    assert MemoryUsedPayload.from_dict(payload.to_dict()) == payload


def test_legacy_memory_used_payload_defaults_to_unattributed_statuses() -> None:
    payload = MemoryUsedPayload.from_dict(
        {
            "reason": "Legacy retrieval event.",
            "retrieval_rank": 2,
            "helpfulness": "helpful",
        }
    )

    assert payload.usage_status == "legacy_unattributed"
    assert payload.feedback_status == "legacy_unattributed"
    assert payload.context_pack_id is None
    assert payload.outcome_event_id is None
    assert payload.verification_evidence_refs == ()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("usage_enum", "usage_status"),
        ("feedback_enum", "feedback_status"),
        ("used_without_pack", "requires context_pack_id"),
        ("not_used_without_pack", "requires context_pack_id"),
        ("feedback_without_outcome", "requires outcome_event_id"),
        ("feedback_without_evidence", "requires outcome_event_id"),
        ("negative_packed_cost", "non-negative integer"),
        ("invalid_truncated", "boolean"),
    ],
)
def test_memory_used_rejects_invalid_enums_and_attribution(case: str, match: str) -> None:
    raw: dict[str, object] = {
        "reason": "Context attribution.",
        "context_pack_id": "context_pack_1",
        "usage_status": "used",
        "truncated": False,
        "feedback_status": "unknown",
        "verification_evidence_refs": [],
    }
    if case == "usage_enum":
        raw["usage_status"] = "maybe"
    elif case == "feedback_enum":
        raw["feedback_status"] = "excellent"
    elif case == "used_without_pack":
        raw.pop("context_pack_id")
    elif case == "not_used_without_pack":
        raw["usage_status"] = "not_used"
        raw.pop("context_pack_id")
    elif case == "feedback_without_outcome":
        raw["feedback_status"] = "helpful"
        raw["verification_evidence_refs"] = ["evidence_verify_1"]
    elif case == "feedback_without_evidence":
        raw["feedback_status"] = "harmful"
        raw["outcome_event_id"] = "event_outcome_1"
    elif case == "negative_packed_cost":
        raw["packed_token_cost"] = -1
    else:
        raw["truncated"] = "false"

    with pytest.raises(EvoEventError, match=match):
        MemoryUsedPayload.from_dict(raw)


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
    context_pack_id = new_evo_id("context_pack")

    assert identifier.startswith("run_")
    assert context_pack_id.startswith("context_pack_")
    assert require_evo_id(identifier, field="run_id", kind="run") == identifier
    with pytest.raises(EvoIdentifierError):
        require_evo_id("bad id", field="run_id")
