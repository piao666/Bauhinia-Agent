from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bauhinia_agent.evolution import (
    EvaluationTrialRecordedPayload,
    EvidenceAdapter,
    EvidenceInput,
    EvoEvent,
    EvoEventStore,
    EvoReferences,
    OutcomeClassifier,
    new_evo_id,
)
from bauhinia_agent.self_model import ProfileSelector, SelfModelError, SelfModelService, TaskClassification


def _classification(
    *,
    project: str = "project_a",
    model: str = "a" * 64,
    evaluator: str = "eval-v1",
    environment: str = "b" * 64,
    language: str = "python",
    risk: str = "low",
) -> TaskClassification:
    return TaskClassification(
        project_id=project,
        model_config_hash=model,
        evaluator_version=evaluator,
        environment_hash=environment,
        language=language,
        repository_scale="medium",
        task_type="bugfix",
        tool_category="pytest",
        risk_level=risk,
    )


def _selector(classification: TaskClassification, **overrides: object) -> ProfileSelector:
    values: dict[str, object] = classification.to_dict()
    values.update(overrides)
    return ProfileSelector(**values)


def _trial(
    store: EvoEventStore,
    *,
    success: bool,
    model: str = "a" * 64,
    evaluator: str = "eval-v1",
    environment: str = "b" * 64,
    status: str = "completed",
    risk_events: tuple[str, ...] = (),
    dangling_evidence: bool = False,
) -> EvoEvent:
    trial_id = new_evo_id("eval_trial")
    run_id = new_evo_id("run")
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="passed" if success else "failed",
            verified=True,
            command="pytest -q",
            exit_code=0 if success else 1,
        )
    )
    assert evidence.evidence is not None
    event = EvoEvent(
        event_id=new_evo_id("event"),
        event_type="EvaluationTrialRecorded",
        refs=EvoReferences(run_id=run_id, evaluation_id=trial_id),
        payload=EvaluationTrialRecordedPayload(
            evaluation_schema_version="v1",
            trial_id=trial_id,
            trial_key=new_evo_id("event").removeprefix("event_").ljust(64, "0"),
            attempt=1,
            case_id=new_evo_id("eval_case"),
            corpus_id=new_evo_id("corpus"),
            corpus_version="v1",
            split="held_out",
            variant_id=new_evo_id("eval_variant"),
            variant_kind="baseline",
            artifact_id=None,
            artifact_version=None,
            evaluator_version=evaluator,
            seed=1,
            task_input_hash="c" * 64,
            workspace_baseline_hash="d" * 64,
            environment_hash=environment,
            model_config_hash=model,
            variant_hash="e" * 64,
            task_outcome="task_success" if success else "task_failure",
            evaluation_status=status,
            success=success if status == "completed" else None,
            verification_quality=1.0,
            cost=2.0,
            latency_ms=25.0,
            risk_events=risk_events,
            evidence_refs=(new_evo_id("evidence") if dangling_evidence else evidence.evidence.evidence_id,),
            verification_commands=("pytest -q",),
            verification_skipped=False,
            verification_coverage=1.0,
            claimed_success=success,
            evidence_success=success,
            output_truncated=False,
            accessed_resource_hashes=(),
        ),
    )
    return store.append(event).event


def test_low_samples_are_insufficient_and_scopes_do_not_merge(tmp_path) -> None:
    store = EvoEventStore(tmp_path)
    service = SelfModelService(
        store=store,
        project_id="project_a",
        clock=lambda: datetime(2026, 8, 9, tzinfo=UTC),
    )
    classification = _classification()
    for _ in range(4):
        source = _trial(store, success=True)
        assert service.record_observation(classification, source_event_id=source.event_id).persisted
    other_model = _classification(model="f" * 64)
    other_source = _trial(store, success=False, model="f" * 64)
    service.record_observation(other_model, source_event_id=other_source.event_id)

    profile = service.build_profile(_selector(classification))
    assert profile.sample_count == 4
    assert profile.status == "insufficient_data"
    assert profile.confidence == 0.0
    assert service.build_profile(_selector(other_model)).sample_count == 1
    assert (
        SelfModelService(store=store, project_id="project_b")
        .build_profile(
            ProfileSelector(
                project_id="project_b",
                model_config_hash="a" * 64,
                evaluator_version="eval-v1",
                environment_hash="b" * 64,
            )
        )
        .sample_count
        == 0
    )


def test_reliable_profile_has_nonzero_uncertainty_and_is_publishable(tmp_path) -> None:
    store = EvoEventStore(tmp_path)
    service = SelfModelService(store=store, project_id="project_a")
    classification = _classification()
    for _ in range(20):
        source = _trial(store, success=True)
        service.record_observation(classification, source_event_id=source.event_id)

    published = service.publish_profile(_selector(classification, verification_level="strong"))

    assert published.persisted is True
    assert published.profile is not None
    assert published.profile.status == "reliable"
    assert published.profile.uncertainty is not None and published.profile.uncertainty > 0
    assert published.profile.average_cost == 2.0
    assert published.profile.published_event_id is not None
    events = store.list_events()
    assert sum(event.event_type == "SelfModelUpdated" for event in events) == 1
    assert not any(event.event_type in {"MemoryCreated", "PromotionChanged"} for event in events)


def test_outcome_observation_requires_same_run_evidence_and_duplicate_is_not_counted(tmp_path) -> None:
    store = EvoEventStore(tmp_path)
    run_id = new_evo_id("run")
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="passed",
            exit_code=0,
            verified=True,
        )
    )
    outcome = OutcomeClassifier(store).classify(run_id)
    assert evidence.persisted and outcome.persisted and outcome.outcome is not None
    service = SelfModelService(store=store, project_id="project_a")
    classification = _classification(evaluator="outcome-v1")

    first = service.record_observation(classification, source_event_id=outcome.outcome.event_id)
    duplicate = service.record_observation(classification, source_event_id=outcome.outcome.event_id)

    assert first.persisted is True
    assert first.observation is not None and first.observation.verification_level == "strong"
    assert duplicate.persisted is False
    assert duplicate.diagnostic is not None and duplicate.diagnostic.code == "duplicate_observation"
    assert len(service.list_observations()) == 1


def test_invalid_trial_and_mismatched_model_are_rejected(tmp_path) -> None:
    store = EvoEventStore(tmp_path)
    service = SelfModelService(store=store, project_id="project_a")
    invalid = _trial(store, success=False, status="invalid")
    with pytest.raises(SelfModelError, match="completed"):
        service.record_observation(_classification(), source_event_id=invalid.event_id)

    valid = _trial(store, success=True)
    with pytest.raises(SelfModelError, match="model"):
        service.record_observation(_classification(model="f" * 64), source_event_id=valid.event_id)

    dangling = _trial(store, success=True, dangling_evidence=True)
    with pytest.raises(SelfModelError, match="Evidence is invalid"):
        service.record_observation(_classification(), source_event_id=dangling.event_id)
