from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bauhinia_agent.evaluation import (
    EvalCase,
    EvalCorpusCase,
    EvalCorpusManifest,
    EvalCorpusRegistry,
    EvalObservation,
    EvalRunInput,
    EvalVariant,
    EvaluationComparisonError,
    EvaluationComparisonService,
    EvaluationComparisonSpec,
    HeldOutEvalHarness,
    PromotionThresholds,
    hash_text,
    private_reference_hash,
)
from bauhinia_agent.evolution.artifact_shadow import ArtifactShadowService, ShadowTrialSpec
from bauhinia_agent.evolution.candidate_artifacts import CandidateArtifactLifecycle
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    CandidateShadowTrialRecordedPayload,
    EvidenceRecordedPayload,
    EvoEvent,
    EvoReferences,
)
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.promotion import (
    CandidateLifecycleService,
    PromotionError,
    PromotionGate,
    PromotionReviewerIdentity,
)
from bauhinia_agent.evolution.store import EvoEventStore


class _IdentityProvider:
    def __init__(self, subject: str, role: str) -> None:
        self._identity = PromotionReviewerIdentity(
            subject=subject,
            role=role,
            authenticated_by="test_session_auth",
        )

    def current_reviewer(self) -> PromotionReviewerIdentity:
        return self._identity


def test_eligible_held_out_comparison_validates_then_requires_human_promotion(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)
    lifecycle = CandidateLifecycleService(
        store,
        identity_provider=_IdentityProvider(
            "project-maintainer",
            "maintainer",
        ),
    )

    gated = PromotionGate(store).evaluate_and_validate(_spec(artifact_id))

    assert gated.validated is True
    assert gated.report is not None
    metrics = gated.report.payload
    assert metrics.eligible is True
    assert metrics.baseline_success_rate == 0.0
    assert metrics.candidate_success_rate == 1.0
    assert metrics.baseline_verification_quality == 0.9
    assert metrics.candidate_verification_quality == 1.0
    assert metrics.baseline_cost == 10.0
    assert metrics.candidate_cost == 11.0
    assert metrics.baseline_latency_ms == 100.0
    assert metrics.candidate_latency_ms == 110.0
    assert 0.0 < metrics.uncertainty <= 0.25
    assert lifecycle.state(artifact_id) is CandidateArtifactLifecycle.VALIDATED
    assert lifecycle.active_promoted() == ()
    with pytest.raises(PromotionError, match="authenticated"):
        CandidateLifecycleService(store).approve(
            artifact_id,
            metrics.report_id,
            reviewer="project-maintainer",
            reviewer_role="maintainer",
            reason="Caller-supplied identity is not sufficient.",
        )
    with pytest.raises(PromotionError, match="maintainer or owner"):
        CandidateLifecycleService(
            store,
            identity_provider=_IdentityProvider("observer", "viewer"),
        ).approve(
            artifact_id,
            metrics.report_id,
            reviewer="observer",
            reviewer_role="viewer",
            reason="Not authorized.",
        )

    approved = lifecycle.approve(
        artifact_id,
        metrics.report_id,
        reviewer="project-maintainer",
        reviewer_role="maintainer",
        reason="Held-out metrics and safety evidence meet the fixed gate.",
    )

    assert approved.persisted is True
    assert approved.promotion is not None
    assert approved.promotion.payload.extensions["permissions_changed"] is False
    assert approved.promotion.payload.extensions["materialized"] is False
    assert approved.promotion.payload.extensions["reviewer_authenticated_by"] == "test_session_auth"
    assert lifecycle.state(artifact_id) is CandidateArtifactLifecycle.PROMOTED
    assert lifecycle.active_promoted()[0].artifact_id == artifact_id


def test_low_sample_candidate_stays_in_shadow_with_separate_report(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=2)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)

    gated = PromotionGate(store).evaluate_and_validate(_spec(artifact_id))

    assert gated.validated is False
    assert gated.report is not None
    assert gated.report.payload.eligible is False
    assert any("At least 5" in reason for reason in gated.report.payload.blocking_reasons)
    assert CandidateLifecycleService(store).state(artifact_id) is CandidateArtifactLifecycle.SHADOW


def test_lifecycle_rejects_hand_appended_eligible_report_that_conflicts_with_trial_facts(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=2)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)
    factual = EvaluationComparisonService(store).compare(_spec(artifact_id)).report
    assert factual is not None
    assert factual.payload.eligible is False
    forged_report_id = new_evo_id("evaluation")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationComparisonCompleted",
            refs=EvoReferences(
                run_id=factual.run_id,
                artifact_id=artifact_id,
                evaluation_id=forged_report_id,
                parent_event_id=factual.event_id,
            ),
            payload=replace(
                factual.payload,
                report_id=forged_report_id,
                eligible=True,
                blocking_reasons=(),
            ),
        )
    )

    with pytest.raises(PromotionError, match="report facts"):
        CandidateLifecycleService(store).validate_from_report(
            artifact_id,
            forged_report_id,
        )


def test_lifecycle_rejects_eligible_report_reusing_trials_from_another_artifact(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    source_artifact_id = _artifact(store)
    target_artifact_id = _artifact(store)
    _start_shadow(store, source_artifact_id)
    _start_shadow(store, target_artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, source_artifact_id)
    source = EvaluationComparisonService(store).compare(_spec(source_artifact_id)).report
    assert source is not None
    assert source.payload.eligible is True
    forged_report_id = new_evo_id("evaluation")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationComparisonCompleted",
            refs=EvoReferences(
                run_id=source.run_id,
                artifact_id=target_artifact_id,
                evaluation_id=forged_report_id,
                parent_event_id=source.event_id,
            ),
            payload=replace(
                source.payload,
                report_id=forged_report_id,
                artifact_id=target_artifact_id,
            ),
        )
    )

    with pytest.raises(PromotionError, match="report facts"):
        CandidateLifecycleService(store).validate_from_report(
            target_artifact_id,
            forged_report_id,
        )


def test_lifecycle_rejects_stale_eligible_report_after_new_matching_trial(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)
    report = EvaluationComparisonService(store).compare(_spec(artifact_id)).report
    assert report is not None
    assert report.payload.eligible is True
    source = next(event for event in store.list_events() if event.event_type == "EvaluationTrialRecorded" and event.payload.variant_id == "eval_variant_candidate")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationTrialRecorded",
            refs=replace(
                source.refs,
                evaluation_id=new_evo_id("eval_trial"),
            ),
            payload=replace(
                source.payload,
                trial_id=new_evo_id("eval_trial"),
                trial_key="7" * 64,
            ),
        )
    )

    with pytest.raises(PromotionError, match="stale"):
        CandidateLifecycleService(store).validate_from_report(
            artifact_id,
            report.payload.report_id,
        )


def test_lifecycle_rejects_forged_report_using_trial_supported_only_by_future_evidence(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)
    factual = EvaluationComparisonService(store).compare(_spec(artifact_id)).report
    assert factual is not None
    assert factual.payload.eligible is True
    source = next(event for event in store.list_events() if event.event_type == "EvaluationTrialRecorded" and event.payload.variant_id == "eval_variant_candidate")
    late_run_id = new_evo_id("run")
    late_evidence_id = new_evo_id("evidence")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationTrialRecorded",
            refs=replace(
                source.refs,
                run_id=late_run_id,
                evaluation_id=new_evo_id("eval_trial"),
            ),
            payload=replace(
                source.payload,
                trial_id=new_evo_id("eval_trial"),
                trial_key="6" * 64,
                evidence_refs=(late_evidence_id,),
            ),
        )
    )
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvidenceRecorded",
            refs=EvoReferences(
                run_id=late_run_id,
                evidence_id=late_evidence_id,
            ),
            payload=EvidenceRecordedPayload(
                evidence_type="test",
                source="pytest",
                summary="retroactive pass",
                verified=True,
                command="pytest -q",
                exit_code=0,
            ),
        )
    )
    forged_report_id = new_evo_id("evaluation")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationComparisonCompleted",
            refs=EvoReferences(
                run_id=factual.run_id,
                artifact_id=artifact_id,
                evaluation_id=forged_report_id,
                parent_event_id=factual.event_id,
            ),
            payload=replace(factual.payload, report_id=forged_report_id),
        )
    )

    with pytest.raises(PromotionError, match="report facts"):
        CandidateLifecycleService(store).validate_from_report(
            artifact_id,
            forged_report_id,
        )


def test_promotion_comparison_rejects_weaker_than_governed_thresholds(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)

    with pytest.raises(EvaluationComparisonError, match="cannot be weaker"):
        EvaluationComparisonService(store).compare(
            EvaluationComparisonSpec(
                artifact_id=artifact_id,
                artifact_version=1,
                corpus_id="corpus_promotion_gate",
                corpus_version="v1",
                evaluator_version="deterministic-v1+heldout-audit-v1",
                baseline_variant_id="eval_variant_baseline",
                candidate_variant_id="eval_variant_candidate",
                thresholds=PromotionThresholds(minimum_cases=1),
            )
        )


def test_shadow_transition_rejects_retroactively_appended_evidence(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    late_run_id = new_evo_id("run")
    late_evidence_id = new_evo_id("evidence")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateShadowTrialRecorded",
            refs=EvoReferences(
                run_id=late_run_id,
                artifact_id=artifact_id,
            ),
            payload=CandidateShadowTrialRecordedPayload(
                trial_id=new_evo_id("shadow_trial"),
                artifact_id=artifact_id,
                artifact_version=1,
                mode="shadow",
                task_input_hash="a" * 64,
                workspace_baseline_hash="b" * 64,
                environment_hash="c" * 64,
                baseline_summary="Baseline suggestion.",
                candidate_summary="Candidate suggestion.",
                evidence_refs=(late_evidence_id,),
                passed=True,
                real_effects_applied=False,
            ),
        )
    )
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvidenceRecorded",
            refs=EvoReferences(
                run_id=late_run_id,
                evidence_id=late_evidence_id,
            ),
            payload=EvidenceRecordedPayload(
                evidence_type="test",
                source="pytest",
                summary="late shadow pass",
                verified=True,
                exit_code=0,
            ),
        )
    )

    with pytest.raises(PromotionError, match="requires at least one recorded"):
        CandidateLifecycleService(store).start_shadow(
            artifact_id,
            reviewer="curator",
            reason="Forged Trial must not enter Shadow.",
        )


def test_comparison_rejects_directly_appended_trial_with_dangling_evidence(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)
    source = next(event for event in store.list_events() if event.event_type == "EvaluationTrialRecorded" and event.payload.variant_id == "eval_variant_candidate")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationTrialRecorded",
            refs=replace(
                source.refs,
                run_id=new_evo_id("run"),
                evaluation_id=new_evo_id("eval_trial"),
            ),
            payload=replace(
                source.payload,
                trial_id=new_evo_id("eval_trial"),
                trial_key="9" * 64,
                evidence_refs=(new_evo_id("evidence"),),
                extensions={
                    **source.payload.extensions,
                    "evidence_integrity_valid": True,
                },
            ),
        )
    )

    report = EvaluationComparisonService(store).compare(_spec(artifact_id)).report

    assert report is not None
    assert report.payload.eligible is False
    assert report.payload.invalid_trial_count == 1
    assert any("does not exist" in violation for violation in report.payload.integrity_violations)


def test_comparison_rejects_trial_retroactively_supported_by_later_evidence(
    tmp_path: Path,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)
    source = next(event for event in store.list_events() if event.event_type == "EvaluationTrialRecorded" and event.payload.variant_id == "eval_variant_candidate")
    late_run_id = new_evo_id("run")
    late_evidence_id = new_evo_id("evidence")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationTrialRecorded",
            refs=replace(
                source.refs,
                run_id=late_run_id,
                evaluation_id=new_evo_id("eval_trial"),
            ),
            payload=replace(
                source.payload,
                trial_id=new_evo_id("eval_trial"),
                trial_key="8" * 64,
                evidence_refs=(late_evidence_id,),
                extensions={
                    **source.payload.extensions,
                    "evidence_integrity_valid": True,
                },
            ),
        )
    )
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvidenceRecorded",
            refs=EvoReferences(
                run_id=late_run_id,
                evidence_id=late_evidence_id,
            ),
            payload=EvidenceRecordedPayload(
                evidence_type="test",
                source="pytest",
                summary="late candidate pass",
                verified=True,
                command="pytest -q",
                exit_code=0,
            ),
        )
    )

    report = EvaluationComparisonService(store).compare(_spec(artifact_id)).report

    assert report is not None
    assert report.payload.eligible is False
    assert report.payload.invalid_trial_count == 1
    assert any("precede" in violation for violation in report.payload.integrity_violations)


@pytest.mark.parametrize("regression", ["cost", "risk"])
def test_cost_or_risk_regression_is_rejected(tmp_path: Path, regression: str) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id, regression=regression)

    gated = PromotionGate(store).evaluate_and_validate(_spec(artifact_id))

    assert gated.report is not None
    assert gated.report.payload.eligible is False
    expected = "cost" if regression == "cost" else "risk events"
    assert expected in " ".join(gated.report.payload.blocking_reasons).lower()
    assert CandidateLifecycleService(store).state(artifact_id) is CandidateArtifactLifecycle.REJECTED


@pytest.mark.parametrize("regression", ["skip", "coverage", "false_claim", "truncated"])
def test_reward_integrity_violation_is_reported_and_rejected(tmp_path: Path, regression: str) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id, regression=regression)

    gated = PromotionGate(store).evaluate_and_validate(_spec(artifact_id))

    assert gated.report is not None
    assert gated.report.payload.eligible is False
    assert gated.report.payload.integrity_violations
    assert gated.transition is not None
    assert gated.transition.payload.to_state == "Rejected"
    assert CandidateLifecycleService(store).state(artifact_id) is CandidateArtifactLifecycle.REJECTED


def _manifest(*, case_count: int) -> EvalCorpusManifest:
    cases: list[EvalCorpusCase] = []
    for index in range(case_count):
        public = f"Repair held-out provider regression {index}."
        private = f"Private expected verification result {index}."
        case = EvalCase(
            case_id=f"eval_case_provider_{index}",
            corpus_id="corpus_promotion_gate",
            corpus_version="v1",
            split="held_out",
            public_input=public,
            task_input_hash=hash_text(public),
            workspace_baseline_hash=hash_text(f"workspace-{index}"),
            environment_hash="c" * 64,
        )
        cases.append(EvalCorpusCase(case, private, private_reference_hash(private)))
    return EvalCorpusManifest(
        corpus_id="corpus_promotion_gate",
        version="v1",
        license_spdx="MIT",
        provenance="Repository-authored Promotion Gate fixtures.",
        cases=tuple(cases),
    )


def _run_trials(
    store: EvoEventStore,
    manifest: EvalCorpusManifest,
    artifact_id: str,
    *,
    regression: str | None = None,
) -> None:
    baseline = EvalVariant(
        variant_id="eval_variant_baseline",
        kind="baseline",
        model_config_hash="d" * 64,
        strategy_hash="e" * 64,
    )
    candidate = EvalVariant(
        variant_id="eval_variant_candidate",
        kind="candidate",
        model_config_hash="d" * 64,
        strategy_hash="f" * 64,
        artifact_id=artifact_id,
        artifact_version=1,
    )
    evaluator = _ComparisonEvaluator(store, regression=regression)
    held_out = HeldOutEvalHarness(store)
    for item in manifest.cases:
        for seed in (1, 2):
            assert held_out.run(manifest, item.case.case_id, baseline, evaluator, seed=seed).trial is not None
            assert held_out.run(manifest, item.case.case_id, candidate, evaluator, seed=seed).trial is not None


def _spec(artifact_id: str) -> EvaluationComparisonSpec:
    return EvaluationComparisonSpec(
        artifact_id=artifact_id,
        artifact_version=1,
        corpus_id="corpus_promotion_gate",
        corpus_version="v1",
        evaluator_version="deterministic-v1+heldout-audit-v1",
        baseline_variant_id="eval_variant_baseline",
        candidate_variant_id="eval_variant_candidate",
    )


def _artifact(store: EvoEventStore) -> str:
    artifact_id = new_evo_id("artifact")
    run_id = new_evo_id("run")
    candidate_id = new_evo_id("candidate")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateArtifactCreated",
            refs=EvoReferences(run_id=run_id, artifact_id=artifact_id),
            payload=CandidateArtifactCreatedPayload(
                artifact_schema_version="v1",
                lineage_id=artifact_id,
                artifact_version=1,
                kind="plan_template",
                name="provider-promotion-gate",
                description="Provider promotion candidate.",
                instructions="Apply verified provider workflow.",
                inputs=("provider task",),
                outputs=("evidence",),
                dependencies=("pytest",),
                effects=("read",),
                triggers=("provider",),
                scope="project",
                applicability="Provider repairs.",
                risks=("Requires held-out gate.",),
                source_candidate_ids=(candidate_id,),
                support_candidate_ids=(candidate_id,),
                counterexample_candidate_ids=(),
                source_run_ids=(run_id,),
                evidence_refs=(new_evo_id("evidence"),),
                counterexamples=(),
                confidence=0.7,
                content_hash="a" * 64,
            ),
        )
    )
    return artifact_id


def _start_shadow(store: EvoEventStore, artifact_id: str) -> None:
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=new_evo_id("run"),
            evidence_type="test",
            source="pytest",
            summary="Shadow verification passed",
            verified=True,
            command="pytest -q",
            exit_code=0,
        )
    )
    assert evidence.evidence is not None
    trial = ArtifactShadowService(store).record_trial(
        ShadowTrialSpec(
            artifact_id=artifact_id,
            mode="shadow",
            task_input_hash="a" * 64,
            workspace_baseline_hash="b" * 64,
            environment_hash="c" * 64,
            baseline_summary="Baseline suggestion.",
            candidate_summary="Candidate suggestion.",
            evidence_refs=(evidence.evidence.evidence_id,),
            passed=True,
        )
    )
    assert trial.persisted is True
    started = CandidateLifecycleService(store).start_shadow(
        artifact_id,
        reviewer="curator",
        reason="Begin controlled Shadow evaluation.",
    )
    assert started.persisted is True


class _ComparisonEvaluator:
    version = "deterministic-v1"

    def __init__(self, store: EvoEventStore, *, regression: str | None) -> None:
        self._store = store
        self.regression = regression

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        candidate = request.variant.kind == "candidate"
        command = None if candidate and self.regression == "skip" else "pytest -q"
        recorded = EvidenceAdapter(self._store).record(
            EvidenceInput(
                run_id=request.run_id,
                evidence_type="test",
                source="pytest",
                summary="candidate passed" if candidate else "baseline failed",
                verified=True,
                command=command,
                exit_code=0 if candidate else 1,
            )
        )
        assert recorded.evidence is not None
        return EvalObservation(
            task_outcome="task_success" if candidate else "task_failure",
            verification_quality=1.0 if candidate else 0.9,
            cost=20.0 if candidate and self.regression == "cost" else (11.0 if candidate else 10.0),
            latency_ms=110.0 if candidate else 100.0,
            risk_events=("unsafe write",) if candidate and self.regression == "risk" else (),
            evidence_refs=(recorded.evidence.evidence_id,),
            verification_commands=() if command is None else (command,),
            verification_skipped=candidate and self.regression == "skip",
            verification_coverage=0.2 if candidate and self.regression == "coverage" else 1.0,
            claimed_success=candidate,
            evidence_success=False if candidate and self.regression == "false_claim" else candidate,
            output_truncated=candidate and self.regression == "truncated",
        )
