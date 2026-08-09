from __future__ import annotations

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
from bauhinia_agent.evolution.events import CandidateArtifactCreatedPayload, EvoEvent, EvoReferences
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.promotion import CandidateLifecycleService, PromotionError, PromotionGate
from bauhinia_agent.evolution.store import EvoEventStore


def test_eligible_held_out_comparison_validates_then_requires_human_promotion(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    _start_shadow(store, artifact_id)
    manifest = _manifest(case_count=5)
    EvalCorpusRegistry(store).register(manifest)
    _run_trials(store, manifest, artifact_id)
    lifecycle = CandidateLifecycleService(store)

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
    with pytest.raises(PromotionError, match="maintainer or owner"):
        lifecycle.approve(
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
    evaluator = _ComparisonEvaluator(regression=regression)
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
    trial = ArtifactShadowService(store).record_trial(
        ShadowTrialSpec(
            artifact_id=artifact_id,
            mode="shadow",
            task_input_hash="a" * 64,
            workspace_baseline_hash="b" * 64,
            environment_hash="c" * 64,
            baseline_summary="Baseline suggestion.",
            candidate_summary="Candidate suggestion.",
            evidence_refs=(new_evo_id("evidence"),),
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

    def __init__(self, *, regression: str | None) -> None:
        self.regression = regression

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        candidate = request.variant.kind == "candidate"
        return EvalObservation(
            task_outcome="task_success" if candidate else "task_failure",
            verification_quality=1.0 if candidate else 0.9,
            cost=20.0 if candidate and self.regression == "cost" else (11.0 if candidate else 10.0),
            latency_ms=110.0 if candidate else 100.0,
            risk_events=("unsafe write",) if candidate and self.regression == "risk" else (),
            evidence_refs=(new_evo_id("evidence"),),
            verification_commands=() if candidate and self.regression == "skip" else ("pytest -q",),
            verification_skipped=candidate and self.regression == "skip",
            verification_coverage=0.2 if candidate and self.regression == "coverage" else 1.0,
            claimed_success=candidate,
            evidence_success=False if candidate and self.regression == "false_claim" else candidate,
            output_truncated=candidate and self.regression == "truncated",
        )
