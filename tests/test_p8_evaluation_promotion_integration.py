from __future__ import annotations

from pathlib import Path

from bauhinia_agent.evaluation import (
    EvalCase,
    EvalCorpusCase,
    EvalCorpusManifest,
    EvalCorpusRegistry,
    EvalObservation,
    EvalRunInput,
    EvalVariant,
    EvaluationComparisonSpec,
    HeldOutEvalHarness,
    hash_text,
    private_reference_hash,
)
from bauhinia_agent.evolution.artifact_shadow import ArtifactShadowService, ShadowTrialSpec
from bauhinia_agent.evolution.candidate_artifacts import (
    CandidateArtifactDraft,
    CandidateArtifactKind,
    CandidateArtifactLifecycle,
    CandidateArtifactRegistry,
)
from bauhinia_agent.evolution.candidate_review import CandidateReview, CandidateReviewService
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import EvoEvent, EvoReferences, ExperienceCandidateCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.promotion import (
    CandidateLifecycleService,
    PromotionGate,
    PromotionReviewerIdentity,
)
from bauhinia_agent.evolution.store import EvoEventStore


class _OwnerIdentityProvider:
    def current_reviewer(self) -> PromotionReviewerIdentity:
        return PromotionReviewerIdentity(
            subject="repository-owner",
            role="owner",
            authenticated_by="integration_test_auth",
        )


from bauhinia_agent.skills.discovery import discover_project_skills


def test_p8_gate_promotes_only_after_repeated_held_out_comparison_and_human_approval(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _reviewed_artifact(store)
    shadow_evidence = EvidenceAdapter(store).record(
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
    assert shadow_evidence.evidence is not None
    shadow = ArtifactShadowService(store).record_trial(
        ShadowTrialSpec(
            artifact_id=artifact_id,
            mode="shadow",
            task_input_hash="a" * 64,
            workspace_baseline_hash="b" * 64,
            environment_hash="c" * 64,
            baseline_summary="Baseline suggestion recorded.",
            candidate_summary="Candidate suggestion recorded without execution.",
            evidence_refs=(shadow_evidence.evidence.evidence_id,),
            passed=True,
        )
    )
    assert shadow.trial is not None
    lifecycle = CandidateLifecycleService(
        store,
        identity_provider=_OwnerIdentityProvider(),
    )
    assert lifecycle.start_shadow(
        artifact_id,
        reviewer="curator",
        reason="Begin controlled evaluation after Shadow evidence.",
    ).persisted

    manifest = _manifest()
    registered = EvalCorpusRegistry(store).register(manifest)
    assert registered.persisted
    baseline = EvalVariant(
        variant_id="eval_variant_gate_baseline",
        kind="baseline",
        model_config_hash="d" * 64,
        strategy_hash="e" * 64,
    )
    candidate = EvalVariant(
        variant_id="eval_variant_gate_candidate",
        kind="candidate",
        model_config_hash="d" * 64,
        strategy_hash="f" * 64,
        artifact_id=artifact_id,
        artifact_version=1,
    )
    evaluator = _GateEvaluator(store)
    held_out = HeldOutEvalHarness(store)
    trial_run_ids: list[str] = []
    for item in manifest.cases:
        for seed in (101, 202):
            for variant in (baseline, candidate):
                result = held_out.run(manifest, item.case.case_id, variant, evaluator, seed=seed)
                assert result.trial is not None
                assert result.trial.payload.evaluation_status == "completed"
                trial_run_ids.append(result.trial.run_id)

    gated = PromotionGate(store).evaluate_and_validate(
        EvaluationComparisonSpec(
            artifact_id=artifact_id,
            artifact_version=1,
            corpus_id=manifest.corpus_id,
            corpus_version=manifest.version,
            evaluator_version="gate-evaluator-v1+heldout-audit-v1",
            baseline_variant_id=baseline.variant_id,
            candidate_variant_id=candidate.variant_id,
        )
    )

    assert gated.validated is True
    assert gated.report is not None
    assert gated.report.payload.case_ids == tuple(sorted(item.case.case_id for item in manifest.cases))
    assert gated.report.payload.baseline_sample_count == 10
    assert gated.report.payload.candidate_sample_count == 10
    assert lifecycle.state(artifact_id) is CandidateArtifactLifecycle.VALIDATED
    assert lifecycle.active_promoted() == ()

    promoted = lifecycle.approve(
        artifact_id,
        gated.report.payload.report_id,
        reviewer="repository-owner",
        reviewer_role="owner",
        reason="Five independent held-out Cases with two fixed repeats passed all governed thresholds.",
    )

    assert promoted.persisted is True
    assert lifecycle.state(artifact_id) is CandidateArtifactLifecycle.PROMOTED
    assert lifecycle.active_promoted()[0].artifact_id == artifact_id
    assert all(lifecycle_event.payload.extensions.get("runtime_permissions_changed") is False for lifecycle_event in _promotion_events(store))
    assert all(event.event_type not in {"MemoryCreated", "ToolExecuted"} for event in store.list_events())
    assert not discover_project_skills(tmp_path).skills
    assert len(set(trial_run_ids)) == 20
    assert all(held_out_run_events(store, run_id) for run_id in trial_run_ids)
    serialized = (tmp_path / ".bauhinia-agent" / "evo" / "events.jsonl").read_text(encoding="utf-8")
    assert all(item.private_reference not in serialized for item in manifest.cases)


def held_out_run_events(store: EvoEventStore, run_id: str) -> tuple[EvoEvent, ...]:
    return tuple(event for event in store.list_events() if event.refs.run_id == run_id)


def _reviewed_artifact(store: EvoEventStore) -> str:
    run_id = new_evo_id("run")
    candidate_id = new_evo_id("candidate")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ExperienceCandidateCreated",
            refs=EvoReferences(run_id=run_id, candidate_id=candidate_id),
            payload=ExperienceCandidateCreatedPayload(
                kind="plan_template",
                summary="Use deterministic provider verification.",
                scope="project",
                applicability="Provider regression repairs.",
                confidence=0.4,
                source_event_ids=(new_evo_id("event"),),
                evidence_refs=(new_evo_id("evidence"),),
                source_run_ids=(run_id,),
            ),
        )
    )
    assert (
        CandidateReviewService(store)
        .review(
            candidate_id,
            CandidateReview(
                decision="accept",
                reviewer="curator",
                reason="Source evidence and scope are reviewable.",
            ),
        )
        .persisted
    )
    result = CandidateArtifactRegistry(store).create(
        CandidateArtifactDraft(
            kind=CandidateArtifactKind.PLAN_TEMPLATE,
            name="provider-held-out-gate",
            description="Provider workflow awaiting held-out validation.",
            instructions="Apply deterministic provider verification and preserve tests.",
            inputs=("provider task",),
            outputs=("verification evidence",),
            dependencies=("pytest",),
            effects=("read",),
            scope="project",
            applicability="Provider regression repairs.",
            risks=("Must not bypass held-out evaluation.",),
            source_candidate_ids=(candidate_id,),
            confidence=0.6,
            triggers=("provider regression",),
        )
    )
    assert result.artifact is not None
    return result.artifact.artifact_id


def _manifest() -> EvalCorpusManifest:
    cases: list[EvalCorpusCase] = []
    for index in range(5):
        public = f"Held-out provider task {index}."
        private = f"Private expected assertions {index}."
        case = EvalCase(
            case_id=f"eval_case_gate_{index}",
            corpus_id="corpus_p8_gate",
            corpus_version="v1",
            split="held_out",
            public_input=public,
            task_input_hash=hash_text(public),
            workspace_baseline_hash=hash_text(f"gate-workspace-{index}"),
            environment_hash="c" * 64,
        )
        cases.append(EvalCorpusCase(case, private, private_reference_hash(private)))
    return EvalCorpusManifest(
        corpus_id="corpus_p8_gate",
        version="v1",
        license_spdx="MIT",
        provenance="Repository-authored P8 Gate fixture.",
        cases=tuple(cases),
    )


def _promotion_events(store: EvoEventStore) -> tuple[EvoEvent, ...]:
    return tuple(event for event in store.list_events() if event.event_type == "PromotionChanged")


class _GateEvaluator:
    version = "gate-evaluator-v1"

    def __init__(self, store: EvoEventStore) -> None:
        self._evidence = EvidenceAdapter(store)

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        candidate = request.variant.kind == "candidate"
        recorded = self._evidence.record(
            EvidenceInput(
                run_id=request.run_id,
                evidence_type="test",
                source="pytest",
                summary="held-out verification passed" if candidate else "held-out verification failed",
                verified=True,
                command="pytest -q",
                exit_code=0 if candidate else 1,
            )
        )
        assert recorded.persisted and recorded.evidence is not None
        return EvalObservation(
            task_outcome="task_success" if candidate else "task_failure",
            verification_quality=1.0 if candidate else 0.9,
            cost=11.0 if candidate else 10.0,
            latency_ms=110.0 if candidate else 100.0,
            evidence_refs=(recorded.evidence.evidence_id,),
            verification_commands=("pytest -q",),
            verification_coverage=1.0,
            claimed_success=candidate,
            evidence_success=candidate,
        )
