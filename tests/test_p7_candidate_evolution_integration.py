from __future__ import annotations

from pathlib import Path

from bauhinia_agent.evolution.artifact_compiler import ArtifactDerivationSpec, CandidateArtifactCompiler
from bauhinia_agent.evolution.artifact_shadow import ArtifactControlRequest, ArtifactShadowService, ShadowTrialSpec
from bauhinia_agent.evolution.candidate_artifacts import (
    CandidateArtifactDraft,
    CandidateArtifactKind,
    CandidateArtifactRegistry,
)
from bauhinia_agent.evolution.candidate_review import CandidateReview, CandidateReviewService
from bauhinia_agent.evolution.compiler import ExperienceCompiler
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import DecisionRecordedPayload, EvoEvent, EvoReferences, PlanCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.outcomes import OutcomeClassifier
from bauhinia_agent.evolution.store import EvoEventStore


def test_verified_experience_becomes_versioned_shadow_only_skill_candidate(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    supports = tuple(_compile_reviewed_run(store, index=index, passed=True) for index in range(3))
    counterexample = _compile_reviewed_run(store, index=9, passed=False)

    derived = CandidateArtifactCompiler(store).derive(
        ArtifactDerivationSpec(
            kind=CandidateArtifactKind.SKILL_DRAFT,
            name="provider-verification-skill",
            support_candidate_ids=supports,
            counterexample_candidate_ids=(counterexample,),
            inputs=("provider task context",),
            outputs=("verification recommendation",),
            dependencies=("pytest",),
            effects=("execute",),
            scope="project",
            triggers=("provider adapter",),
        )
    )
    first = derived.artifact
    assert first is not None
    second = (
        CandidateArtifactRegistry(store)
        .create(
            CandidateArtifactDraft(
                kind=CandidateArtifactKind.SKILL_DRAFT,
                name=first.payload.name,
                description="Narrowed provider verification Skill candidate.",
                instructions=first.payload.instructions + "\n- Stop when permission is denied.",
                inputs=first.payload.inputs,
                outputs=first.payload.outputs,
                dependencies=first.payload.dependencies,
                effects=first.payload.effects,
                scope=first.payload.scope,
                applicability=first.payload.applicability,
                risks=first.payload.risks,
                source_candidate_ids=first.payload.source_candidate_ids,
                support_candidate_ids=first.payload.support_candidate_ids,
                counterexample_candidate_ids=first.payload.counterexample_candidate_ids,
                confidence=first.payload.confidence,
                triggers=first.payload.triggers,
                supersedes_artifact_id=first.artifact_id,
            )
        )
        .artifact
    )
    assert second is not None

    shadow = ArtifactShadowService(store)
    trial = shadow.record_trial(
        ShadowTrialSpec(
            artifact_id=second.artifact_id,
            mode="shadow",
            task_input_hash="a" * 64,
            workspace_baseline_hash="b" * 64,
            environment_hash="c" * 64,
            baseline_summary="Baseline verification passed.",
            candidate_summary="Candidate omitted one required check.",
            evidence_refs=(new_evo_id("evidence"),),
            passed=False,
            failure_reason="Shadow verification quality regressed.",
        )
    ).trial
    assert trial is not None
    shadow.control(
        ArtifactControlRequest(
            artifact_id=second.artifact_id,
            action="disable_shadow",
            reviewer="curator",
            reason="Disable the regressed version.",
            evidence_refs=trial.payload.evidence_refs,
        )
    )
    shadow.control(
        ArtifactControlRequest(
            artifact_id=second.artifact_id,
            action="rollback_shadow",
            reviewer="curator",
            reason="Use the prior version for future Shadow comparisons.",
            evidence_refs=trial.payload.evidence_refs,
            target_artifact_id=first.artifact_id,
        )
    )

    manifest = CandidateArtifactRegistry(store).manifest()
    assert manifest.versions(first.payload.lineage_id) == (first, second)
    assert shadow.list_suggestions()[0].artifact_id == first.artifact_id
    assert shadow.list_for_runtime() == ()
    assert CandidateReviewService(store).list_for_retrieval() == []
    assert all(artifact.payload.lifecycle_state == "Candidate" for artifact in manifest.artifacts)
    assert all(event.event_type not in {"PromotionChanged", "MemoryCreated", "MemoryUsed"} for event in store.list_events())
    assert not (tmp_path / ".agents").exists()
    assert len(OutcomeClassifier(store).list_for_run(store.list_events()[0].refs.run_id)) == 1


def _compile_reviewed_run(store: EvoEventStore, *, index: int, passed: bool) -> str:
    run_id = new_evo_id("run")
    plan_id = new_evo_id("plan")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="PlanCreated",
            refs=EvoReferences(run_id=run_id, plan_id=plan_id),
            payload=PlanCreatedPayload(goal=f"verify provider scenario {index}"),
        )
    )
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="DecisionRecorded",
            refs=EvoReferences(run_id=run_id, plan_id=plan_id),
            payload=DecisionRecordedPayload(
                subgoal="verify provider behavior",
                evidence_refs=(),
                assumptions=(),
                options_considered=("run focused tests",),
                selected_action="run focused provider tests",
                rationale_summary="deterministic provider evidence is required",
                confidence=0.8,
                expected_observation="focused tests pass",
                verification_method="pytest",
            ),
        )
    )
    EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="focused provider tests passed" if passed else "focused provider test failed",
            exit_code=0 if passed else 1,
            verified=True,
        )
    )
    OutcomeClassifier(store).classify(run_id)
    compiled = ExperienceCompiler(store).compile(run_id, environment_summary="Windows; Python 3.12")
    assert compiled.candidates
    candidate_id = compiled.candidates[0].candidate_id
    CandidateReviewService(store).review(
        candidate_id,
        CandidateReview(decision="accept", reviewer="curator", reason="Run evidence and task identity are complete."),
    )
    return candidate_id
