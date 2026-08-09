from __future__ import annotations

from pathlib import Path

from bauhinia_agent.evolution.candidate_artifacts import CandidateArtifactLifecycle
from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    EvaluationComparisonCompletedPayload,
    EvoEvent,
    EvoReferences,
    PromotionChangedPayload,
)
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.promotion import CandidateLifecycleService
from bauhinia_agent.evolution.store import EvoEventStore


def test_regression_rolls_back_to_previous_promoted_version_and_preserves_history(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    first, second = _artifact_pair(store)
    _mark_promoted(store, first)
    _mark_promoted(store, second)
    report_id = _regression_report(store, second)
    lifecycle = CandidateLifecycleService(store)
    assert lifecycle.active_promoted()[0].artifact_id == second[0]
    history_before = tuple(store.list_events())

    result = lifecycle.rollback_on_regression(
        second[0],
        report_id,
        impact_scope="project/provider",
        reason="Critical held-out risk regression detected.",
    )

    assert result.persisted is True
    assert result.promotion is not None
    assert result.promotion.payload.to_state == "Deprecated"
    assert result.promotion.payload.rollback_target == first[0]
    assert result.promotion.payload.extensions["rollback_sla"] == "immediate"
    assert result.promotion.payload.extensions["materialized_side_effects_reverted"] is False
    assert lifecycle.state(second[0]) is CandidateArtifactLifecycle.DEPRECATED
    assert lifecycle.state(first[0]) is CandidateArtifactLifecycle.PROMOTED
    assert lifecycle.active_promoted()[0].artifact_id == first[0]
    assert tuple(store.list_events())[: len(history_before)] == history_before
    assert any(event.refs.run_id == second[1] for event in store.list_events())


def test_regression_without_previous_promoted_version_disables_lineage(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    only = _artifact(store, version=1, lineage_id=None, supersedes=None)
    _mark_promoted(store, only)
    report_id = _regression_report(store, only, integrity=False)

    result = CandidateLifecycleService(store).rollback_on_regression(
        only[0],
        report_id,
        impact_scope="project",
        reason="Cost regression requires immediate logical disable.",
    )

    assert result.promotion is not None
    assert result.promotion.payload.rollback_target is None
    assert result.promotion.payload.extensions["rollback_sla"] == "logical-disable-now-review-within-24h"
    assert CandidateLifecycleService(store).active_promoted() == ()


def _artifact_pair(store: EvoEventStore) -> tuple[tuple[str, str], tuple[str, str]]:
    first = _artifact(store, version=1, lineage_id=None, supersedes=None)
    second = _artifact(store, version=2, lineage_id=first[0], supersedes=first[0])
    return first, second


def _artifact(
    store: EvoEventStore,
    *,
    version: int,
    lineage_id: str | None,
    supersedes: str | None,
) -> tuple[str, str]:
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
                lineage_id=lineage_id or artifact_id,
                artifact_version=version,
                kind="plan_template",
                name="rollback-provider-template",
                description=f"Rollback candidate version {version}.",
                instructions="Apply provider workflow and verify.",
                inputs=("provider task",),
                outputs=("evidence",),
                dependencies=("pytest",),
                effects=("read",),
                triggers=("provider",),
                scope="project",
                applicability="Provider repairs.",
                risks=("Requires monitoring.",),
                source_candidate_ids=(candidate_id,),
                support_candidate_ids=(candidate_id,),
                counterexample_candidate_ids=(),
                source_run_ids=(run_id,),
                evidence_refs=(new_evo_id("evidence"),),
                counterexamples=(),
                confidence=0.7,
                content_hash=("a" if version == 1 else "b") * 64,
                supersedes_artifact_id=supersedes,
            ),
        )
    )
    return artifact_id, run_id


def _mark_promoted(store: EvoEventStore, artifact: tuple[str, str]) -> None:
    artifact_id, run_id = artifact
    for source, target in (("Candidate", "Shadow"), ("Shadow", "Validated"), ("Validated", "Promoted")):
        store.append(
            EvoEvent(
                event_id=new_evo_id("event"),
                event_type="PromotionChanged",
                refs=EvoReferences(
                    run_id=run_id,
                    artifact_id=artifact_id,
                    promotion_id=new_evo_id("promotion"),
                ),
                payload=PromotionChangedPayload(
                    from_state=source,
                    to_state=target,
                    reason="Deterministic test lifecycle.",
                    reviewer="maintainer" if target == "Promoted" else None,
                ),
            )
        )


def _regression_report(
    store: EvoEventStore,
    artifact: tuple[str, str],
    *,
    integrity: bool = True,
) -> str:
    artifact_id, run_id = artifact
    report_id = new_evo_id("evaluation")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationComparisonCompleted",
            refs=EvoReferences(run_id=run_id, artifact_id=artifact_id, evaluation_id=report_id),
            payload=EvaluationComparisonCompletedPayload(
                report_id=report_id,
                artifact_id=artifact_id,
                artifact_version=2 if integrity else 1,
                corpus_id="corpus_regression",
                corpus_version="v2",
                evaluator_version="deterministic-v1+heldout-audit-v1",
                baseline_variant_id="eval_variant_baseline",
                candidate_variant_id="eval_variant_candidate",
                case_ids=tuple(f"eval_case_{index}" for index in range(5)),
                trial_event_ids=tuple(new_evo_id("event") for _ in range(20)),
                baseline_sample_count=10,
                candidate_sample_count=10,
                invalid_trial_count=0,
                minimum_repeats=2,
                baseline_success_rate=1.0,
                candidate_success_rate=0.5,
                baseline_verification_quality=1.0,
                candidate_verification_quality=0.5,
                baseline_cost=10.0,
                candidate_cost=20.0,
                baseline_latency_ms=100.0,
                candidate_latency_ms=200.0,
                baseline_risk_event_count=0,
                candidate_risk_event_count=1 if integrity else 0,
                uncertainty=0.2,
                eligible=False,
                blocking_reasons=("Candidate regressed after Promotion.",),
                integrity_violations=("Candidate skipped verification.",) if integrity else (),
            ),
        )
    )
    return report_id
