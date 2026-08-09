from __future__ import annotations

from pathlib import Path

import pytest

from bauhinia_agent.evolution.artifact_compiler import ArtifactDerivationSpec, CandidateArtifactCompiler
from bauhinia_agent.evolution.candidate_artifacts import CandidateArtifactKind
from bauhinia_agent.evolution.candidate_review import CandidateReview, CandidateReviewService
from bauhinia_agent.evolution.events import EvoEvent, EvoReferences, ExperienceCandidateCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore


def test_single_success_cannot_create_artifact(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    support = _candidate(store, index=1, kind="plan_template", outcome="task_success")
    counterexample = _candidate(store, index=9, kind="anti_pattern", outcome="verification_failure")

    result = CandidateArtifactCompiler(store).derive(_spec(CandidateArtifactKind.PLAN_TEMPLATE, (support,), (counterexample,)))

    assert result.persisted is False
    assert result.artifact is None
    assert result.diagnostic is not None
    assert result.diagnostic.code == "insufficient_support"
    assert all(event.event_type != "CandidateArtifactCreated" for event in store.list_events())


@pytest.mark.parametrize(
    ("artifact_kind", "source_kind"),
    [
        (CandidateArtifactKind.PLAN_TEMPLATE, "plan_template"),
        (CandidateArtifactKind.SKILL_DRAFT, "plan_template"),
        (CandidateArtifactKind.MEMORY_RULE, "stable_fact"),
    ],
)
def test_three_independent_successes_and_failure_counterexample_create_reviewable_artifact(tmp_path: Path, artifact_kind: CandidateArtifactKind, source_kind: str) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    supports = tuple(_candidate(store, index=index, kind=source_kind, outcome="task_success") for index in range(3))
    counterexample = _candidate(store, index=9, kind="anti_pattern", outcome="verification_failure")
    compiler = CandidateArtifactCompiler(store)
    spec = _spec(artifact_kind, supports, (counterexample,))

    result = compiler.derive(spec)

    assert result.persisted is True
    assert result.artifact is not None
    assert result.artifact.payload.support_candidate_ids == supports
    assert result.artifact.payload.counterexample_candidate_ids == (counterexample,)
    assert len(result.artifact.payload.source_run_ids) == 4
    assert len(result.artifact.payload.evidence_refs) == 4
    assert result.artifact.payload.confidence == pytest.approx(0.6)
    assert result.artifact.payload.lifecycle_state == "Candidate"
    assert any("Shadow" in risk for risk in result.artifact.payload.risks)
    assert all(event.event_type not in {"PromotionChanged", "MemoryCreated"} for event in store.list_events())

    repeated = compiler.derive(spec)
    assert repeated.persisted is False
    assert repeated.artifact == result.artifact
    assert repeated.diagnostic is not None
    assert repeated.diagnostic.code == "already_derived"
    assert sum(event.event_type == "CandidateArtifactCreated" for event in store.list_events()) == 1


def test_repeated_failures_create_tool_policy_without_promotion(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    supports = tuple(_candidate(store, index=index, kind="debug_hint", outcome="verification_failure") for index in range(3))

    result = CandidateArtifactCompiler(store).derive(_spec(CandidateArtifactKind.TOOL_INVOCATION_POLICY, supports, (), effects=("execute",)))

    assert result.persisted is True
    assert result.artifact is not None
    assert result.artifact.payload.kind == "tool_invocation_policy"
    assert result.artifact.payload.counterexample_candidate_ids == ()
    assert result.artifact.payload.extensions["runtime_enabled"] is False


@pytest.mark.parametrize(
    ("duplicate_field", "expected_code"),
    [("task", "insufficient_task_diversity"), ("evidence", "dependent_evidence"), ("run", "dependent_runs")],
)
def test_derivation_rejects_non_independent_support(tmp_path: Path, duplicate_field: str, expected_code: str) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    shared_evidence = new_evo_id("evidence")
    shared_run = new_evo_id("run")
    supports = tuple(
        _candidate(
            store,
            index=index,
            kind="plan_template",
            outcome="task_success",
            task_signature="same-task" if duplicate_field == "task" else None,
            evidence_ref=shared_evidence if duplicate_field == "evidence" else None,
            run_id=shared_run if duplicate_field == "run" else None,
        )
        for index in range(3)
    )
    counterexample = _candidate(store, index=9, kind="anti_pattern", outcome="verification_failure")

    result = CandidateArtifactCompiler(store).derive(_spec(CandidateArtifactKind.PLAN_TEMPLATE, supports, (counterexample,)))

    assert result.persisted is False
    assert result.diagnostic is not None
    assert result.diagnostic.code == expected_code
    assert all(event.event_type != "CandidateArtifactCreated" for event in store.list_events())


@pytest.mark.parametrize(
    ("duplicate_field", "expected_code"),
    [("evidence", "dependent_evidence"), ("run", "dependent_runs")],
)
def test_derivation_rejects_counterexample_that_reuses_support_trace(
    tmp_path: Path,
    duplicate_field: str,
    expected_code: str,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    shared_evidence = new_evo_id("evidence")
    shared_run = new_evo_id("run")
    supports = tuple(
        _candidate(
            store,
            index=index,
            kind="plan_template",
            outcome="task_success",
            evidence_ref=shared_evidence if index == 0 and duplicate_field == "evidence" else None,
            run_id=shared_run if index == 0 and duplicate_field == "run" else None,
        )
        for index in range(3)
    )
    counterexample = _candidate(
        store,
        index=9,
        kind="anti_pattern",
        outcome="verification_failure",
        evidence_ref=shared_evidence if duplicate_field == "evidence" else None,
        run_id=shared_run if duplicate_field == "run" else None,
    )

    result = CandidateArtifactCompiler(store).derive(_spec(CandidateArtifactKind.PLAN_TEMPLATE, supports, (counterexample,)))

    assert result.persisted is False
    assert result.diagnostic is not None
    assert result.diagnostic.code == expected_code
    assert all(event.event_type != "CandidateArtifactCreated" for event in store.list_events())


def test_derivation_requires_accepted_sources_and_matching_pattern(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    supports = tuple(_candidate(store, index=index, kind="plan_template", outcome="task_success") for index in range(3))
    counterexample = _candidate(
        store,
        index=9,
        kind="anti_pattern",
        outcome="verification_failure",
        pattern_key="different-pattern",
        accepted=False,
    )
    compiler = CandidateArtifactCompiler(store)

    unaccepted = compiler.derive(_spec(CandidateArtifactKind.PLAN_TEMPLATE, supports, (counterexample,)))
    assert unaccepted.diagnostic is not None
    assert unaccepted.diagnostic.code == "unaccepted_sources"

    CandidateReviewService(store).review(
        counterexample,
        CandidateReview(decision="accept", reviewer="curator", reason="Failure boundary is verified."),
    )
    mismatch = compiler.derive(_spec(CandidateArtifactKind.PLAN_TEMPLATE, supports, (counterexample,)))
    assert mismatch.diagnostic is not None
    assert mismatch.diagnostic.code == "pattern_mismatch"


def _spec(
    kind: CandidateArtifactKind,
    supports: tuple[str, ...],
    counterexamples: tuple[str, ...],
    *,
    effects: tuple[str, ...] = ("read",),
) -> ArtifactDerivationSpec:
    return ArtifactDerivationSpec(
        kind=kind,
        name=f"derived-{kind.value.replace('_', '-')}",
        support_candidate_ids=supports,
        counterexample_candidate_ids=counterexamples,
        inputs=("task context",),
        outputs=("reviewable recommendation",),
        dependencies=("verified evidence",),
        effects=effects,
        scope="project",
        triggers=("provider change",),
    )


def _candidate(
    store: EvoEventStore,
    *,
    index: int,
    kind: str,
    outcome: str,
    task_signature: str | None = None,
    pattern_key: str = "shared-provider-pattern",
    evidence_ref: str | None = None,
    run_id: str | None = None,
    accepted: bool = True,
) -> str:
    candidate_id = new_evo_id("candidate")
    actual_run_id = run_id or new_evo_id("run")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ExperienceCandidateCreated",
            refs=EvoReferences(run_id=actual_run_id, candidate_id=candidate_id),
            payload=ExperienceCandidateCreatedPayload(
                kind=kind,
                summary=f"Apply focused provider verification pattern {index}.",
                scope="project",
                applicability="Provider adapter changes.",
                confidence=0.4,
                source_event_ids=(new_evo_id("event"),),
                evidence_refs=(evidence_ref or new_evo_id("evidence"),),
                counterexamples=(f"Failure boundary {index}.",),
                source_run_ids=(actual_run_id,),
                extensions={
                    "outcome_category": outcome,
                    "task_signature": task_signature or f"task-{index}",
                    "pattern_key": pattern_key,
                },
            ),
        )
    )
    if accepted:
        CandidateReviewService(store).review(
            candidate_id,
            CandidateReview(decision="accept", reviewer="curator", reason="Evidence and task identity are verified."),
        )
    return candidate_id
