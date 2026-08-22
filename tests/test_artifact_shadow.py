from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bauhinia_agent.evolution.artifact_shadow import (
    ArtifactControlRequest,
    ArtifactShadowError,
    ArtifactShadowService,
    ShadowTrialSpec,
)
from bauhinia_agent.evolution.candidate_artifacts import (
    CandidateArtifactDraft,
    CandidateArtifactKind,
    CandidateArtifactRecord,
    CandidateArtifactRegistry,
)
from bauhinia_agent.evolution.candidate_review import CandidateReview, CandidateReviewService
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.events import EvoEvent, EvoReferences, ExperienceCandidateCreatedPayload
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError


def test_shadow_suggestions_select_latest_version_but_never_enter_runtime(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    first, second = _artifact_pair(store)
    service = ArtifactShadowService(store)

    suggestions = service.list_suggestions()

    assert len(suggestions) == 1
    assert suggestions[0].artifact_id == second.artifact_id
    assert suggestions[0].artifact_version == 2
    assert suggestions[0].effect_risk == "high"
    assert service.list_for_runtime() == ()
    assert first.payload.lifecycle_state == second.payload.lifecycle_state == "Candidate"


def test_shadow_trial_records_redacted_observation_without_real_effects(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    _, artifact = _artifact_pair(store)
    protected = tmp_path / "core.py"
    protected.write_text("original\n", encoding="utf-8")

    result = ArtifactShadowService(store).record_trial(
        _trial(
            store,
            artifact.artifact_id,
            passed=True,
            baseline_summary="Baseline result.",
            candidate_summary="Authorization: Bearer shadow-secret",
        )
    )

    assert result.persisted is True
    assert result.trial is not None
    assert result.trial.payload.real_effects_applied is False
    assert result.trial.payload.extensions["execution_mode"] == "observe_only"
    assert protected.read_text(encoding="utf-8") == "original\n"
    assert all(event.event_type not in {"PromotionChanged", "MemoryCreated"} for event in store.list_events())
    serialized = (tmp_path / ".bauhinia-agent" / "evo" / "events.jsonl").read_text(encoding="utf-8")
    assert "shadow-secret" not in serialized


def test_failed_shadow_can_be_disabled_and_resumed_without_losing_trial_evidence(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    _, artifact = _artifact_pair(store)
    service = ArtifactShadowService(store)
    failed = service.record_trial(
        _trial(
            store,
            artifact.artifact_id,
            passed=False,
            baseline_summary="Baseline passed.",
            candidate_summary="Candidate failed.",
            failure_reason="Focused verification failed.",
        )
    )
    assert failed.trial is not None

    disabled = service.control(
        ArtifactControlRequest(
            artifact_id=artifact.artifact_id,
            action="disable_shadow",
            reviewer="curator",
            reason="Disable after failed Shadow evidence.",
            evidence_refs=failed.trial.payload.evidence_refs,
        )
    )

    assert disabled.persisted is True
    assert service.list_suggestions() == ()
    assert service.list_trials(artifact.artifact_id) == (failed.trial,)
    with pytest.raises(ArtifactShadowError, match="not enabled"):
        service.record_trial(_trial(store, artifact.artifact_id, passed=True))

    resumed = service.control(
        ArtifactControlRequest(
            artifact_id=artifact.artifact_id,
            action="resume_shadow",
            reviewer="curator",
            reason="Resume only for another controlled comparison.",
            evidence_refs=failed.trial.payload.evidence_refs,
        )
    )
    assert resumed.persisted is True
    assert service.list_suggestions()[0].artifact_id == artifact.artifact_id


def test_shadow_rollback_selects_prior_version_and_preserves_history(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    first, second = _artifact_pair(store)
    other, _ = _artifact_pair(store, name="other-shadow-policy")
    service = ArtifactShadowService(store)
    trial = service.record_trial(_trial(store, second.artifact_id, passed=False, failure_reason="Regression."))
    assert trial.trial is not None

    rollback = service.control(
        ArtifactControlRequest(
            artifact_id=second.artifact_id,
            action="rollback_shadow",
            reviewer="curator",
            reason="Return Shadow comparisons to the previous version.",
            evidence_refs=trial.trial.payload.evidence_refs,
            target_artifact_id=first.artifact_id,
        )
    )

    assert rollback.persisted is True
    selected = {suggestion.name: suggestion for suggestion in service.list_suggestions()}
    assert selected[first.payload.name].artifact_id == first.artifact_id
    assert service.list_trials(second.artifact_id) == (trial.trial,)
    with pytest.raises(ArtifactShadowError, match="same Artifact lineage"):
        service.control(
            ArtifactControlRequest(
                artifact_id=first.artifact_id,
                action="rollback_shadow",
                reviewer="curator",
                reason="Invalid cross-lineage rollback.",
                evidence_refs=trial.trial.payload.evidence_refs,
                target_artifact_id=other.artifact_id,
            )
        )


def test_shadow_validation_and_store_failure_are_safe(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    _, artifact = _artifact_pair(store)
    service = ArtifactShadowService(store)
    with pytest.raises(ArtifactShadowError, match="failure_reason"):
        service.record_trial(_trial(store, artifact.artifact_id, passed=False))
    with pytest.raises(ArtifactShadowError, match="hexadecimal digest"):
        service.record_trial(
            _trial(
                store,
                artifact.artifact_id,
                passed=True,
                task_input_hash="raw task input",
            )
        )

    valid_trial = _trial(store, artifact.artifact_id, passed=True)
    result = ArtifactShadowService(_FailingStore(store.list_events())).record_trial(valid_trial)
    assert result.persisted is False
    assert result.trial is None
    assert result.diagnostic is not None
    assert result.diagnostic.code == "shadow_trial_recording_failed"


def test_shadow_and_control_fail_closed_on_noncanonical_evidence(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    _, artifact = _artifact_pair(store)
    service = ArtifactShadowService(store)
    canonical = _trial(store, artifact.artifact_id, passed=True)

    with pytest.raises(ArtifactShadowError, match="does not exist"):
        service.record_trial(replace(canonical, evidence_refs=(new_evo_id("evidence"),)))
    with pytest.raises(ArtifactShadowError, match="conflicts"):
        service.record_trial(replace(canonical, passed=False, failure_reason="Forged failure."))
    with pytest.raises(ArtifactShadowError, match="does not exist"):
        service.control(
            ArtifactControlRequest(
                artifact_id=artifact.artifact_id,
                action="disable_shadow",
                reviewer="curator",
                reason="Forged control evidence.",
                evidence_refs=(new_evo_id("evidence"),),
            )
        )


def _artifact_pair(
    store: EvoEventStore,
    *,
    name: str = "provider-shadow-policy",
) -> tuple[CandidateArtifactRecord, CandidateArtifactRecord]:
    candidate_id = _accepted_candidate(store)
    registry = CandidateArtifactRegistry(store)
    first = registry.create(_draft(candidate_id, name=name, description="Version one.")).artifact
    assert first is not None
    second = registry.create(
        _draft(
            candidate_id,
            name=name,
            description="Version two with narrower scope guidance.",
            supersedes_artifact_id=first.artifact_id,
        )
    ).artifact
    assert second is not None
    return first, second


def _draft(
    candidate_id: str,
    *,
    name: str,
    description: str,
    supersedes_artifact_id: str | None = None,
) -> CandidateArtifactDraft:
    return CandidateArtifactDraft(
        kind=CandidateArtifactKind.TOOL_INVOCATION_POLICY,
        name=name,
        description=description,
        instructions="Compare the suggested tool sequence without executing it.",
        inputs=("task hash",),
        outputs=("Shadow observation",),
        dependencies=("verified evidence",),
        effects=("write", "execute"),
        scope="project",
        applicability="Provider adapter verification.",
        risks=("Must remain suggestion-only before promotion.",),
        source_candidate_ids=(candidate_id,),
        confidence=0.5,
        supersedes_artifact_id=supersedes_artifact_id,
    )


def _trial(
    store: EvoEventStore,
    artifact_id: str,
    *,
    passed: bool,
    baseline_summary: str = "Baseline observation.",
    candidate_summary: str = "Candidate observation.",
    failure_reason: str | None = None,
    task_input_hash: str = "a" * 64,
) -> ShadowTrialSpec:
    run_id = new_evo_id("run")
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="Shadow verification passed" if passed else "Shadow verification failed",
            verified=True,
            command="pytest -q",
            exit_code=0 if passed else 1,
        )
    )
    assert evidence.evidence is not None
    return ShadowTrialSpec(
        artifact_id=artifact_id,
        mode="shadow",
        task_input_hash=task_input_hash,
        workspace_baseline_hash="b" * 64,
        environment_hash="c" * 64,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        evidence_refs=(evidence.evidence.evidence_id,),
        passed=passed,
        failure_reason=failure_reason,
    )


def _accepted_candidate(store: EvoEventStore) -> str:
    candidate_id = new_evo_id("candidate")
    run_id = new_evo_id("run")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ExperienceCandidateCreated",
            refs=EvoReferences(run_id=run_id, candidate_id=candidate_id),
            payload=ExperienceCandidateCreatedPayload(
                kind="debug_hint",
                summary="Do not repeat a failed provider tool sequence.",
                scope="project",
                applicability="Provider adapter verification.",
                confidence=0.4,
                source_event_ids=(new_evo_id("event"),),
                evidence_refs=(new_evo_id("evidence"),),
                source_run_ids=(run_id,),
            ),
        )
    )
    CandidateReviewService(store).review(
        candidate_id,
        CandidateReview(decision="accept", reviewer="curator", reason="Source evidence is complete."),
    )
    return candidate_id


class _FailingStore:
    def __init__(self, events: list[EvoEvent]) -> None:
        self._events = events

    def list_events(self) -> list[EvoEvent]:
        return self._events

    def append(self, event: EvoEvent) -> object:
        raise EvoStoreError("store offline")
