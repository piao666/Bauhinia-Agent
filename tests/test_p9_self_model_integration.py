from __future__ import annotations

from bauhinia_agent.evaluation import EvalCase, EvalHarness, EvalObservation, EvalRunInput, EvalVariant, hash_text
from bauhinia_agent.evolution import EvidenceAdapter, EvidenceInput, EvoEventStore
from bauhinia_agent.self_model import PolicySuggestionEngine, ProfileSelector, SelfModelService, TaskClassification


class _Evaluator:
    version = "eval-v1"

    def __init__(self, store: EvoEventStore, *, success: bool, risk: bool = False) -> None:
        self._evidence = EvidenceAdapter(store)
        self._success = success
        self._risk = risk

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        recorded = self._evidence.record(
            EvidenceInput(
                run_id=request.run_id,
                evidence_type="test",
                source="pytest",
                summary="migration verification passed" if self._success else "migration verification failed",
                verified=True,
                command="pytest -q",
                exit_code=0 if self._success else 1,
            )
        )
        assert recorded.persisted and recorded.evidence is not None
        return EvalObservation(
            task_outcome="task_success" if self._success else "task_failure",
            verification_quality=1.0,
            cost=2.0,
            latency_ms=20.0,
            risk_events=("unsafe request rejected",) if self._risk else (),
            evidence_refs=(recorded.evidence.evidence_id,),
            verification_commands=("pytest -q",),
            claimed_success=self._success,
            evidence_success=self._success,
        )


def _case(index: int) -> EvalCase:
    return EvalCase(
        case_id=f"eval_case_{index}",
        corpus_id="corpus_p9",
        corpus_version="v1",
        split="held_out",
        public_input=f"task {index}",
        task_input_hash=hash_text(f"task {index}"),
        workspace_baseline_hash=hash_text("workspace"),
        environment_hash=hash_text("environment"),
    )


def test_p9_profile_and_policy_are_auditable_and_cannot_change_runtime_authority(tmp_path) -> None:
    store = EvoEventStore(tmp_path)
    harness = EvalHarness(store)
    variant = EvalVariant(
        variant_id="eval_variant_baseline",
        kind="baseline",
        model_config_hash=hash_text("model"),
        strategy_hash=hash_text("strategy"),
    )
    service = SelfModelService(store=store, project_id="project_a")
    classification = TaskClassification(
        project_id="project_a",
        model_config_hash=variant.model_config_hash,
        evaluator_version="eval-v1",
        environment_hash=hash_text("environment"),
        language="python",
        repository_scale="medium",
        task_type="migration",
        tool_category="pytest",
        risk_level="high",
    )
    for index in range(5):
        trial = harness.run(_case(index), variant, _Evaluator(store, success=False, risk=True), seed=index)
        assert trial.persisted and trial.trial is not None
        observed = service.record_observation(classification, source_event_id=trial.trial.event_id)
        assert observed.persisted

    selector = ProfileSelector(**classification.to_dict(), verification_level="strong")
    published = service.publish_profile(selector)
    assert published.persisted and published.profile is not None
    before_suggestions = tuple(store.list_events())

    suggestions = PolicySuggestionEngine().suggest(published.profile)
    replay = PolicySuggestionEngine().suggest(published.profile)

    assert published.profile.status == "unreliable"
    assert published.profile.sample_count == 5
    assert published.profile.confidence_high is not None and published.profile.confidence_high < 0.6
    assert suggestions == replay
    assert all(item.permission_effect == "none" for item in suggestions.suggestions)
    assert tuple(store.list_events()) == before_suggestions
    assert not any(event.event_type in {"MemoryCreated", "PromotionChanged", "CandidateArtifactControlChanged"} for event in store.list_events())
