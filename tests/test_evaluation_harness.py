from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bauhinia_agent.evaluation import (
    EvalCase,
    EvalHarness,
    EvalHarnessError,
    EvalObservation,
    EvalRunInput,
    EvalVariant,
    hash_text,
)
from bauhinia_agent.evolution.events import CandidateArtifactCreatedPayload, EvoEvent, EvoReferences
from bauhinia_agent.evolution.evidence import EvidenceAdapter, EvidenceInput
from bauhinia_agent.evolution.identifiers import new_evo_id
from bauhinia_agent.evolution.store import EvoEventStore, EvoStoreError


def test_same_fixed_trial_reruns_with_stable_key_and_standard_run(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    harness = EvalHarness(store)
    case = _case()
    variant = _baseline()
    evaluator = _Evaluator(
        store,
        EvalObservation(
            task_outcome="task_success",
            verification_quality=1.0,
            cost=2.0,
            latency_ms=30.0,
        ),
    )

    first = harness.run(case, variant, evaluator, seed=17)
    second = harness.run(case, variant, evaluator, seed=17)

    assert first.trial is not None and second.trial is not None
    assert first.trial.payload.trial_key == second.trial.payload.trial_key
    assert (first.trial.payload.attempt, second.trial.payload.attempt) == (1, 2)
    assert first.trial.run_id != second.trial.run_id
    assert harness.list_trials(trial_key=first.trial.payload.trial_key) == (first.trial, second.trial)
    first_run = harness.open_run(first.trial.run_id)
    assert tuple(event.event_type for event in first_run) == (
        "EvidenceRecorded",
        "EvaluationTrialRecorded",
    )
    assert first_run[-1].event_id == first.trial.event_id
    assert case.public_input not in (tmp_path / ".bauhinia-agent" / "evo" / "events.jsonl").read_text(encoding="utf-8")


def test_candidate_variant_is_pinned_to_existing_artifact_version(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    artifact_id = _artifact(store)
    variant = EvalVariant(
        variant_id=new_evo_id("eval_variant"),
        kind="candidate",
        model_config_hash="d" * 64,
        strategy_hash="e" * 64,
        artifact_id=artifact_id,
        artifact_version=1,
    )

    result = EvalHarness(store).run(
        _case(),
        variant,
        _Evaluator(store, EvalObservation(task_outcome="task_success")),
        seed=3,
    )

    assert result.persisted is True
    assert result.trial is not None
    assert result.trial.payload.artifact_id == artifact_id
    assert result.trial.payload.artifact_version == 1
    with pytest.raises(EvalHarnessError, match="existing Artifact version"):
        EvalHarness(store).run(
            _case(),
            EvalVariant(
                variant_id=new_evo_id("eval_variant"),
                kind="candidate",
                model_config_hash="d" * 64,
                strategy_hash="e" * 64,
                artifact_id=artifact_id,
                artifact_version=2,
            ),
            _Evaluator(store, EvalObservation(task_outcome="task_success")),
            seed=3,
        )


def test_task_failure_and_evaluator_failure_are_distinct(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    harness = EvalHarness(store)
    task_failure = harness.run(
        _case(),
        _baseline(),
        _Evaluator(
            store,
            EvalObservation(task_outcome="task_failure", verification_quality=0.8),
        ),
        seed=1,
    )
    evaluator_failure = harness.run(_case(), _baseline(), _FailingEvaluator(), seed=2)

    assert task_failure.trial is not None
    assert task_failure.trial.payload.task_outcome == "task_failure"
    assert task_failure.trial.payload.evaluation_status == "completed"
    assert task_failure.trial.payload.success is False
    assert evaluator_failure.trial is not None
    assert evaluator_failure.trial.payload.task_outcome == "not_run"
    assert evaluator_failure.trial.payload.evaluation_status == "evaluator_failure"
    assert evaluator_failure.trial.payload.success is None
    assert evaluator_failure.diagnostic is not None
    assert evaluator_failure.diagnostic.code == "evaluator_failure"


def test_invalid_fixed_input_and_store_failure_are_safe(tmp_path: Path) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")
    case = _case()
    invalid = EvalCase(
        case_id=case.case_id,
        corpus_id=case.corpus_id,
        corpus_version=case.corpus_version,
        split=case.split,
        public_input=case.public_input,
        task_input_hash="0" * 64,
        workspace_baseline_hash=case.workspace_baseline_hash,
        environment_hash=case.environment_hash,
    )
    with pytest.raises(EvalHarnessError, match="must match"):
        EvalHarness(store).run(
            invalid,
            _baseline(),
            _Evaluator(store, EvalObservation(task_outcome="task_success")),
            seed=1,
        )

    result = EvalHarness(_FailingStore()).run(
        case,
        _baseline(),
        _Evaluator(_FailingStore(), EvalObservation(task_outcome="task_success")),
        seed=1,
    )
    assert result.persisted is False
    assert result.trial is None
    assert result.diagnostic is not None
    assert result.diagnostic.code == "trial_recording_failed"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("missing", "does not exist"),
        ("cross_run", "different Run"),
        ("unverified", "not verified"),
        ("manual", "not deterministic"),
        ("no_exit", "no exit code"),
        ("outcome_mismatch", "do not support"),
        ("reported_mismatch", "evidence_success conflicts"),
        ("command_mismatch", "verification_commands conflict"),
    ],
)
def test_completed_trial_fails_closed_on_invalid_evidence(
    tmp_path: Path,
    mode: str,
    reason: str,
) -> None:
    store = EvoEventStore(tmp_path / ".bauhinia-agent")

    result = EvalHarness(store).run(
        _case(),
        _baseline(),
        _InvalidEvidenceEvaluator(store, mode),
        seed=11,
    )

    assert result.persisted is True
    assert result.trial is not None
    assert result.trial.payload.evaluation_status == "invalid"
    assert result.trial.payload.task_outcome == "not_run"
    assert result.trial.payload.success is None
    assert any(reason in item for item in result.trial.payload.invalid_reasons)
    assert result.diagnostic is not None
    assert result.diagnostic.code == "invalid_evidence"


def _case() -> EvalCase:
    public_input = "Fix the provider retry regression without reading the held-out answer."
    return EvalCase(
        case_id="eval_case_provider_retry",
        corpus_id="corpus_provider",
        corpus_version="v1",
        split="held_out",
        public_input=public_input,
        task_input_hash=hash_text(public_input),
        workspace_baseline_hash="b" * 64,
        environment_hash="c" * 64,
    )


def _baseline() -> EvalVariant:
    return EvalVariant(
        variant_id="eval_variant_baseline",
        kind="baseline",
        model_config_hash="d" * 64,
        strategy_hash="e" * 64,
    )


def _artifact(store: EvoEventStore) -> str:
    artifact_id = new_evo_id("artifact")
    source_candidate_id = new_evo_id("candidate")
    store.append(
        EvoEvent(
            event_id=new_evo_id("event"),
            event_type="CandidateArtifactCreated",
            refs=EvoReferences(run_id=new_evo_id("run"), artifact_id=artifact_id),
            payload=CandidateArtifactCreatedPayload(
                artifact_schema_version="v1",
                lineage_id=artifact_id,
                artifact_version=1,
                kind="plan_template",
                name="provider-retry-eval",
                description="Evaluate provider retries.",
                instructions="Apply bounded retry verification.",
                inputs=("provider task",),
                outputs=("verification evidence",),
                dependencies=("pytest",),
                effects=("read",),
                triggers=("provider retry",),
                scope="project",
                applicability="Provider retry changes.",
                risks=("Must pass held-out evaluation.",),
                source_candidate_ids=(source_candidate_id,),
                support_candidate_ids=(source_candidate_id,),
                counterexample_candidate_ids=(),
                source_run_ids=(new_evo_id("run"),),
                evidence_refs=(new_evo_id("evidence"),),
                counterexamples=(),
                confidence=0.6,
                content_hash="f" * 64,
            ),
        )
    )
    return artifact_id


class _Evaluator:
    version = "deterministic-v1"

    def __init__(self, store: object, observation: EvalObservation) -> None:
        self._store = store
        self._observation = observation

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        assert request.public_input
        success = self._observation.task_outcome == "task_success"
        recorded = EvidenceAdapter(self._store).record(  # type: ignore[arg-type]
            EvidenceInput(
                run_id=request.run_id,
                evidence_type="test",
                source="pytest",
                summary="deterministic evaluation verification",
                verified=True,
                command="pytest -q",
                exit_code=0 if success else 1,
            )
        )
        if recorded.evidence is None:
            raise RuntimeError("evaluation evidence could not be persisted")
        return replace(
            self._observation,
            evidence_refs=(recorded.evidence.evidence_id,),
            verification_commands=("pytest -q",),
            evidence_success=success,
        )


class _FailingEvaluator:
    version = "broken-v1"

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        del request
        raise RuntimeError("evaluator unavailable")


class _InvalidEvidenceEvaluator:
    version = "invalid-evidence-v1"

    def __init__(self, store: EvoEventStore, mode: str) -> None:
        self._store = store
        self._mode = mode

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        if self._mode == "missing":
            evidence_id = new_evo_id("evidence")
        else:
            recorded = EvidenceAdapter(self._store).record(
                EvidenceInput(
                    run_id=(new_evo_id("run") if self._mode == "cross_run" else request.run_id),
                    evidence_type="manual" if self._mode == "manual" else "test",
                    source="pytest",
                    summary="adversarial evaluation fixture",
                    verified=self._mode != "unverified",
                    command="pytest -q",
                    exit_code=(None if self._mode == "no_exit" else 1 if self._mode == "outcome_mismatch" else 0),
                )
            )
            assert recorded.evidence is not None
            evidence_id = recorded.evidence.evidence_id
        return EvalObservation(
            task_outcome="task_success",
            verification_quality=1.0,
            cost=1.0,
            latency_ms=1.0,
            evidence_refs=(evidence_id,),
            verification_commands=("different-command" if self._mode == "command_mismatch" else "pytest -q",),
            evidence_success=False if self._mode == "reported_mismatch" else True,
        )


class _FailingStore:
    def list_events(self) -> list[EvoEvent]:
        return []

    def append(self, event: EvoEvent) -> object:
        del event
        raise EvoStoreError("store offline")
