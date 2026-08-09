"""Repeatable Eval harness backed by append-only Evo Trial events."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

from bauhinia_agent.evaluation.models import EvalCase, EvalObservation, EvalRunInput, EvalVariant, Evaluator
from bauhinia_agent.evolution.evidence import redact_text
from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    EvaluationTrialRecordedPayload,
    EvoEvent,
    EvoReferences,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

EVALUATION_SCHEMA_VERSION = "v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class EvalHarnessError(ValueError):
    """Raised when an evaluation input violates reproducibility constraints."""


class _EvalStore(Protocol):
    def append(self, event: EvoEvent[EvaluationTrialRecordedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class EvalTrialRecord:
    event_id: str
    run_id: str
    occurred_at: str
    payload: EvaluationTrialRecordedPayload


@dataclass(frozen=True, slots=True)
class EvalDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EvalTrialResult:
    persisted: bool
    trial: EvalTrialRecord | None = None
    diagnostic: EvalDiagnostic | None = None


class EvalHarness:
    """Invoke an Evaluator with a fixed public input and append one Trial Run."""

    def __init__(self, store: EvoEventStore | _EvalStore) -> None:
        self._store = store

    def run(self, case: EvalCase, variant: EvalVariant, evaluator: Evaluator, *, seed: int) -> EvalTrialResult:
        events = self._store.list_events()
        _validate_case(case)
        _validate_variant(variant, events)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise EvalHarnessError("seed must be a non-negative integer")
        evaluator_version = getattr(evaluator, "version", None)
        if not isinstance(evaluator_version, str) or not evaluator_version.strip():
            raise EvalHarnessError("Evaluator.version must be a non-blank string")
        variant_hash = _variant_hash(variant)
        trial_key = _trial_key(case, variant_hash, evaluator_version, seed)
        attempt = 1 + sum(event.event_type == "EvaluationTrialRecorded" and isinstance(event.payload, EvaluationTrialRecordedPayload) and event.payload.trial_key == trial_key for event in events)
        diagnostic = None
        try:
            observation = evaluator.evaluate(
                EvalRunInput(
                    case_id=case.case_id,
                    corpus_id=case.corpus_id,
                    corpus_version=case.corpus_version,
                    split=case.split,
                    public_input=case.public_input,
                    workspace_baseline_hash=case.workspace_baseline_hash,
                    environment_hash=case.environment_hash,
                    variant=variant,
                    seed=seed,
                )
            )
            _validate_observation(observation)
        except Exception as error:  # noqa: BLE001 - evaluator failure is an evaluation result, not a task failure
            observation = EvalObservation(
                task_outcome="not_run",
                evaluation_status="evaluator_failure",
                verification_coverage=0.0,
                invalid_reasons=("Evaluator failed before producing a valid observation.",),
            )
            diagnostic = EvalDiagnostic("evaluator_failure", redact_text(str(error))[0])

        trial_id = new_evo_id("eval_trial")
        run_id = new_evo_id("run")
        payload = _payload(
            trial_id=trial_id,
            trial_key=trial_key,
            attempt=attempt,
            case=case,
            variant=variant,
            variant_hash=variant_hash,
            evaluator_version=evaluator_version,
            held_out_audit_version=getattr(evaluator, "held_out_audit_version", None),
            seed=seed,
            observation=observation,
        )
        parent_event_id = _artifact_event_id(events, variant.artifact_id)
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationTrialRecorded",
            refs=EvoReferences(
                run_id=run_id,
                artifact_id=variant.artifact_id,
                evaluation_id=trial_id,
                parent_event_id=parent_event_id,
            ),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return EvalTrialResult(False, diagnostic=EvalDiagnostic("trial_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - recorder failure cannot alter the evaluated task
            return EvalTrialResult(False, diagnostic=EvalDiagnostic("trial_recording_failed", f"unexpected Trial recorder failure: {error}"))
        if appended.diagnostic is not None and diagnostic is None:
            diagnostic = EvalDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return EvalTrialResult(True, _trial_from_event(appended.event), diagnostic)

    def list_trials(self, *, trial_key: str | None = None) -> tuple[EvalTrialRecord, ...]:
        if trial_key is not None:
            _require_hash(trial_key, field="trial_key")
        return tuple(
            _trial_from_event(event)
            for event in self._store.list_events()
            if event.event_type == "EvaluationTrialRecorded" and isinstance(event.payload, EvaluationTrialRecordedPayload) and (trial_key is None or event.payload.trial_key == trial_key)
        )

    def open_run(self, run_id: str) -> tuple[EvoEvent, ...]:
        require_evo_id(run_id, field="run_id", kind="run")
        return tuple(event for event in self._store.list_events() if event.refs.run_id == run_id)


def hash_text(value: str) -> str:
    if not isinstance(value, str):
        raise EvalHarnessError("hash_text value must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_case(case: EvalCase) -> None:
    for field, value, kind in (
        ("case_id", case.case_id, "eval_case"),
        ("corpus_id", case.corpus_id, "corpus"),
    ):
        require_evo_id(value, field=field, kind=kind)  # type: ignore[arg-type]
    if not isinstance(case.corpus_version, str) or not case.corpus_version.strip():
        raise EvalHarnessError("corpus_version must be a non-blank string")
    if case.split not in {"source", "development", "held_out"}:
        raise EvalHarnessError("split must be source, development, or held_out")
    if not isinstance(case.public_input, str) or not case.public_input.strip():
        raise EvalHarnessError("public_input must be a non-blank string")
    if hash_text(case.public_input) != case.task_input_hash:
        raise EvalHarnessError("task_input_hash must match public_input")
    for field, value in (
        ("task_input_hash", case.task_input_hash),
        ("workspace_baseline_hash", case.workspace_baseline_hash),
        ("environment_hash", case.environment_hash),
    ):
        _require_hash(value, field=field)
    _unique_ids(case.origin_run_ids, field="origin_run_ids", kind="run")
    _unique_ids(case.origin_evidence_refs, field="origin_evidence_refs", kind="evidence")


def _validate_variant(variant: EvalVariant, events: list[EvoEvent]) -> None:
    require_evo_id(variant.variant_id, field="variant_id", kind="eval_variant")
    if variant.kind not in {"baseline", "candidate"}:
        raise EvalHarnessError("Variant kind must be baseline or candidate")
    _require_hash(variant.model_config_hash, field="model_config_hash")
    _require_hash(variant.strategy_hash, field="strategy_hash")
    if variant.kind == "baseline":
        if variant.artifact_id is not None or variant.artifact_version is not None:
            raise EvalHarnessError("baseline Variant cannot reference an Artifact")
        return
    if variant.artifact_id is None or variant.artifact_version is None:
        raise EvalHarnessError("candidate Variant requires artifact_id and artifact_version")
    require_evo_id(variant.artifact_id, field="artifact_id", kind="artifact")
    artifact = next(
        (event for event in events if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id == variant.artifact_id),
        None,
    )
    if artifact is None or artifact.payload.artifact_version != variant.artifact_version:
        raise EvalHarnessError("candidate Variant must reference an existing Artifact version")


def _validate_observation(observation: EvalObservation) -> None:
    if not isinstance(observation, EvalObservation):
        raise EvalHarnessError("Evaluator must return EvalObservation")
    if observation.task_outcome not in {"task_success", "task_failure", "cancelled", "not_run"}:
        raise EvalHarnessError("unsupported task_outcome")
    if observation.evaluation_status not in {"completed", "invalid", "cancelled"}:
        raise EvalHarnessError("Evaluator cannot directly report evaluator_failure")
    if observation.evaluation_status == "completed" and observation.task_outcome not in {"task_success", "task_failure"}:
        raise EvalHarnessError("completed evaluation requires a task success or failure")
    for field, value in (("verification_quality", observation.verification_quality), ("verification_coverage", observation.verification_coverage)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1):
            raise EvalHarnessError(f"{field} must be between 0 and 1")
    for field, value in (("cost", observation.cost), ("latency_ms", observation.latency_ms)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
            raise EvalHarnessError(f"{field} must be non-negative")
    _unique_ids(observation.evidence_refs, field="evidence_refs", kind="evidence")
    for value in observation.accessed_resource_hashes:
        _require_hash(value, field="accessed_resource_hashes[]")
    if observation.evaluation_status == "invalid" and not observation.invalid_reasons:
        raise EvalHarnessError("invalid evaluation requires invalid_reasons")


def _payload(
    *,
    trial_id: str,
    trial_key: str,
    attempt: int,
    case: EvalCase,
    variant: EvalVariant,
    variant_hash: str,
    evaluator_version: str,
    held_out_audit_version: object,
    seed: int,
    observation: EvalObservation,
) -> EvaluationTrialRecordedPayload:
    success = None
    if observation.task_outcome in {"task_success", "task_failure"}:
        success = observation.task_outcome == "task_success"
    return EvaluationTrialRecordedPayload(
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        trial_id=trial_id,
        trial_key=trial_key,
        attempt=attempt,
        case_id=case.case_id,
        corpus_id=case.corpus_id,
        corpus_version=case.corpus_version,
        split=case.split,
        variant_id=variant.variant_id,
        variant_kind=variant.kind,
        artifact_id=variant.artifact_id,
        artifact_version=variant.artifact_version,
        evaluator_version=evaluator_version,
        seed=seed,
        task_input_hash=case.task_input_hash,
        workspace_baseline_hash=case.workspace_baseline_hash,
        environment_hash=case.environment_hash,
        model_config_hash=variant.model_config_hash,
        variant_hash=variant_hash,
        task_outcome=observation.task_outcome,
        evaluation_status=observation.evaluation_status,
        success=success,
        verification_quality=None if observation.verification_quality is None else float(observation.verification_quality),
        cost=None if observation.cost is None else float(observation.cost),
        latency_ms=None if observation.latency_ms is None else float(observation.latency_ms),
        risk_events=_redacted_items(observation.risk_events),
        evidence_refs=observation.evidence_refs,
        verification_commands=_redacted_items(observation.verification_commands),
        verification_skipped=observation.verification_skipped,
        verification_coverage=float(observation.verification_coverage),
        claimed_success=observation.claimed_success,
        evidence_success=observation.evidence_success,
        output_truncated=observation.output_truncated,
        accessed_resource_hashes=observation.accessed_resource_hashes,
        invalid_reasons=_redacted_items(observation.invalid_reasons),
        extensions={
            "harness_version": "p8-001",
            "private_reference_exposed": False,
            **({"held_out_audit_version": held_out_audit_version} if isinstance(held_out_audit_version, str) and held_out_audit_version else {}),
        },
    )


def _trial_key(case: EvalCase, variant_hash: str, evaluator_version: str, seed: int) -> str:
    fixed = {
        "case_id": case.case_id,
        "corpus_id": case.corpus_id,
        "corpus_version": case.corpus_version,
        "task_input_hash": case.task_input_hash,
        "workspace_baseline_hash": case.workspace_baseline_hash,
        "environment_hash": case.environment_hash,
        "variant_hash": variant_hash,
        "evaluator_version": evaluator_version,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(fixed, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _variant_hash(variant: EvalVariant) -> str:
    content = {
        "variant_id": variant.variant_id,
        "kind": variant.kind,
        "model_config_hash": variant.model_config_hash,
        "strategy_hash": variant.strategy_hash,
        "artifact_id": variant.artifact_id,
        "artifact_version": variant.artifact_version,
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _artifact_event_id(events: list[EvoEvent], artifact_id: str | None) -> str | None:
    if artifact_id is None:
        return None
    return next((event.event_id for event in events if event.refs.artifact_id == artifact_id and event.event_type == "CandidateArtifactCreated"), None)


def _unique_ids(values: tuple[str, ...], *, field: str, kind: str) -> None:
    if not isinstance(values, tuple):
        raise EvalHarnessError(f"{field} must be a tuple")
    for value in values:
        require_evo_id(value, field=f"{field}[]", kind=kind)  # type: ignore[arg-type]
    if len(set(values)) != len(values):
        raise EvalHarnessError(f"{field} must contain unique IDs")


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise EvalHarnessError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _redacted_items(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise EvalHarnessError("observation list fields must be tuples")
    return tuple(redact_text(value)[0] for value in values)


def _trial_from_event(event: EvoEvent[EvaluationTrialRecordedPayload]) -> EvalTrialRecord:
    return EvalTrialRecord(event.event_id, event.refs.run_id, event.occurred_at, event.payload)
