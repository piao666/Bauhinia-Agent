"""Licensed, immutable Eval Corpus manifests and held-out contamination audits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Protocol

from bauhinia_agent.evaluation.harness import EvalDiagnostic, EvalHarness, EvalHarnessError, EvalTrialResult, hash_text
from bauhinia_agent.evaluation.models import EvalCase, EvalObservation, EvalRunInput, EvalVariant, Evaluator
from bauhinia_agent.evolution.evidence import redact_text
from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    EvaluationCorpusRegisteredPayload,
    EvoEvent,
    EvoReferences,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError

CORPUS_SCHEMA_VERSION = "v1"
HELD_OUT_AUDIT_VERSION = "v1"
ALLOWED_CORPUS_LICENSES = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "Public-Domain",
    }
)
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class EvalCorpusError(ValueError):
    """Raised when a Corpus violates license, immutability, or split isolation."""


class _CorpusStore(Protocol):
    def append(self, event: EvoEvent[EvaluationCorpusRegisteredPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class EvalCorpusCase:
    case: EvalCase
    private_reference: str
    private_reference_hash: str


@dataclass(frozen=True, slots=True)
class EvalCorpusManifest:
    corpus_id: str
    version: str
    license_spdx: str
    provenance: str
    cases: tuple[EvalCorpusCase, ...]


@dataclass(frozen=True, slots=True)
class EvalCorpusRecord:
    event_id: str
    run_id: str
    occurred_at: str
    payload: EvaluationCorpusRegisteredPayload


@dataclass(frozen=True, slots=True)
class EvalCorpusDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EvalCorpusResult:
    persisted: bool
    corpus: EvalCorpusRecord | None = None
    diagnostic: EvalCorpusDiagnostic | None = None


class EvalCorpusRegistry:
    """Append immutable Corpus manifests; private answers remain caller-owned."""

    def __init__(self, store: EvoEventStore | _CorpusStore) -> None:
        self._store = store

    def register(self, manifest: EvalCorpusManifest) -> EvalCorpusResult:
        payload = _manifest_payload(manifest)
        events = self._store.list_events()
        existing = _registration(events, manifest.corpus_id, manifest.version)
        if existing is not None:
            if existing.payload.manifest_hash != payload.manifest_hash:
                raise EvalCorpusError("an existing Corpus version is immutable")
            return EvalCorpusResult(
                False,
                existing,
                EvalCorpusDiagnostic("already_registered", "the identical Corpus version is already registered"),
            )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationCorpusRegistered",
            refs=EvoReferences(run_id=new_evo_id("run"), evaluation_id=manifest.corpus_id),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return EvalCorpusResult(False, diagnostic=EvalCorpusDiagnostic("corpus_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - Corpus recording cannot alter tasks or Candidates
            return EvalCorpusResult(
                False,
                diagnostic=EvalCorpusDiagnostic("corpus_recording_failed", f"unexpected Corpus recorder failure: {error}"),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = EvalCorpusDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return EvalCorpusResult(True, _corpus_from_event(appended.event), diagnostic)

    def list_registered(self) -> tuple[EvalCorpusRecord, ...]:
        return tuple(
            _corpus_from_event(event) for event in self._store.list_events() if event.event_type == "EvaluationCorpusRegistered" and isinstance(event.payload, EvaluationCorpusRegisteredPayload)
        )


class HeldOutEvalHarness:
    """Run a Trial through split/source/access audits without exposing answers."""

    def __init__(self, store: EvoEventStore | _CorpusStore) -> None:
        self._store = store

    def run(
        self,
        manifest: EvalCorpusManifest,
        case_id: str,
        variant: EvalVariant,
        evaluator: Evaluator,
        *,
        seed: int,
    ) -> EvalTrialResult:
        require_evo_id(case_id, field="case_id", kind="eval_case")
        payload = _manifest_payload(manifest)
        selected = next((item for item in manifest.cases if item.case.case_id == case_id), None)
        if selected is None:
            raise EvalCorpusError(f"unknown Corpus Case: {case_id}")
        events = self._store.list_events()
        reasons: list[str] = []
        registered = _registration(events, manifest.corpus_id, manifest.version)
        if registered is None:
            reasons.append("Corpus version is not registered.")
        elif registered.payload.manifest_hash != payload.manifest_hash:
            reasons.append("Corpus Manifest hash does not match the registered version.")
        if selected.case.split != "held_out":
            reasons.append("Case is not in the held_out split.")
        if variant.kind == "candidate":
            artifact = _artifact(events, variant.artifact_id)
            if artifact is not None:
                if set(artifact.payload.source_run_ids) & set(selected.case.origin_run_ids):
                    reasons.append("Candidate source Run overlaps the held-out Case origin.")
                if set(artifact.payload.evidence_refs) & set(selected.case.origin_evidence_refs):
                    reasons.append("Candidate source Evidence overlaps the held-out Case origin.")
        audited = _HeldOutEvaluator(evaluator, selected.private_reference_hash, tuple(reasons))
        result = EvalHarness(self._store).run(selected.case, variant, audited, seed=seed)
        if result.trial is not None and result.trial.payload.evaluation_status == "invalid" and result.diagnostic is None:
            return EvalTrialResult(
                result.persisted,
                result.trial,
                EvalDiagnostic("held_out_contamination", "; ".join(result.trial.payload.invalid_reasons)),
            )
        return result


class _HeldOutEvaluator:
    def __init__(self, delegate: Evaluator, private_reference_hash: str, preflight_reasons: tuple[str, ...]) -> None:
        version = getattr(delegate, "version", None)
        if not isinstance(version, str) or not version.strip():
            raise EvalHarnessError("Evaluator.version must be a non-blank string")
        self.version = f"{version}+heldout-audit-{HELD_OUT_AUDIT_VERSION}"
        self.held_out_audit_version = HELD_OUT_AUDIT_VERSION
        self._delegate = delegate
        self._private_reference_hash = private_reference_hash
        self._preflight_reasons = preflight_reasons

    def evaluate(self, request: EvalRunInput) -> EvalObservation:
        if self._preflight_reasons:
            return EvalObservation(
                task_outcome="not_run",
                evaluation_status="invalid",
                verification_coverage=0.0,
                invalid_reasons=self._preflight_reasons,
            )
        observation = self._delegate.evaluate(request)
        if not isinstance(observation, EvalObservation):
            return observation
        if self._private_reference_hash not in observation.accessed_resource_hashes:
            return observation
        reasons = (*observation.invalid_reasons, "Candidate accessed the held-out private reference resource.")
        return replace(observation, evaluation_status="invalid", invalid_reasons=tuple(dict.fromkeys(reasons)))


def private_reference_hash(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvalCorpusError("private_reference must be a non-empty string")
    return hash_text(value)


def corpus_manifest_hash(manifest: EvalCorpusManifest) -> str:
    return _manifest_payload(manifest).manifest_hash


def _manifest_payload(manifest: EvalCorpusManifest) -> EvaluationCorpusRegisteredPayload:
    require_evo_id(manifest.corpus_id, field="corpus_id", kind="corpus")
    if not isinstance(manifest.version, str) or not manifest.version.strip():
        raise EvalCorpusError("Corpus version must be a non-blank string")
    if manifest.license_spdx not in ALLOWED_CORPUS_LICENSES:
        raise EvalCorpusError(f"Corpus license is not approved: {manifest.license_spdx}")
    if not isinstance(manifest.provenance, str) or not manifest.provenance.strip():
        raise EvalCorpusError("Corpus provenance must be a non-blank string")
    if not isinstance(manifest.cases, tuple) or not manifest.cases:
        raise EvalCorpusError("Corpus cases must be a non-empty tuple")
    case_ids: list[str] = []
    splits: list[str] = []
    task_hashes: list[str] = []
    workspace_hashes: list[str] = []
    environment_hashes: list[str] = []
    reference_hashes: list[str] = []
    case_hashes: list[str] = []
    for item in manifest.cases:
        case = item.case
        require_evo_id(case.case_id, field="case_id", kind="eval_case")
        if case.corpus_id != manifest.corpus_id or case.corpus_version != manifest.version:
            raise EvalCorpusError("every Case must match the enclosing Corpus ID and version")
        if case.split not in {"source", "development", "held_out"}:
            raise EvalCorpusError("unsupported Corpus split")
        if hash_text(case.public_input) != case.task_input_hash:
            raise EvalCorpusError("Case task_input_hash must match public_input")
        for field, value in (
            ("task_input_hash", case.task_input_hash),
            ("workspace_baseline_hash", case.workspace_baseline_hash),
            ("environment_hash", case.environment_hash),
            ("private_reference_hash", item.private_reference_hash),
        ):
            _require_hash(value, field=field)
        if private_reference_hash(item.private_reference) != item.private_reference_hash:
            raise EvalCorpusError("private_reference_hash must match the private reference")
        case_content = {
            "case_id": case.case_id,
            "split": case.split,
            "task_input_hash": case.task_input_hash,
            "workspace_baseline_hash": case.workspace_baseline_hash,
            "environment_hash": case.environment_hash,
            "private_reference_hash": item.private_reference_hash,
            "origin_run_ids": sorted(case.origin_run_ids),
            "origin_evidence_refs": sorted(case.origin_evidence_refs),
        }
        case_ids.append(case.case_id)
        splits.append(case.split)
        task_hashes.append(case.task_input_hash)
        workspace_hashes.append(case.workspace_baseline_hash)
        environment_hashes.append(case.environment_hash)
        reference_hashes.append(item.private_reference_hash)
        case_hashes.append(_canonical_hash(case_content))
    if len(set(case_ids)) != len(case_ids):
        raise EvalCorpusError("Case IDs must be unique within a Corpus version")
    if len(set(task_hashes)) != len(task_hashes):
        raise EvalCorpusError("task input fingerprints cannot repeat across Corpus splits")
    manifest_content = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_id": manifest.corpus_id,
        "corpus_version": manifest.version,
        "license_spdx": manifest.license_spdx,
        "provenance": redact_text(manifest.provenance.strip())[0],
        "case_manifest_hashes": case_hashes,
    }
    return EvaluationCorpusRegisteredPayload(
        corpus_schema_version=CORPUS_SCHEMA_VERSION,
        corpus_id=manifest.corpus_id,
        corpus_version=manifest.version,
        license_spdx=manifest.license_spdx,
        provenance=manifest_content["provenance"],
        case_ids=tuple(case_ids),
        case_splits=tuple(splits),
        task_input_hashes=tuple(task_hashes),
        workspace_baseline_hashes=tuple(workspace_hashes),
        environment_hashes=tuple(environment_hashes),
        private_reference_hashes=tuple(reference_hashes),
        case_manifest_hashes=tuple(case_hashes),
        manifest_hash=_canonical_hash(manifest_content),
        extensions={"registry_version": "p8-002", "private_answers_persisted": False},
    )


def _registration(events: list[EvoEvent], corpus_id: str, version: str) -> EvalCorpusRecord | None:
    return next(
        (
            _corpus_from_event(event)
            for event in events
            if event.event_type == "EvaluationCorpusRegistered"
            and isinstance(event.payload, EvaluationCorpusRegisteredPayload)
            and event.payload.corpus_id == corpus_id
            and event.payload.corpus_version == version
        ),
        None,
    )


def _artifact(events: list[EvoEvent], artifact_id: str | None) -> EvoEvent[CandidateArtifactCreatedPayload] | None:
    if artifact_id is None:
        return None
    return next(
        (event for event in events if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id == artifact_id),
        None,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise EvalCorpusError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _corpus_from_event(event: EvoEvent[EvaluationCorpusRegisteredPayload]) -> EvalCorpusRecord:
    return EvalCorpusRecord(event.event_id, event.refs.run_id, event.occurred_at, event.payload)
