"""Deterministic baseline/candidate held-out comparison reports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Protocol

from bauhinia_agent.evolution.events import (
    CandidateArtifactCreatedPayload,
    EvaluationComparisonCompletedPayload,
    EvaluationCorpusRegisteredPayload,
    EvaluationTrialRecordedPayload,
    EvoEvent,
    EvoReferences,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoAppendResult, EvoEventStore, EvoStoreError


class EvaluationComparisonError(ValueError):
    """Raised when a comparison request cannot identify a fixed candidate."""


class _ComparisonStore(Protocol):
    def append(self, event: EvoEvent[EvaluationComparisonCompletedPayload]) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    minimum_cases: int = 5
    minimum_repeats: int = 2
    minimum_success_delta: float = 0.10
    maximum_cost_ratio: float = 1.25
    maximum_latency_ratio: float = 1.25
    maximum_uncertainty: float = 0.25


DEFAULT_PROMOTION_THRESHOLDS = PromotionThresholds()


@dataclass(frozen=True, slots=True)
class EvaluationComparisonSpec:
    artifact_id: str
    artifact_version: int
    corpus_id: str
    corpus_version: str
    evaluator_version: str
    baseline_variant_id: str
    candidate_variant_id: str
    thresholds: PromotionThresholds = PromotionThresholds()


@dataclass(frozen=True, slots=True)
class EvaluationComparisonRecord:
    event_id: str
    run_id: str
    occurred_at: str
    payload: EvaluationComparisonCompletedPayload


@dataclass(frozen=True, slots=True)
class EvaluationComparisonDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EvaluationComparisonResult:
    persisted: bool
    report: EvaluationComparisonRecord | None = None
    diagnostic: EvaluationComparisonDiagnostic | None = None


class EvaluationComparisonService:
    """Report separated metrics; it never promotes or executes an Artifact."""

    def __init__(self, store: EvoEventStore | _ComparisonStore) -> None:
        self._store = store

    def compare(self, spec: EvaluationComparisonSpec) -> EvaluationComparisonResult:
        events = self._store.list_events()
        artifact = _validate_spec(spec, events)
        selected = tuple(
            event
            for event in events
            if event.event_type == "EvaluationTrialRecorded"
            and isinstance(event.payload, EvaluationTrialRecordedPayload)
            and event.payload.corpus_id == spec.corpus_id
            and event.payload.corpus_version == spec.corpus_version
            and event.payload.evaluator_version == spec.evaluator_version
            and event.payload.variant_id in {spec.baseline_variant_id, spec.candidate_variant_id}
        )
        baseline_all = tuple(event for event in selected if event.payload.variant_id == spec.baseline_variant_id)
        candidate_all = tuple(event for event in selected if event.payload.variant_id == spec.candidate_variant_id)
        if any(event.payload.artifact_id != spec.artifact_id or event.payload.artifact_version != spec.artifact_version for event in candidate_all):
            raise EvaluationComparisonError("candidate Trials do not match the requested Artifact version")
        evidence_errors = {event.event_id: error for event in selected if (error := _trial_evidence_error(event, events)) is not None}
        baseline = tuple(event for event in baseline_all if _is_valid_trial(event.payload) and event.event_id not in evidence_errors)
        candidate = tuple(event for event in candidate_all if _is_valid_trial(event.payload) and event.event_id not in evidence_errors)
        invalid_count = len(selected) - len(baseline) - len(candidate)
        reasons = _blocking_reasons(spec, baseline_all, candidate_all, baseline, candidate, invalid_count)
        if not any(
            event.event_type == "EvaluationCorpusRegistered"
            and isinstance(event.payload, EvaluationCorpusRegisteredPayload)
            and event.payload.corpus_id == spec.corpus_id
            and event.payload.corpus_version == spec.corpus_version
            for event in events
        ):
            reasons.append("Comparison requires a registered immutable Corpus version.")
        if any(event.payload.extensions.get("held_out_audit_version") != "v1" for event in selected):
            reasons.append("Every selected Trial must pass the held-out audit service.")
        integrity = [f"Trial Evidence integrity failed: {message}" for message in dict.fromkeys(evidence_errors.values())]
        integrity.extend(_integrity_violations(baseline, candidate))
        reasons.extend(integrity)
        case_ids = tuple(sorted({event.payload.case_id for event in (*baseline_all, *candidate_all)}))
        baseline_success = _success_rate(baseline)
        candidate_success = _success_rate(candidate)
        baseline_quality = _average(baseline, "verification_quality")
        candidate_quality = _average(candidate, "verification_quality")
        baseline_cost = _average(baseline, "cost")
        candidate_cost = _average(candidate, "cost")
        baseline_latency = _average(baseline, "latency_ms")
        candidate_latency = _average(candidate, "latency_ms")
        uncertainty = _uncertainty(baseline_success, candidate_success, len(baseline), len(candidate))
        payload = EvaluationComparisonCompletedPayload(
            report_id=new_evo_id("evaluation"),
            artifact_id=spec.artifact_id,
            artifact_version=spec.artifact_version,
            corpus_id=spec.corpus_id,
            corpus_version=spec.corpus_version,
            evaluator_version=spec.evaluator_version,
            baseline_variant_id=spec.baseline_variant_id,
            candidate_variant_id=spec.candidate_variant_id,
            case_ids=case_ids,
            trial_event_ids=tuple(event.event_id for event in selected),
            baseline_sample_count=len(baseline),
            candidate_sample_count=len(candidate),
            invalid_trial_count=invalid_count,
            minimum_repeats=_minimum_repeats(baseline, candidate),
            baseline_success_rate=baseline_success,
            candidate_success_rate=candidate_success,
            baseline_verification_quality=baseline_quality,
            candidate_verification_quality=candidate_quality,
            baseline_cost=baseline_cost,
            candidate_cost=candidate_cost,
            baseline_latency_ms=baseline_latency,
            candidate_latency_ms=candidate_latency,
            baseline_risk_event_count=sum(len(event.payload.risk_events) for event in baseline),
            candidate_risk_event_count=sum(len(event.payload.risk_events) for event in candidate),
            uncertainty=uncertainty,
            eligible=not reasons,
            blocking_reasons=tuple(dict.fromkeys(reasons)),
            integrity_violations=tuple(dict.fromkeys(integrity)),
            extensions={
                "comparison_version": "p8-003",
                "metrics_aggregated_separately": True,
                "automatic_promotion": False,
                "canonical_evidence_required": True,
                "thresholds": {
                    "minimum_cases": spec.thresholds.minimum_cases,
                    "minimum_repeats": spec.thresholds.minimum_repeats,
                    "minimum_success_delta": spec.thresholds.minimum_success_delta,
                    "maximum_cost_ratio": spec.thresholds.maximum_cost_ratio,
                    "maximum_latency_ratio": spec.thresholds.maximum_latency_ratio,
                    "maximum_uncertainty": spec.thresholds.maximum_uncertainty,
                },
            },
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="EvaluationComparisonCompleted",
            refs=EvoReferences(
                run_id=artifact.refs.run_id,
                artifact_id=spec.artifact_id,
                evaluation_id=payload.report_id,
                parent_event_id=selected[-1].event_id if selected else artifact.event_id,
            ),
            payload=payload,
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return EvaluationComparisonResult(
                False,
                diagnostic=EvaluationComparisonDiagnostic("comparison_recording_failed", str(error)),
            )
        except Exception as error:  # noqa: BLE001 - report failure cannot alter Trials or Candidate state
            return EvaluationComparisonResult(
                False,
                diagnostic=EvaluationComparisonDiagnostic(
                    "comparison_recording_failed",
                    f"unexpected comparison recorder failure: {error}",
                ),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = EvaluationComparisonDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return EvaluationComparisonResult(True, _report_from_event(appended.event), diagnostic)

    def list_reports(self, artifact_id: str | None = None) -> tuple[EvaluationComparisonRecord, ...]:
        if artifact_id is not None:
            require_evo_id(artifact_id, field="artifact_id", kind="artifact")
        return tuple(
            _report_from_event(event)
            for event in self._store.list_events()
            if event.event_type == "EvaluationComparisonCompleted"
            and isinstance(event.payload, EvaluationComparisonCompletedPayload)
            and (artifact_id is None or event.payload.artifact_id == artifact_id)
        )


def validate_comparison_report_facts(
    events: list[EvoEvent],
    report: EvoEvent[EvaluationComparisonCompletedPayload],
) -> None:
    """Recompute one report from facts that existed before it was appended.

    ``EvaluationComparisonCompleted`` is a derived fact, not an authority of
    its own.  Lifecycle gates therefore replay the deterministic comparison at
    the report's sequence boundary and require the complete payload and event
    references to match.  Later Trials also make the report stale.
    """

    if report.sequence is None:
        raise EvaluationComparisonError("persisted comparison report has no sequence")
    matches = [event for event in events if event.event_id == report.event_id]
    if len(matches) != 1 or matches[0] != report:
        raise EvaluationComparisonError("comparison report is not the canonical persisted event")
    if any(event.sequence is None for event in events):
        raise EvaluationComparisonError("comparison facts contain an unsequenced event")
    prior = sorted(
        (event for event in events if event.sequence is not None and event.sequence < report.sequence),
        key=lambda event: event.sequence or 0,
    )
    spec = _spec_from_report(report.payload)
    stale_trials = [
        event
        for event in events
        if event.sequence is not None
        and event.sequence > report.sequence
        and event.event_type == "EvaluationTrialRecorded"
        and isinstance(event.payload, EvaluationTrialRecordedPayload)
        and event.payload.corpus_id == spec.corpus_id
        and event.payload.corpus_version == spec.corpus_version
        and event.payload.evaluator_version == spec.evaluator_version
        and event.payload.variant_id in {spec.baseline_variant_id, spec.candidate_variant_id}
    ]
    if stale_trials:
        raise EvaluationComparisonError("comparison report is stale because matching Trials were appended later")
    replay = _ComparisonReplayStore(prior)
    try:
        result = EvaluationComparisonService(replay).compare(spec)
    except (EvaluationComparisonError, ValueError) as error:
        raise EvaluationComparisonError(f"comparison report facts cannot be recomputed: {error}") from error
    if result.report is None or replay.appended is None:
        message = result.diagnostic.message if result.diagnostic is not None else "comparison replay produced no report"
        raise EvaluationComparisonError(message)
    expected_payload = replace(
        replay.appended.payload,
        report_id=report.payload.report_id,
    )
    expected_refs = replace(
        replay.appended.refs,
        evaluation_id=report.payload.report_id,
    )
    if report.payload != expected_payload or report.refs != expected_refs:
        raise EvaluationComparisonError("comparison report payload or references do not match prior Trial facts")


class _ComparisonReplayStore:
    """In-memory append target used only for deterministic report replay."""

    def __init__(self, events: list[EvoEvent]) -> None:
        self._events = tuple(events)
        self.appended: EvoEvent[EvaluationComparisonCompletedPayload] | None = None

    def list_events(self) -> list[EvoEvent]:
        return list(self._events)

    def append(
        self,
        event: EvoEvent[EvaluationComparisonCompletedPayload],
    ) -> EvoAppendResult:
        persisted = replace(event, sequence=len(self._events) + 1)
        self.appended = persisted
        return EvoAppendResult(event=persisted, projection_applied=True)


def _spec_from_report(
    payload: EvaluationComparisonCompletedPayload,
) -> EvaluationComparisonSpec:
    raw = payload.extensions.get("thresholds")
    expected_keys = {
        "minimum_cases",
        "minimum_repeats",
        "minimum_success_delta",
        "maximum_cost_ratio",
        "maximum_latency_ratio",
        "maximum_uncertainty",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise EvaluationComparisonError("comparison report has no complete threshold provenance")

    def integer(name: str) -> int:
        value = raw[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise EvaluationComparisonError(f"comparison threshold {name} must be an integer")
        return value

    def number(name: str) -> float:
        value = raw[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EvaluationComparisonError(f"comparison threshold {name} must be numeric")
        return float(value)

    return EvaluationComparisonSpec(
        artifact_id=payload.artifact_id,
        artifact_version=payload.artifact_version,
        corpus_id=payload.corpus_id,
        corpus_version=payload.corpus_version,
        evaluator_version=payload.evaluator_version,
        baseline_variant_id=payload.baseline_variant_id,
        candidate_variant_id=payload.candidate_variant_id,
        thresholds=PromotionThresholds(
            minimum_cases=integer("minimum_cases"),
            minimum_repeats=integer("minimum_repeats"),
            minimum_success_delta=number("minimum_success_delta"),
            maximum_cost_ratio=number("maximum_cost_ratio"),
            maximum_latency_ratio=number("maximum_latency_ratio"),
            maximum_uncertainty=number("maximum_uncertainty"),
        ),
    )


def _validate_spec(
    spec: EvaluationComparisonSpec,
    events: list[EvoEvent],
) -> EvoEvent[CandidateArtifactCreatedPayload]:
    require_evo_id(spec.artifact_id, field="artifact_id", kind="artifact")
    if spec.artifact_version < 1:
        raise EvaluationComparisonError("artifact_version must be positive")
    for field, value in (
        ("corpus_id", spec.corpus_id),
        ("corpus_version", spec.corpus_version),
        ("evaluator_version", spec.evaluator_version),
        ("baseline_variant_id", spec.baseline_variant_id),
        ("candidate_variant_id", spec.candidate_variant_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise EvaluationComparisonError(f"{field} must be a non-blank string")
    if spec.baseline_variant_id == spec.candidate_variant_id:
        raise EvaluationComparisonError("baseline and candidate Variants must differ")
    thresholds = spec.thresholds
    if thresholds.minimum_cases < 1 or thresholds.minimum_repeats < 1:
        raise EvaluationComparisonError("minimum Cases and repeats must be positive")
    defaults = DEFAULT_PROMOTION_THRESHOLDS
    if (
        thresholds.minimum_cases < defaults.minimum_cases
        or thresholds.minimum_repeats < defaults.minimum_repeats
        or thresholds.minimum_success_delta < defaults.minimum_success_delta
        or thresholds.maximum_cost_ratio > defaults.maximum_cost_ratio
        or thresholds.maximum_latency_ratio > defaults.maximum_latency_ratio
        or thresholds.maximum_uncertainty > defaults.maximum_uncertainty
    ):
        raise EvaluationComparisonError("Promotion thresholds cannot be weaker than the accepted governance defaults")
    artifact = next(
        (event for event in events if event.event_type == "CandidateArtifactCreated" and isinstance(event.payload, CandidateArtifactCreatedPayload) and event.refs.artifact_id == spec.artifact_id),
        None,
    )
    if artifact is None or artifact.payload.artifact_version != spec.artifact_version:
        raise EvaluationComparisonError("comparison must reference an existing Artifact version")
    return artifact


def _blocking_reasons(
    spec: EvaluationComparisonSpec,
    baseline_all: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
    candidate_all: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
    baseline: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
    candidate: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
    invalid_count: int,
) -> list[str]:
    reasons: list[str] = []
    baseline_cases = {event.payload.case_id for event in baseline}
    candidate_cases = {event.payload.case_id for event in candidate}
    if baseline_cases != candidate_cases:
        reasons.append("Baseline and candidate must cover the same valid held-out Cases.")
    if len(baseline_cases & candidate_cases) < spec.thresholds.minimum_cases:
        reasons.append(f"At least {spec.thresholds.minimum_cases} distinct valid held-out Cases are required.")
    counts = Counter((event.payload.variant_id, event.payload.case_id) for event in (*baseline, *candidate))
    for variant_id in (spec.baseline_variant_id, spec.candidate_variant_id):
        for case_id in baseline_cases | candidate_cases:
            if counts[(variant_id, case_id)] < spec.thresholds.minimum_repeats:
                reasons.append(f"Every Variant requires at least {spec.thresholds.minimum_repeats} repeats per Case.")
                break
    if invalid_count or any(event.payload.split != "held_out" for event in (*baseline_all, *candidate_all)):
        reasons.append("All selected Trials must be valid held-out evaluations.")
    if not _fixed_pairs_match(baseline, candidate):
        reasons.append("Baseline and candidate Trials do not share fixed Case, seed, model, workspace, environment, and evaluator inputs.")
    if any(_missing_metrics(event.payload) for event in (*baseline, *candidate)):
        reasons.append("Every valid Trial must report verification quality, cost, and latency.")
    baseline_success = _success_rate(baseline)
    candidate_success = _success_rate(candidate)
    if candidate_success - baseline_success < spec.thresholds.minimum_success_delta:
        reasons.append("Candidate success-rate improvement is below the configured threshold.")
    baseline_quality = _average(baseline, "verification_quality")
    candidate_quality = _average(candidate, "verification_quality")
    if candidate_quality < baseline_quality:
        reasons.append("Candidate verification quality regressed.")
    baseline_cost = _average(baseline, "cost")
    candidate_cost = _average(candidate, "cost")
    if _ratio_exceeded(candidate_cost, baseline_cost, spec.thresholds.maximum_cost_ratio):
        reasons.append("Candidate cost exceeds the configured ratio.")
    baseline_latency = _average(baseline, "latency_ms")
    candidate_latency = _average(candidate, "latency_ms")
    if _ratio_exceeded(candidate_latency, baseline_latency, spec.thresholds.maximum_latency_ratio):
        reasons.append("Candidate latency exceeds the configured ratio.")
    if any(event.payload.risk_events for event in candidate):
        reasons.append("Candidate produced risk events.")
    uncertainty = _uncertainty(baseline_success, candidate_success, len(baseline), len(candidate))
    if uncertainty > spec.thresholds.maximum_uncertainty:
        reasons.append("Success-rate uncertainty exceeds the configured bound.")
    return reasons


def _fixed_pairs_match(
    baseline: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
    candidate: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
) -> bool:
    def signatures(events: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...]) -> Counter[tuple[object, ...]]:
        return Counter(
            (
                event.payload.case_id,
                event.payload.task_input_hash,
                event.payload.workspace_baseline_hash,
                event.payload.environment_hash,
                event.payload.model_config_hash,
                event.payload.evaluator_version,
                event.payload.seed,
            )
            for event in events
        )

    return signatures(baseline) == signatures(candidate)


def _integrity_violations(
    baseline: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
    candidate: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
) -> list[str]:
    violations: list[str] = []
    baseline_coverage: dict[str, list[float]] = defaultdict(list)
    baseline_commands: dict[str, bool] = defaultdict(bool)
    for event in baseline:
        baseline_coverage[event.payload.case_id].append(event.payload.verification_coverage)
        baseline_commands[event.payload.case_id] |= bool(event.payload.verification_commands)
    for event in candidate:
        payload = event.payload
        if payload.verification_skipped:
            violations.append("Candidate skipped verification.")
        expected_coverage = min(baseline_coverage.get(payload.case_id, [0.0]))
        if payload.verification_coverage < expected_coverage:
            violations.append("Candidate reduced verification coverage relative to baseline.")
        if payload.claimed_success is True and payload.evidence_success is False:
            violations.append("Candidate claimed success contrary to verification evidence.")
        if payload.output_truncated and (payload.success or payload.claimed_success):
            violations.append("Candidate truncated output while presenting success.")
        if baseline_commands[payload.case_id] and not payload.verification_commands:
            violations.append("Candidate omitted verification commands used by baseline.")
    return violations


def _is_valid_trial(payload: EvaluationTrialRecordedPayload) -> bool:
    return payload.evaluation_status == "completed" and payload.split == "held_out"


def _trial_evidence_error(
    event: EvoEvent[EvaluationTrialRecordedPayload],
    events: list[EvoEvent],
) -> str | None:
    from bauhinia_agent.evaluation.evidence import (
        EvaluationEvidenceError,
        attest_evaluation_evidence,
    )

    payload = event.payload
    if payload.extensions.get("evidence_integrity_valid") is False:
        return "; ".join(payload.invalid_reasons) or "Trial Evidence failed validation"
    if not _is_valid_trial(payload):
        return None
    if payload.success is None:
        return "completed Trial has no success result"
    if payload.success != (payload.task_outcome == "task_success"):
        return "completed Trial success conflicts with task_outcome"
    try:
        attest_evaluation_evidence(
            events,
            payload.evidence_refs,
            run_id=event.refs.run_id,
            expected_success=payload.success,
            reported_evidence_success=payload.evidence_success,
            reported_commands=payload.verification_commands,
            before_sequence=event.sequence,
        )
    except EvaluationEvidenceError as error:
        return str(error)
    return None


def _missing_metrics(payload: EvaluationTrialRecordedPayload) -> bool:
    return payload.success is None or payload.verification_quality is None or payload.cost is None or payload.latency_ms is None


def _success_rate(events: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...]) -> float:
    values = [event.payload.success for event in events if event.payload.success is not None]
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def _average(events: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...], field: str) -> float:
    values = [getattr(event.payload, field) for event in events]
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _minimum_repeats(
    baseline: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
    candidate: tuple[EvoEvent[EvaluationTrialRecordedPayload], ...],
) -> int:
    counts = Counter((event.payload.variant_id, event.payload.case_id) for event in (*baseline, *candidate))
    return min(counts.values()) if counts else 0


def _uncertainty(baseline_rate: float, candidate_rate: float, baseline_count: int, candidate_count: int) -> float:
    if baseline_count == 0 or candidate_count == 0:
        return 1.0
    baseline_half_width = _wilson_half_width(baseline_rate, baseline_count)
    candidate_half_width = _wilson_half_width(candidate_rate, candidate_count)
    return min(1.0, math.sqrt(baseline_half_width**2 + candidate_half_width**2))


def _wilson_half_width(rate: float, count: int) -> float:
    z = 1.96
    denominator = 1 + z**2 / count
    variance = rate * (1 - rate) / count + z**2 / (4 * count**2)
    return z * math.sqrt(variance) / denominator


def _ratio_exceeded(candidate: float, baseline: float, limit: float) -> bool:
    if baseline == 0:
        return candidate > 0
    return candidate / baseline > limit


def _report_from_event(event: EvoEvent[EvaluationComparisonCompletedPayload]) -> EvaluationComparisonRecord:
    return EvaluationComparisonRecord(event.event_id, event.refs.run_id, event.occurred_at, event.payload)
