"""Evidence-backed Self Model observations and rebuildable profiles."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from bauhinia_agent.evaluation.evidence import EvaluationEvidenceError, attest_evaluation_evidence
from bauhinia_agent.evolution import (
    DETERMINISTIC_EVIDENCE_TYPES,
    EvaluationTrialRecordedPayload,
    EvoAppendResult,
    EvoEvent,
    EvoEventStore,
    EvoReferences,
    EvoStoreError,
    OutcomeClassifiedPayload,
    OutcomeIntegrityError,
    SelfModelObservationRecordedPayload,
    SelfModelUpdatedPayload,
    attest_outcome_event,
    new_evo_id,
    require_evo_id,
)
from bauhinia_agent.self_model.models import (
    MIN_PROFILE_SAMPLES,
    ProfileStatus,
    ProfileSelector,
    SelfModelError,
    SelfModelObservation,
    SelfModelProfile,
    TaskClassification,
    VerificationLevel,
    parse_utc,
)


class _SelfModelStore(Protocol):
    def append(self, event: EvoEvent) -> EvoAppendResult: ...

    def list_events(self) -> list[EvoEvent]: ...


@dataclass(frozen=True, slots=True)
class SelfModelDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ObservationResult:
    persisted: bool
    observation: SelfModelObservation | None = None
    diagnostic: SelfModelDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class ProfilePublishResult:
    persisted: bool
    profile: SelfModelProfile | None = None
    diagnostic: SelfModelDiagnostic | None = None


class SelfModelService:
    """Build transparent profiles without owning execution or permission APIs."""

    def __init__(
        self,
        *,
        store: EvoEventStore | _SelfModelStore,
        project_id: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        require_evo_id(project_id, field="project_id")
        self._store = store
        self._project_id = project_id
        self._clock = clock or (lambda: datetime.now(UTC))

    def record_observation(self, classification: TaskClassification, *, source_event_id: str) -> ObservationResult:
        """Append one fact derived from a verified Outcome or completed Eval Trial."""

        if classification.project_id != self._project_id:
            raise SelfModelError("classification is outside this project scope")
        require_evo_id(source_event_id, field="source_event_id", kind="event")
        events = self._store.list_events()
        for event in events:
            if event.event_type == "SelfModelObservationRecorded" and isinstance(event.payload, SelfModelObservationRecordedPayload):
                if event.payload.project_id == self._project_id and event.payload.source_event_id == source_event_id:
                    return ObservationResult(
                        False,
                        _observation_from_event(event),
                        SelfModelDiagnostic("duplicate_observation", "source event is already represented in this project profile"),
                    )
        source = next((event for event in events if event.event_id == source_event_id), None)
        if source is None:
            raise SelfModelError("source event does not exist")
        derived = _derive_observation(source, events, classification)
        observation_id = new_evo_id("self_model")
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="SelfModelObservationRecorded",
            refs=EvoReferences(
                run_id=source.refs.run_id,
                self_model_id=observation_id,
                parent_event_id=source.event_id,
            ),
            payload=SelfModelObservationRecordedPayload(
                project_id=classification.project_id,
                model_config_hash=classification.model_config_hash,
                evaluator_version=classification.evaluator_version,
                environment_hash=classification.environment_hash,
                language=classification.language,
                repository_scale=classification.repository_scale,
                task_type=classification.task_type,
                tool_category=classification.tool_category,
                risk_level=classification.risk_level,
                verification_level=derived.verification_level,
                source_event_id=source.event_id,
                success=derived.success,
                outcome_category=derived.outcome_category,
                verification_quality=derived.verification_quality,
                cost=derived.cost,
                latency_ms=derived.latency_ms,
                risk_event_count=derived.risk_event_count,
                evidence_refs=derived.evidence_refs,
            ),
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return ObservationResult(False, diagnostic=SelfModelDiagnostic("observation_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - observation failure must not alter the source Run
            return ObservationResult(False, diagnostic=SelfModelDiagnostic("observation_recording_failed", f"unexpected observation recorder failure: {error}"))
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = SelfModelDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return ObservationResult(True, _observation_from_event(appended.event), diagnostic)

    def list_observations(self) -> tuple[SelfModelObservation, ...]:
        records = [
            _observation_from_event(event)
            for event in self._store.list_events()
            if event.event_type == "SelfModelObservationRecorded" and isinstance(event.payload, SelfModelObservationRecordedPayload) and event.payload.project_id == self._project_id
        ]
        return tuple(sorted(records, key=lambda item: (item.occurred_at, item.event_id)))

    def build_profile(self, selector: ProfileSelector) -> SelfModelProfile:
        if selector.project_id != self._project_id:
            raise SelfModelError("selector is outside this project scope")
        observations = tuple(item for item in self.list_observations() if _matches(selector, item))
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
            raise SelfModelError("Self Model clock must return UTC")
        window_start = min((item.occurred_at for item in observations), default=now)
        window_end = max((item.occurred_at for item in observations), default=now)
        sample_count = len(observations)
        success_count = sum(item.success for item in observations)
        if sample_count:
            success_rate = success_count / sample_count
            confidence_low, confidence_high = _wilson_interval(success_count, sample_count)
            uncertainty = (confidence_high - confidence_low) / 2
        else:
            success_rate = confidence_low = confidence_high = uncertainty = None
        status = _profile_status(sample_count, confidence_low, confidence_high)
        confidence = 0.0 if sample_count < MIN_PROFILE_SAMPLES or uncertainty is None else max(0.0, 1.0 - 2 * uncertainty)
        failures = Counter(item.outcome_category for item in observations if not item.success)
        return SelfModelProfile(
            selector=selector,
            sample_count=sample_count,
            success_count=success_count,
            success_rate=success_rate,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            uncertainty=uncertainty,
            confidence=confidence,
            status=status,
            window_start=window_start,
            window_end=window_end,
            average_verification_quality=_mean(item.verification_quality for item in observations),
            average_cost=_mean(item.cost for item in observations if item.cost is not None),
            average_latency_ms=_mean(item.latency_ms for item in observations if item.latency_ms is not None),
            risk_event_count=sum(item.risk_event_count for item in observations),
            failure_counts=tuple(sorted(failures.items())),
            source_event_ids=tuple(item.event_id for item in observations),
            source_run_ids=tuple(dict.fromkeys(item.run_id for item in observations)),
        )

    def publish_profile(
        self,
        selector: ProfileSelector,
        *,
        run_id: str | None = None,
    ) -> ProfilePublishResult:
        """Append an auditable snapshot; observations remain the rebuildable source.

        Runtime consumers may bind the snapshot to the Agent Run that consumed it.
        Callers that only need an offline profile keep the historical behaviour of
        allocating an independent Run identifier.
        """

        profile = self.build_profile(selector)
        profile_id = new_evo_id("self_model")
        resolved_run_id = new_evo_id("run") if run_id is None else require_evo_id(run_id, field="run_id", kind="run")
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="SelfModelUpdated",
            refs=EvoReferences(
                run_id=resolved_run_id,
                self_model_id=profile_id,
                parent_event_id=profile.source_event_ids[-1] if profile.source_event_ids else None,
            ),
            payload=SelfModelUpdatedPayload(
                dimension=selector.dimension,
                scope=f"project:{self._project_id}",
                sample_count=profile.sample_count,
                window_start=_utc_text(profile.window_start),
                window_end=_utc_text(profile.window_end),
                confidence=profile.confidence,
                metrics=profile.metrics(),
                extensions={
                    "profile_schema_version": "v1",
                    "profile_key": profile.profile_key,
                    "selector": selector.to_dict(),
                    "source_event_ids": profile.source_event_ids,
                    "source_run_ids": profile.source_run_ids,
                    "minimum_sample_count": MIN_PROFILE_SAMPLES,
                    "aggregation_declared": True,
                },
            ),
        )
        try:
            appended = self._store.append(event)
        except EvoStoreError as error:
            return ProfilePublishResult(False, diagnostic=SelfModelDiagnostic("profile_recording_failed", str(error)))
        except Exception as error:  # noqa: BLE001 - profile persistence cannot alter source observations
            return ProfilePublishResult(False, diagnostic=SelfModelDiagnostic("profile_recording_failed", f"unexpected profile recorder failure: {error}"))
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = SelfModelDiagnostic(appended.diagnostic.code, appended.diagnostic.message)
        return ProfilePublishResult(True, replace(profile, published_event_id=appended.event.event_id), diagnostic)


@dataclass(frozen=True, slots=True)
class _DerivedObservation:
    verification_level: VerificationLevel
    success: bool
    outcome_category: str
    verification_quality: float
    cost: float | None
    latency_ms: float | None
    risk_event_count: int
    evidence_refs: tuple[str, ...]


def _derive_observation(source: EvoEvent, events: list[EvoEvent], classification: TaskClassification) -> _DerivedObservation:
    if source.event_type == "EvaluationTrialRecorded" and isinstance(source.payload, EvaluationTrialRecordedPayload):
        payload = source.payload
        if payload.evaluation_status != "completed" or payload.success is None:
            raise SelfModelError("only completed, valid Evaluation Trials can update the Self Model")
        if payload.model_config_hash != classification.model_config_hash:
            raise SelfModelError("classification model does not match the Evaluation Trial")
        if payload.evaluator_version != classification.evaluator_version:
            raise SelfModelError("classification evaluator does not match the Evaluation Trial")
        if payload.environment_hash != classification.environment_hash:
            raise SelfModelError("classification environment does not match the Evaluation Trial")
        try:
            attest_evaluation_evidence(
                events,
                payload.evidence_refs,
                run_id=source.refs.run_id,
                expected_success=payload.success,
                reported_evidence_success=payload.evidence_success,
                reported_commands=payload.verification_commands,
                before_sequence=source.sequence,
            )
        except EvaluationEvidenceError as error:
            raise SelfModelError(f"Evaluation Trial Evidence is invalid: {error}") from error
        quality = payload.verification_quality if payload.verification_quality is not None else payload.verification_coverage
        level: VerificationLevel = "none" if payload.verification_skipped or quality <= 0 else "strong" if quality >= 0.8 else "partial"
        return _DerivedObservation(
            level,
            payload.success,
            payload.task_outcome,
            quality,
            payload.cost,
            payload.latency_ms,
            len(payload.risk_events),
            payload.evidence_refs,
        )
    if source.event_type == "OutcomeClassified" and isinstance(source.payload, OutcomeClassifiedPayload):
        payload = source.payload
        if payload.outcome == "unknown" or not payload.evidence_refs:
            raise SelfModelError("Outcome observations require a classified result and Evidence references")
        try:
            records = attest_outcome_event(events, source)
        except (OutcomeIntegrityError, ValueError) as error:
            raise SelfModelError("Outcome does not match the canonical prior Evidence for its Run") from error
        deterministic = [record for record in records if record.payload.evidence_type in DETERMINISTIC_EVIDENCE_TYPES and record.payload.verified and record.payload.exit_code is not None]
        if deterministic:
            quality = 1.0
            level = "strong"
        elif records:
            quality = 0.5
            level = "partial"
        else:
            quality = 0.0
            level = "none"
        return _DerivedObservation(level, payload.outcome == "success", payload.category, quality, None, None, 0, payload.evidence_refs)
    raise SelfModelError("Self Model source must be an OutcomeClassified or EvaluationTrialRecorded event")


def _observation_from_event(event: EvoEvent[SelfModelObservationRecordedPayload]) -> SelfModelObservation:
    payload = event.payload
    classification = TaskClassification(
        project_id=payload.project_id,
        model_config_hash=payload.model_config_hash,
        evaluator_version=payload.evaluator_version,
        environment_hash=payload.environment_hash,
        language=payload.language,
        repository_scale=payload.repository_scale,  # type: ignore[arg-type]
        task_type=payload.task_type,
        tool_category=payload.tool_category,
        risk_level=payload.risk_level,  # type: ignore[arg-type]
    )
    return SelfModelObservation(
        observation_id=event.refs.self_model_id or "",
        event_id=event.event_id,
        source_event_id=payload.source_event_id,
        run_id=event.refs.run_id,
        occurred_at=parse_utc(event.occurred_at),
        classification=classification,
        verification_level=payload.verification_level,  # type: ignore[arg-type]
        success=payload.success,
        outcome_category=payload.outcome_category,
        verification_quality=payload.verification_quality,
        cost=payload.cost,
        latency_ms=payload.latency_ms,
        risk_event_count=payload.risk_event_count,
        evidence_refs=payload.evidence_refs,
    )


def _matches(selector: ProfileSelector, observation: SelfModelObservation) -> bool:
    classification = observation.classification
    if (
        classification.project_id != selector.project_id
        or classification.model_config_hash != selector.model_config_hash
        or classification.evaluator_version != selector.evaluator_version
        or classification.environment_hash != selector.environment_hash
    ):
        return False
    for field_name in ("language", "repository_scale", "task_type", "tool_category", "risk_level"):
        selected = getattr(selector, field_name)
        if selected is not None and selected != getattr(classification, field_name):
            return False
    return selector.verification_level is None or selector.verification_level == observation.verification_level


def _wilson_interval(successes: int, sample_count: int) -> tuple[float, float]:
    if sample_count < 1:
        raise SelfModelError("Wilson interval requires at least one sample")
    z = 1.959963984540054
    proportion = successes / sample_count
    denominator = 1 + z * z / sample_count
    center = (proportion + z * z / (2 * sample_count)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / sample_count + z * z / (4 * sample_count * sample_count)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _profile_status(sample_count: int, low: float | None, high: float | None) -> ProfileStatus:
    if sample_count < MIN_PROFILE_SAMPLES or low is None or high is None:
        return "insufficient_data"
    if low >= 0.8:
        return "reliable"
    if high < 0.6:
        return "unreliable"
    return "mixed"


def _mean(values) -> float | None:
    collected = tuple(values)
    return sum(collected) / len(collected) if collected else None


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
