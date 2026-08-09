"""Transparent Self Model domain records and scope contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from bauhinia_agent.evolution import require_evo_id

RepositoryScale = Literal["small", "medium", "large", "unknown"]
RiskLevel = Literal["low", "medium", "high", "unknown"]
VerificationLevel = Literal["none", "partial", "strong"]
ProfileStatus = Literal["insufficient_data", "reliable", "mixed", "unreliable"]

MIN_PROFILE_SAMPLES = 5
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.+-]*\Z")


class SelfModelError(ValueError):
    """Raised when Self Model data would be ambiguous or unsafe to aggregate."""


@dataclass(frozen=True, slots=True)
class TaskClassification:
    """Explicit task dimensions attached to one evidence-backed observation."""

    project_id: str
    model_config_hash: str
    evaluator_version: str
    environment_hash: str
    language: str
    repository_scale: RepositoryScale
    task_type: str
    tool_category: str
    risk_level: RiskLevel

    def __post_init__(self) -> None:
        require_evo_id(self.project_id, field="project_id")
        _digest(self.model_config_hash, field="model_config_hash")
        _digest(self.environment_hash, field="environment_hash")
        _text(self.evaluator_version, field="evaluator_version")
        object.__setattr__(self, "language", _token(self.language, field="language"))
        object.__setattr__(self, "task_type", _token(self.task_type, field="task_type"))
        object.__setattr__(self, "tool_category", _token(self.tool_category, field="tool_category"))
        if self.repository_scale not in {"small", "medium", "large", "unknown"}:
            raise SelfModelError("repository_scale must be small, medium, large, or unknown")
        if self.risk_level not in {"low", "medium", "high", "unknown"}:
            raise SelfModelError("risk_level must be low, medium, high, or unknown")

    def to_dict(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "model_config_hash": self.model_config_hash,
            "evaluator_version": self.evaluator_version,
            "environment_hash": self.environment_hash,
            "language": self.language,
            "repository_scale": self.repository_scale,
            "task_type": self.task_type,
            "tool_category": self.tool_category,
            "risk_level": self.risk_level,
        }


@dataclass(frozen=True, slots=True)
class ProfileSelector:
    """Declared aggregation boundary; project/model/evaluator/environment are mandatory."""

    project_id: str
    model_config_hash: str
    evaluator_version: str
    environment_hash: str
    language: str | None = None
    repository_scale: RepositoryScale | None = None
    task_type: str | None = None
    tool_category: str | None = None
    risk_level: RiskLevel | None = None
    verification_level: VerificationLevel | None = None

    def __post_init__(self) -> None:
        require_evo_id(self.project_id, field="project_id")
        _digest(self.model_config_hash, field="model_config_hash")
        _digest(self.environment_hash, field="environment_hash")
        _text(self.evaluator_version, field="evaluator_version")
        for field_name in ("language", "task_type", "tool_category"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _token(value, field=field_name))
        if self.repository_scale not in {None, "small", "medium", "large", "unknown"}:
            raise SelfModelError("repository_scale selector is invalid")
        if self.risk_level not in {None, "low", "medium", "high", "unknown"}:
            raise SelfModelError("risk_level selector is invalid")
        if self.verification_level not in {None, "none", "partial", "strong"}:
            raise SelfModelError("verification_level selector is invalid")

    @property
    def key(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def dimension(self) -> str:
        selected = [name for name in ("language", "repository_scale", "task_type", "tool_category", "risk_level", "verification_level") if getattr(self, name) is not None]
        return "+".join(selected) or "all_tasks"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "project_id": self.project_id,
            "model_config_hash": self.model_config_hash,
            "evaluator_version": self.evaluator_version,
            "environment_hash": self.environment_hash,
            "language": self.language,
            "repository_scale": self.repository_scale,
            "task_type": self.task_type,
            "tool_category": self.tool_category,
            "risk_level": self.risk_level,
            "verification_level": self.verification_level,
        }


@dataclass(frozen=True, slots=True)
class SelfModelObservation:
    observation_id: str
    event_id: str
    source_event_id: str
    run_id: str
    occurred_at: datetime
    classification: TaskClassification
    verification_level: VerificationLevel
    success: bool
    outcome_category: str
    verification_quality: float
    cost: float | None
    latency_ms: float | None
    risk_event_count: int
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("observation_id", self.observation_id),
            ("event_id", self.event_id),
            ("source_event_id", self.source_event_id),
            ("run_id", self.run_id),
        ):
            require_evo_id(value, field=field_name)
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() != UTC.utcoffset(self.occurred_at):
            raise SelfModelError("occurred_at must be UTC")
        if self.verification_level not in {"none", "partial", "strong"}:
            raise SelfModelError("verification_level is invalid")
        _rate(self.verification_quality, field="verification_quality")
        _optional_non_negative(self.cost, field="cost")
        _optional_non_negative(self.latency_ms, field="latency_ms")
        if isinstance(self.risk_event_count, bool) or self.risk_event_count < 0:
            raise SelfModelError("risk_event_count must be a non-negative integer")
        _text(self.outcome_category, field="outcome_category")


@dataclass(frozen=True, slots=True)
class SelfModelProfile:
    selector: ProfileSelector
    sample_count: int
    success_count: int
    success_rate: float | None
    confidence_low: float | None
    confidence_high: float | None
    uncertainty: float | None
    confidence: float
    status: ProfileStatus
    window_start: datetime
    window_end: datetime
    average_verification_quality: float | None
    average_cost: float | None
    average_latency_ms: float | None
    risk_event_count: int
    failure_counts: tuple[tuple[str, int], ...]
    source_event_ids: tuple[str, ...]
    source_run_ids: tuple[str, ...]
    published_event_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise SelfModelError("sample_count must be a non-negative integer")
        if isinstance(self.success_count, bool) or not isinstance(self.success_count, int) or not 0 <= self.success_count <= self.sample_count:
            raise SelfModelError("success_count must be between zero and sample_count")
        for field_name in ("success_rate", "confidence_low", "confidence_high", "uncertainty", "average_verification_quality"):
            value = getattr(self, field_name)
            if value is not None:
                _rate(value, field=field_name)
        _rate(self.confidence, field="confidence")
        _optional_non_negative(self.average_cost, field="average_cost")
        _optional_non_negative(self.average_latency_ms, field="average_latency_ms")
        if self.status not in {"insufficient_data", "reliable", "mixed", "unreliable"}:
            raise SelfModelError("status is invalid")
        if self.sample_count < MIN_PROFILE_SAMPLES and self.status != "insufficient_data":
            raise SelfModelError("low-sample profiles must report insufficient_data")
        for field_name, value in (("window_start", self.window_start), ("window_end", self.window_end)):
            if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
                raise SelfModelError(f"{field_name} must be UTC")
        if self.window_end < self.window_start:
            raise SelfModelError("window_end must not precede window_start")
        if isinstance(self.risk_event_count, bool) or not isinstance(self.risk_event_count, int) or self.risk_event_count < 0:
            raise SelfModelError("risk_event_count must be a non-negative integer")
        seen_failures: set[str] = set()
        for category, count in self.failure_counts:
            _text(category, field="failure category")
            if category in seen_failures or isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise SelfModelError("failure_counts must contain unique positive counts")
            seen_failures.add(category)
        for field_name, values in (("source_event_ids", self.source_event_ids), ("source_run_ids", self.source_run_ids)):
            if len(values) != len(set(values)):
                raise SelfModelError(f"{field_name} must be unique")
            for value in values:
                require_evo_id(value, field=field_name)
        if self.published_event_id is not None:
            require_evo_id(self.published_event_id, field="published_event_id", kind="event")

    @property
    def profile_key(self) -> str:
        return self.selector.key

    @property
    def insufficient_data(self) -> bool:
        return self.status == "insufficient_data"

    def metrics(self) -> dict[str, object]:
        return {
            "status": self.status,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "uncertainty": self.uncertainty,
            "average_verification_quality": self.average_verification_quality,
            "average_cost": self.average_cost,
            "average_latency_ms": self.average_latency_ms,
            "risk_event_count": self.risk_event_count,
            "failure_counts": dict(self.failure_counts),
            "insufficient_data": self.insufficient_data,
        }


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SelfModelError("timestamp must be ISO-8601 UTC") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise SelfModelError("timestamp must be UTC")
    return parsed


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SelfModelError(f"{field} must be a non-blank string")
    return value


def _token(value: object, *, field: str) -> str:
    normalized = _text(value, field=field).lower()
    if not _TOKEN.fullmatch(normalized):
        raise SelfModelError(f"{field} must be a stable lowercase token")
    return normalized


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise SelfModelError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _rate(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise SelfModelError(f"{field} must be a finite number between 0 and 1")
    return float(value)


def _optional_non_negative(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise SelfModelError(f"{field} must be a finite non-negative number or null")
    return float(value)
