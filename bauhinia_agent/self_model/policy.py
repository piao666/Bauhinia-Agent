"""Deterministic, explainable policy suggestions derived from Self Model profiles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from bauhinia_agent.self_model.models import SelfModelProfile

SuggestionAction = Literal[
    "decompose_task",
    "increase_verification",
    "use_conservative_template",
    "request_user_confirmation",
    "reduce_concurrency",
]
SuggestionSeverity = Literal["info", "warning", "high"]


@dataclass(frozen=True, slots=True)
class PolicySuggestion:
    suggestion_id: str
    action: SuggestionAction
    severity: SuggestionSeverity
    reason_code: str
    rationale: str
    profile_key: str
    profile_event_id: str | None
    source_event_ids: tuple[str, ...]
    permission_effect: Literal["none"] = "none"


@dataclass(frozen=True, slots=True)
class PolicySuggestionResult:
    enabled: bool
    profile_key: str
    suggestions: tuple[PolicySuggestion, ...]
    status: Literal["disabled", "no_change", "suggested"]


class PolicySuggestionEngine:
    """Suggest more conservative planning; it has no execution or permission port."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def suggest(self, profile: SelfModelProfile) -> PolicySuggestionResult:
        if not self._enabled:
            return PolicySuggestionResult(False, profile.profile_key, (), "disabled")

        proposed: list[tuple[SuggestionAction, SuggestionSeverity, str, str]] = []
        if profile.status == "insufficient_data":
            proposed.append(
                (
                    "increase_verification",
                    "warning",
                    "insufficient_samples",
                    f"Only {profile.sample_count} scoped sample(s) are available; add deterministic verification before relying on this profile.",
                )
            )
        elif profile.status == "unreliable":
            proposed.extend(
                (
                    (
                        "use_conservative_template",
                        "high",
                        "low_reliability",
                        "The upper confidence bound remains below the reliability threshold; use a conservative plan template.",
                    ),
                    (
                        "decompose_task",
                        "warning",
                        "low_reliability",
                        "Split the task into independently verifiable steps to reduce the failure surface.",
                    ),
                    (
                        "increase_verification",
                        "high",
                        "low_reliability",
                        "Require stronger deterministic verification because this scoped category has repeatedly failed.",
                    ),
                )
            )
        elif profile.status == "mixed":
            proposed.append(
                (
                    "increase_verification",
                    "warning",
                    "uncertain_reliability",
                    "The confidence interval crosses reliability thresholds; strengthen verification until uncertainty narrows.",
                )
            )

        failure_counts = dict(profile.failure_counts)
        if profile.selector.risk_level == "high" or profile.risk_event_count:
            proposed.append(
                (
                    "request_user_confirmation",
                    "high",
                    "high_risk_scope",
                    "High-risk scope or observed risk events require explicit user confirmation; this suggestion does not grant permission.",
                )
            )
        if failure_counts.get("permission_denied", 0):
            proposed.append(
                (
                    "request_user_confirmation",
                    "warning",
                    "permission_denials_observed",
                    "Prior runs were denied by the permission layer; confirm intent instead of retrying or broadening access.",
                )
            )
        if failure_counts.get("timeout", 0) or failure_counts.get("cancelled", 0):
            proposed.append(
                (
                    "reduce_concurrency",
                    "warning",
                    "completion_instability",
                    "Timeout or cancellation evidence suggests using fewer concurrent steps and smaller bounded operations.",
                )
            )
        if failure_counts.get("tool_failure", 0) or failure_counts.get("environment_failure", 0):
            proposed.append(
                (
                    "use_conservative_template",
                    "warning",
                    "tool_or_environment_failures",
                    "Use a plan that validates tool and environment preconditions before execution.",
                )
            )

        suggestions = _deduplicate(profile, proposed)
        return PolicySuggestionResult(True, profile.profile_key, suggestions, "suggested" if suggestions else "no_change")


def _deduplicate(
    profile: SelfModelProfile,
    proposed: list[tuple[SuggestionAction, SuggestionSeverity, str, str]],
) -> tuple[PolicySuggestion, ...]:
    seen: set[tuple[str, str]] = set()
    suggestions: list[PolicySuggestion] = []
    for action, severity, reason_code, rationale in proposed:
        identity = (action, reason_code)
        if identity in seen:
            continue
        seen.add(identity)
        digest = hashlib.sha256(f"{profile.profile_key}:{action}:{reason_code}".encode("utf-8")).hexdigest()[:16]
        suggestions.append(
            PolicySuggestion(
                suggestion_id=f"suggestion_{digest}",
                action=action,
                severity=severity,
                reason_code=reason_code,
                rationale=rationale,
                profile_key=profile.profile_key,
                profile_event_id=profile.published_event_id,
                source_event_ids=profile.source_event_ids,
            )
        )
    return tuple(suggestions)
