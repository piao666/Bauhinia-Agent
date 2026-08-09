from __future__ import annotations

from datetime import UTC, datetime

from bauhinia_agent.self_model import PolicySuggestionEngine, ProfileSelector, SelfModelProfile


def _profile(*, status: str, sample_count: int, risk: str = "low", failures: tuple[tuple[str, int], ...] = ()) -> SelfModelProfile:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    return SelfModelProfile(
        selector=ProfileSelector(
            project_id="project_a",
            model_config_hash="a" * 64,
            evaluator_version="eval-v1",
            environment_hash="b" * 64,
            language="python",
            task_type="bugfix",
            risk_level=risk,
        ),
        sample_count=sample_count,
        success_count=0 if status == "unreliable" else sample_count,
        success_rate=0.0 if status == "unreliable" else 1.0 if sample_count else None,
        confidence_low=0.0 if sample_count else None,
        confidence_high=0.4 if status == "unreliable" else 1.0 if sample_count else None,
        uncertainty=0.2 if sample_count else None,
        confidence=0.6 if sample_count >= 5 else 0.0,
        status=status,
        window_start=now,
        window_end=now,
        average_verification_quality=1.0 if sample_count else None,
        average_cost=None,
        average_latency_ms=None,
        risk_event_count=1 if risk == "high" else 0,
        failure_counts=failures,
        source_event_ids=("event_observation_1",) if sample_count else (),
        source_run_ids=("run_1",) if sample_count else (),
        published_event_id="event_profile_1",
    )


def test_low_sample_suggestions_are_explainable_replayable_and_disableable() -> None:
    profile = _profile(status="insufficient_data", sample_count=2)
    engine = PolicySuggestionEngine()

    first = engine.suggest(profile)
    replay = engine.suggest(profile)
    engine.set_enabled(False)
    disabled = engine.suggest(profile)

    assert first == replay
    assert first.status == "suggested"
    assert [item.action for item in first.suggestions] == ["increase_verification"]
    assert all(item.rationale and item.source_event_ids == profile.source_event_ids for item in first.suggestions)
    assert disabled.status == "disabled" and disabled.suggestions == ()


def test_unreliable_high_risk_profile_only_proposes_conservative_actions() -> None:
    profile = _profile(
        status="unreliable",
        sample_count=5,
        risk="high",
        failures=(("permission_denied", 2), ("timeout", 1), ("tool_failure", 2)),
    )

    result = PolicySuggestionEngine().suggest(profile)
    actions = {item.action for item in result.suggestions}

    assert {
        "use_conservative_template",
        "decompose_task",
        "increase_verification",
        "request_user_confirmation",
        "reduce_concurrency",
    } <= actions
    assert all(item.permission_effect == "none" for item in result.suggestions)
    assert not actions.intersection({"grant_permission", "execute_tool", "enable_network"})


def test_reliable_low_risk_profile_does_not_invent_a_warning() -> None:
    result = PolicySuggestionEngine().suggest(_profile(status="reliable", sample_count=20))

    assert result.status == "no_change"
    assert result.suggestions == ()
