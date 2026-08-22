from __future__ import annotations

from dataclasses import dataclass, field

from bauhinia_agent.agent.session import AgentSession
from bauhinia_agent.app.runtime import AgentChatRunner, CurrentSessionState
from bauhinia_agent.app.self_model_commands import SelfModelCommandHandler
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.evolution import (
    EvidenceAdapter,
    EvidenceInput,
    EvoEventStore,
    OutcomeClassifiedPayload,
    OutcomeClassifier,
    SelfModelObservationRecordedPayload,
    SelfModelUpdatedPayload,
    new_evo_id,
)
from bauhinia_agent.permissions.manager import PermissionManager
from bauhinia_agent.permissions.policy import DefaultPermissionPolicy
from bauhinia_agent.permissions.types import PermissionMode
from bauhinia_agent.providers.base import ChatProvider
from bauhinia_agent.providers.types import ChatRequest, ChatResponse, ToolCall
from bauhinia_agent.self_model import (
    RuntimeTaskClassifier,
    SelfModelRuntime,
    SelfModelService,
)
from bauhinia_agent.tools.types import Tool


@dataclass
class _Provider(ChatProvider):
    responses: list[ChatResponse]
    requests: list[ChatRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class _FailingService:
    def build_profile(self, _selector):
        raise OSError("profile store unavailable")

    def publish_profile(self, _selector, *, run_id=None):
        del run_id
        raise OSError("profile store unavailable")

    def record_observation(self, _classification, *, source_event_id):
        del source_event_id
        raise OSError("profile store unavailable")


def _classifier() -> RuntimeTaskClassifier:
    return RuntimeTaskClassifier(
        project_id="project_runtime",
        environment_hash="e" * 64,
        project_language="python",
        repository_scale="small",
    )


def _runtime(tmp_path) -> tuple[SelfModelRuntime, SelfModelService, EvoEventStore]:
    store = EvoEventStore(tmp_path)
    service = SelfModelService(store=store, project_id="project_runtime")
    return SelfModelRuntime(service=service, classifier=_classifier()), service, store


def _classification(classifier: RuntimeTaskClassifier, text: str):
    return classifier.classify(
        text,
        provider_name="fake",
        provider_model="fake-model",
        request_options={"temperature": None, "max_tokens": None, "extra_body": {}},
    )


def _seed_outcome(
    store: EvoEventStore,
    service: SelfModelService,
    classification,
    *,
    success: bool,
) -> str:
    run_id = new_evo_id("run")
    evidence = EvidenceAdapter(store).record(
        EvidenceInput(
            run_id=run_id,
            evidence_type="test",
            source="pytest",
            summary="tests passed" if success else "tests failed",
            verified=True,
            command="pytest -q",
            exit_code=0 if success else 1,
        )
    )
    assert evidence.persisted
    outcome = OutcomeClassifier(store).classify(run_id)
    assert outcome.persisted and outcome.outcome is not None
    observation = service.record_observation(
        classification,
        source_event_id=outcome.outcome.event_id,
    )
    assert observation.persisted
    return run_id


def _verification_tool(root) -> Tool:
    from bauhinia_agent.tools.shell import create_shell_tool

    tests_root = root / "tests"
    tests_root.mkdir(exist_ok=True)
    (tests_root / "__init__.py").write_text("", encoding="utf-8")
    (tests_root / "test_runtime.py").write_text(
        "import unittest\n\nclass RuntimeTest(unittest.TestCase):\n" "    def test_runtime(self):\n        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    return create_shell_tool(root)


def test_low_sample_snapshot_is_auditable_and_only_increases_caution(tmp_path) -> None:
    runtime, _service, store = _runtime(tmp_path)

    snapshot = runtime.prepare_task(
        "Run Python pytest verification.",
        provider_name="fake",
        provider_model="fake-model",
        request_options={"temperature": None, "max_tokens": None, "extra_body": {}},
        run_id="run_current",
    )

    assert snapshot.profile is not None
    assert snapshot.profile.status == "insufficient_data"
    assert snapshot.profile.sample_count == 0
    assert snapshot.profile_event_id is not None
    assert [item.action for item in snapshot.suggestions] == ["increase_verification"]
    assert all(item.permission_effect == "none" for item in snapshot.suggestions)
    advisory = runtime.advisory_for(snapshot)
    assert advisory is not None
    assert "not a user message or permission grant" in advisory
    assert "samples=0" in advisory
    assert "uncertainty=insufficient_data" in advisory
    assert snapshot.profile_event_id in advisory
    rendered = runtime.render_user_snapshot()
    assert "Samples: 0 (0 successful)" in rendered
    assert "95% confidence interval: insufficient_data" in rendered
    published = [event for event in store.list_events() if isinstance(event.payload, SelfModelUpdatedPayload)]
    assert len(published) == 1
    assert published[0].refs.run_id == "run_current"
    assert published[0].payload.extensions["source_event_ids"] == []


def test_disabled_runtime_does_not_publish_inject_or_observe_and_command_can_reenable(tmp_path) -> None:
    runtime, service, store = _runtime(tmp_path)
    classification = _classification(_classifier(), "Run Python pytest verification.")
    source_run = _seed_outcome(store, service, classification, success=True)
    source_outcome = next(event for event in store.list_events() if event.refs.run_id == source_run and isinstance(event.payload, OutcomeClassifiedPayload))
    before_events = tuple(store.list_events())
    handler = SelfModelCommandHandler(runtime)

    disabled = handler.handle("/self-model off")
    snapshot = runtime.prepare_task(
        "Run Python pytest verification.",
        provider_name="fake",
        provider_model="fake-model",
        run_id="run_disabled",
    )
    receipt = runtime.record_completed_outcome(snapshot, outcome_event_id=source_outcome.event_id)

    assert disabled.handled and "process:project:project_runtime" in disabled.output
    assert runtime.enabled is False
    assert snapshot.profile is None
    assert runtime.advisory_for(snapshot) is None
    assert receipt.recorded is False
    assert tuple(store.list_events()) == before_events
    assert handler.handle("/self-model on").handled
    assert runtime.enabled is True
    assert handler.handle("/self-model nonsense").output == "Usage: /self-model [on|off]"


def test_agent_loop_without_evo_run_does_not_inject_or_publish_unanchored_profile(tmp_path) -> None:
    runtime, _service, store = _runtime(tmp_path)
    session = AgentSession.create(
        store=JsonlSessionStore(tmp_path),
        session_id="session_self_model_unanchored",
        agents_md="",
    )
    provider = _Provider([ChatResponse(provider="fake", model="fake-model", content="Unanchored advice was omitted.")])
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        tools=[],
        evolution_enabled=False,
        self_model_runtime=runtime,
    )

    response = runner.run_user_turn("Inspect Python code.")

    assert response.content == "Unanchored advice was omitted."
    assert runtime.latest_snapshot is not None
    assert runtime.latest_snapshot.profile is None
    assert runtime.latest_snapshot.diagnostic is not None
    assert runtime.latest_snapshot.diagnostic.code == "self_model_run_unavailable"
    assert store.list_events() == []
    assert not any("Self Model planning advisory" in message.content for message in provider.requests[0].messages)


def test_real_agent_run_consumes_pre_run_profile_once_and_records_verified_outcome_without_permission_change(
    tmp_path,
) -> None:
    data_root = tmp_path / ".bauhinia-agent"
    runtime, service, store = _runtime(data_root)
    task = "Run Python pytest verification."
    classification = _classification(_classifier(), task)
    for _ in range(5):
        _seed_outcome(store, service, classification, success=False)

    tool = _verification_tool(tmp_path)
    permissions = PermissionManager(
        policy=DefaultPermissionPolicy(tmp_path),
        mode=PermissionMode.AGGRESSIVE,
    )
    session = AgentSession.create(
        store=JsonlSessionStore(data_root),
        session_id="session_self_model_runtime",
        agents_md="",
        tools=[tool],
        permission_manager=permissions,
    )
    provider = _Provider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_verify",
                        name="shell",
                        arguments={"command": "python -m unittest -q tests.test_runtime"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="Verification finished.",
            ),
        ]
    )
    before_mode = permissions.mode
    before_grants = permissions.grants.list()
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        tools=[tool],
        evolution_enabled=True,
        self_model_runtime=runtime,
    )

    response = runner.run_user_turn(task)

    assert response.content == "Verification finished."
    assert len(provider.requests) == 2
    advisory_messages = [message for message in provider.requests[0].messages if message.role == "system" and message.content.startswith("Self Model planning advisory")]
    assert len(advisory_messages) == 1
    assert "status=unreliable; samples=5" in advisory_messages[0].content
    assert "increase_verification" in advisory_messages[0].content
    assert not any(message.role == "user" and "Self Model planning advisory" in message.content for request in provider.requests for message in request.messages)
    assert [part.content for message in session.rebuild_view().messages if message.role == "user" for part in message.parts] == [task]
    events = store.list_events()
    profiles = [event for event in events if isinstance(event.payload, SelfModelUpdatedPayload)]
    outcomes = [event for event in events if isinstance(event.payload, OutcomeClassifiedPayload)]
    observations = [event for event in events if isinstance(event.payload, SelfModelObservationRecordedPayload)]
    current_profile = profiles[-1]
    current_outcome = outcomes[-1]
    current_observation = observations[-1]
    assert len(profiles) == 1
    assert current_profile.payload.sample_count == 5
    assert current_profile.refs.run_id == current_outcome.refs.run_id
    assert current_observation.refs.run_id == current_outcome.refs.run_id
    assert current_observation.payload.source_event_id == current_outcome.event_id
    assert len(observations) == 6
    assert permissions.mode is before_mode
    assert permissions.grants.list() == before_grants


def test_runtime_failures_are_diagnostic_and_do_not_change_agent_response_or_history(tmp_path) -> None:
    runtime = SelfModelRuntime(  # type: ignore[arg-type]
        service=_FailingService(),
        classifier=_classifier(),
    )
    session = AgentSession.create(
        store=JsonlSessionStore(tmp_path),
        session_id="session_self_model_failure",
        agents_md="",
    )
    provider = _Provider([ChatResponse(provider="fake", model="fake-model", content="Agent result remains intact.")])
    runner = AgentChatRunner(
        current_session=CurrentSessionState(session),
        provider=provider,
        tools=[],
        evolution_enabled=True,
        self_model_runtime=runtime,
    )

    response = runner.run_user_turn("Inspect Python code.")

    assert response.content == "Agent result remains intact."
    assert runtime.latest_snapshot is not None
    assert runtime.latest_snapshot.diagnostic is not None
    assert runtime.latest_snapshot.diagnostic.code == "self_model_prepare_failed"
    assert runtime.latest_receipt is not None
    assert runtime.latest_receipt.recorded is False
    assert [message.role for message in session.rebuild_view().messages] == ["user", "assistant"]
    assert not any("Self Model planning advisory" in message.content for message in provider.requests[0].messages)
