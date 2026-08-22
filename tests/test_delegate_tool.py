from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from bauhinia_agent.agent.background import BackgroundJobManager
from bauhinia_agent.agent.evo_observer import AgentEvoObserver
from bauhinia_agent.agent.loop import AgentLoop
from bauhinia_agent.agent.session import AgentSession
from bauhinia_agent.agent.subagent import SubagentRequest, SubagentRunner
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.evolution import EvoEventStore
from bauhinia_agent.evolution.events import CollaborationTaskResultRecordedPayload
from bauhinia_agent.planning.evo import PlanBudget, TaskContract
from bauhinia_agent.providers.base import ChatProvider
from bauhinia_agent.providers.types import ChatRequest, ChatResponse, ProviderCapabilities, ToolCall, ToolDefinition
from bauhinia_agent.tools.delegate import create_delegate_tool
from bauhinia_agent.tools.types import Tool, ToolResult, make_text_result


@dataclass
class FakeProvider(ChatProvider):
    responses: list[ChatResponse]
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    requests: list[ChatRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, request: ChatRequest) -> ChatResponse:
        if request.tools == [] and request.tool_choice == "none" and request.max_tokens == 512:
            return ChatResponse(provider="fake", model="fake-model", content='{"decision":"uncertain","basis_message_id":"msg"}')
        self.requests.append(request)
        return self.responses.pop(0)


def _tool(name: str) -> Tool:
    def execute(text: str = "") -> ToolResult:
        return make_text_result(name, f"{name}:{text}")

    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"tool {name}",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        ),
        executor=execute,
    )


def _delegate_call(call_id: str, *, role: str, task: str, **extra) -> ToolCall:
    arguments = {"role": role, "task": task, **extra}
    return ToolCall(id=call_id, name="delegate", arguments=arguments)


def _create_task_plan(session: AgentSession, *, task_id: str) -> None:
    result = session.tool_registry.execute(
        "task_create",
        {
            "mode": "linear",
            "expected_revision": 0,
            "tasks": [{"id": task_id, "content": "Run delegated work", "status": "in_progress"}],
        },
    )
    assert result.ok is True


def _structured_contract() -> TaskContract:
    return TaskContract(
        role="verifier",
        plan_id="plan_delegate_contract",
        node_id="node_delegate_contract",
        goal="Run deterministic verification",
        input_snapshot="tree@abc123",
        allowed_effects=("execute", "write", "network"),
        expected_evidence=("test",),
        budget=PlanBudget(max_tool_calls=2, max_attempts=1, max_tokens=1000),
        capabilities=("shell",),
        resource_claims=("read:tests",),
        minimum_confidence=0.7,
    )


def test_delegate_structured_contract_uses_dispatcher_and_rejects_mixed_legacy_fields(
    tmp_path,
) -> None:
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=FakeProvider([]),
        tools=[_tool("shell")],
    )
    captured: list[tuple[TaskContract, str | None, str | None]] = []

    def dispatch(
        contract: TaskContract,
        collaboration_id: str | None,
        assignment_id: str | None,
    ) -> ToolResult:
        captured.append((contract, collaboration_id, assignment_id))
        return make_text_result("delegate", "queued", job_id="bg_0001")

    tool = create_delegate_tool(
        runner,
        parent_session_id="session_parent",
        collaboration_dispatcher=dispatch,
    )

    result = tool.executor(
        contract=_structured_contract().to_dict(),
        collaboration_id="collaboration_one",
        assignment_id="assignment_one",
    )

    assert result.ok is True
    assert captured == [(_structured_contract(), "collaboration_one", "assignment_one")]
    assert "required" not in tool.definition.parameters
    mixed = tool.executor(
        role="tester",
        task="legacy",
        contract=_structured_contract().to_dict(),
    )
    assert mixed.ok is False
    assert mixed.data["field"] == "contract"


def test_structured_delegate_rejects_generic_background_double_scheduling(
    tmp_path,
) -> None:
    manager = BackgroundJobManager()
    try:
        store = JsonlSessionStore(tmp_path)
        session = AgentSession.create(
            store=store,
            session_id="session_structured_background",
            tools=[_tool("shell")],
        )
        loop = AgentLoop(
            session=session,
            provider=FakeProvider([]),
            background_manager=manager,
        )
        call = ToolCall(
            id="call_structured_background",
            name="delegate",
            arguments={
                "contract": _structured_contract().to_dict(),
                "run_in_background": True,
            },
        )
        session.append_user_message("start")
        session.append_assistant_response(
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[call],
                finish_reason="tool_calls",
            )
        )

        loop.tool_executor.execute_interactive([call])

        tool_part = next(part for message in session.rebuild_view().messages for part in message.parts if part.kind == "tool_result")
        assert tool_part.metadata["ok"] is False
        assert tool_part.metadata["data"]["background_rejected"] == "structured_delegate_owns_scheduling"
        assert manager.list(session_id=session.session_id) == []
    finally:
        manager.shutdown()


def test_subagent_runner_filters_tools_by_profile(tmp_path) -> None:
    provider = FakeProvider([])
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=provider,
        tools=[_tool("view"), _tool("grep"), _tool("write"), _tool("delegate"), _tool("shell")],
    )

    assert [tool.name for tool in runner.tools_for_role("reviewer")] == ["view", "grep"]
    assert "delegate" not in [tool.name for tool in runner.tools_for_role("coder")]
    assert "write" in [tool.name for tool in runner.tools_for_role("coder")]
    assert "write" not in [tool.name for tool in runner.tools_for_role("researcher")]


def test_subagent_runner_creates_metadata_tagged_child_session(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    provider = FakeProvider([ChatResponse(provider="fake", model="fake-model", content="child done")])
    runner = SubagentRunner(store=store, provider=provider, tools=[_tool("view")])

    result = runner.run(
        SubagentRequest(
            role="researcher",
            task="inspect context",
            parent_session_id="parent_1",
            parent_task_hash="task_abc",
            path_hints=["bauhinia_agent/agent"],
        )
    )

    assert result.ok is True
    assert result.summary == "child done"
    view = store.rebuild_session_view(result.child_session_id)
    assert view.metadata["parent_session_id"] == "parent_1"
    assert view.metadata["parent_task_hash"] == "task_abc"
    assert view.metadata["delegate_role"] == "researcher"
    assert view.metadata["delegate_task"] == "inspect context"
    assert "delegate" not in [definition.name for definition in provider.requests[0].tools]
    assert "view" in [definition.name for definition in provider.requests[0].tools]


def test_agent_loop_registers_delegate_and_foreground_returns_summary(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    provider = FakeProvider([ChatResponse(provider="fake", model="fake-model", content="child summary")])
    session = AgentSession.create(store=store, session_id="parent_delegate", tools=[_tool("view")])
    loop = AgentLoop(session=session, provider=provider)

    assert "delegate" in session.tool_registry.names()
    result = session.tool_registry.execute("delegate", {"role": "researcher", "task": "read docs"})

    assert result.ok is True
    assert result.data["role"] == "researcher"
    assert result.data["child_session_id"]
    assert "child summary" in result.content


def test_real_agent_loop_structured_delegate_records_child_outcome_and_aggregate(
    tmp_path,
) -> None:
    from bauhinia_agent.permissions.manager import PermissionManager
    from bauhinia_agent.permissions.policy import DefaultPermissionPolicy
    from bauhinia_agent.permissions.types import PermissionMode
    from bauhinia_agent.tools.shell import create_shell_tool

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_contract.py").write_text(
        "import unittest\n\nclass ContractTest(unittest.TestCase):\n" "    def test_contract(self):\n        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    _init_git_repo(repo)
    data_root = repo / ".bauhinia-agent"
    store = JsonlSessionStore(data_root)
    shell = create_shell_tool(repo)
    provider = FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_child_test",
                        name="shell",
                        arguments={"command": "python -m unittest -q tests.test_contract"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="Structured child verified the contract.",
            ),
        ]
    )
    session = AgentSession.create(
        store=store,
        session_id="session_structured_parent",
        tools=[shell],
        permission_manager=PermissionManager(
            policy=DefaultPermissionPolicy(repo),
            mode=PermissionMode.BYPASS,
        ),
    )
    manager = BackgroundJobManager()
    observer = AgentEvoObserver(session=session, provider=provider)
    try:
        loop = AgentLoop(
            session=session,
            provider=provider,
            tools=[shell],
            background_manager=manager,
            evolution_observer=observer,
        )
        parent_run_id = loop.evolution_observer.begin_turn()  # type: ignore[union-attr]

        queued = session.tool_registry.execute(
            "delegate",
            {
                "contract": _structured_contract().to_dict(),
                "collaboration_id": "collaboration_real_loop",
                "assignment_id": "assignment_real_loop",
            },
        )

        assert queued.ok is True
        assert queued.data["status"] == "running"
        assert manager.wait(timeout=5) is True
        events = EvoEventStore(data_root).list_events()
        event_types = [event.event_type for event in events]
        assert "CollaborationTaskDelegated" in event_types
        assert "EvidenceRecorded" in event_types
        assert "OutcomeClassified" in event_types
        assert "CollaborationTaskResultRecorded" in event_types
        assert "CollaborationRunAggregated" in event_types
        result_event = next(event for event in events if isinstance(event.payload, CollaborationTaskResultRecordedPayload))
        assert result_event.refs.run_id == parent_run_id
        assert result_event.payload.child_run_id is not None
        assert result_event.payload.confidence == 0.95
        assert result_event.payload.eligible_for_learning is True
        assert result_event.payload.evidence_refs
        assert result_event.payload.claims
        child_outcome = next(event for event in events if event.event_type == "OutcomeClassified" and event.refs.run_id == result_event.payload.child_run_id)
        assert result_event.payload.confidence_source_event_id == child_outcome.event_id
    finally:
        manager.shutdown()


def test_background_delegate_returns_placeholder_and_notification(tmp_path) -> None:
    store = JsonlSessionStore(tmp_path)
    manager = BackgroundJobManager()
    gate = threading.Event()

    def child_response() -> ChatResponse:
        gate.wait(5)
        return ChatResponse(provider="fake", model="fake-model", content="background child done")

    class BlockingProvider(FakeProvider):
        def complete(self, request: ChatRequest) -> ChatResponse:
            if request.tools == [] and request.tool_choice == "none" and request.max_tokens == 512:
                return super().complete(request)
            self.requests.append(request)
            return child_response()

    provider = BlockingProvider([])
    try:
        session = AgentSession.create(store=store, session_id="parent_bg_delegate", tools=[_tool("view")])
        loop = AgentLoop(session=session, provider=provider, background_manager=manager)
        _create_task_plan(session, task_id="research_a")
        session.append_user_message("start")
        call = _delegate_call(
            "call_delegate",
            role="researcher",
            task="slow research",
            run_in_background=True,
            task_id="research_a",
        )
        session.append_assistant_response(ChatResponse(provider="fake", model="fake-model", content="", tool_calls=[call], finish_reason="tool_calls"))

        state = loop.tool_executor.execute_interactive([call])
        assert state.pending_input is None
        tool_result = [message for message in session.rebuild_view().messages if message.role == "tool"][0].parts[0]
        assert tool_result.metadata["data"]["background_job_id"] == "bg_0001"
        assert tool_result.metadata["data"]["task_id"] == "research_a"
        assert tool_result.metadata["data"]["observed_revision"] == 1

        gate.set()
        assert manager.wait(timeout=5) is True
        loop._append_background_notifications()
        notifications = [message.parts[0].content for message in session.rebuild_view().messages if message.role == "user" and "<task_notification>" in message.parts[0].content]
        assert len(notifications) == 1
        assert "<task_id>research_a</task_id>" in notifications[0]
        assert "background child done" in notifications[0]
        plan = session.rebuild_view().task_plan
        assert plan is not None
        assert plan.revision == 2
        assert plan.tasks[0].status == "completed"
    finally:
        gate.set()
        manager.wait(timeout=5)
        manager.shutdown()


def test_coder_delegate_background_rejected_without_git_repo(tmp_path) -> None:
    """Background coder needs worktree isolation; a non-git project must be refused.

    Phase 4 allows background coder only when a git worktree can be created.  With
    no permission manager / non-git root, isolation is unavailable, so the call is
    rejected up front with ``worktree_unavailable`` and no job is started.
    """

    store = JsonlSessionStore(tmp_path)
    manager = BackgroundJobManager()
    provider = FakeProvider([])
    try:
        session = AgentSession.create(store=store, session_id="parent_coder_bg", tools=[_tool("view")])
        loop = AgentLoop(session=session, provider=provider, background_manager=manager)
        session.append_user_message("start")
        call = _delegate_call("call_delegate", role="coder", task="edit files", run_in_background=True)
        session.append_assistant_response(ChatResponse(provider="fake", model="fake-model", content="", tool_calls=[call], finish_reason="tool_calls"))

        loop.tool_executor.execute_interactive([call])
        tool_result = [message for message in session.rebuild_view().messages if message.role == "tool"][0].parts[0]
        assert tool_result.metadata["ok"] is False
        assert tool_result.metadata["data"]["background_rejected"] == "worktree_unavailable"
        assert manager.list() == []
    finally:
        manager.shutdown()


def _init_git_repo(root) -> None:
    import subprocess

    def _git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)

    _git("init", "-q")
    _git("config", "user.email", "t@t.co")
    _git("config", "user.name", "t")
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-qm", "init")


def _write_call(call_id: str, path: str, content: str) -> ToolCall:
    return ToolCall(id=call_id, name="write", arguments={"path": path, "content": content})


def test_isolated_coder_writes_only_in_worktree(tmp_path) -> None:
    """Phase 4: a worktree-isolated coder mutates the worktree, never the parent tree."""

    from bauhinia_agent.permissions.manager import PermissionManager
    from bauhinia_agent.permissions.policy import DefaultPermissionPolicy
    from bauhinia_agent.permissions.types import PermissionMode
    from bauhinia_agent.agent.subagent import SubagentRequest, SubagentRunner

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    # Dirty parent proves isolation does not depend on a clean parent tree.
    (repo / "seed.txt").write_text("seed dirty\n", encoding="utf-8")

    write_call = _write_call("w1", "newfile.py", "print('hi')\n")
    provider = FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[write_call],
                finish_reason="tool_calls",
            ),
            ChatResponse(provider="fake", model="fake-model", content="Implemented newfile.py"),
        ]
    )
    store = JsonlSessionStore(repo / ".fc_sessions")
    permission_manager = PermissionManager(policy=DefaultPermissionPolicy(repo), mode=PermissionMode.STANDARD)
    runner = SubagentRunner(
        store=store,
        provider=provider,
        tools=[],
        project_root=repo,
        permission_manager=permission_manager,
    )

    result = runner.run(
        SubagentRequest(
            role="coder",
            task="create newfile.py",
            parent_session_id="p1",
            isolate_worktree=True,
        )
    )

    assert result.ok is True
    assert result.worktree_branch == f"fc/subagent/{result.child_session_id}"
    assert result.worktree_path is not None
    assert "newfile.py" in result.files_changed
    assert "newfile.py" in (result.diff_summary or "")
    # 隔离目录里有新文件，父工作区没有；父的 dirty 文件也没被动过。
    assert (Path(result.worktree_path) / "newfile.py").exists()
    assert not (repo / "newfile.py").exists()
    assert (repo / "seed.txt").read_text(encoding="utf-8").strip() == "seed dirty"


def test_isolated_coder_can_delete_inside_worktree_without_parent_delete(tmp_path) -> None:
    """DELETE_PATH is allowed only in the isolated worktree, never in the parent tree."""

    from bauhinia_agent.permissions.manager import PermissionManager
    from bauhinia_agent.permissions.policy import DefaultPermissionPolicy
    from bauhinia_agent.permissions.types import PermissionMode
    from bauhinia_agent.agent.subagent import SubagentRequest, SubagentRunner

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    delete_call = ToolCall(id="d1", name="delete", arguments={"path": "seed.txt"})
    provider = FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[delete_call],
                finish_reason="tool_calls",
            ),
            ChatResponse(provider="fake", model="fake-model", content="Deleted seed.txt"),
        ]
    )
    runner = SubagentRunner(
        store=JsonlSessionStore(repo / ".fc_sessions"),
        provider=provider,
        tools=[],
        project_root=repo,
        permission_manager=PermissionManager(policy=DefaultPermissionPolicy(repo), mode=PermissionMode.STANDARD),
    )

    result = runner.run(
        SubagentRequest(
            role="coder",
            task="delete seed.txt in isolation",
            parent_session_id="p1",
            isolate_worktree=True,
        )
    )

    assert result.ok is True
    assert "seed.txt" in result.files_changed
    assert result.worktree_path is not None
    assert not (Path(result.worktree_path) / "seed.txt").exists()
    assert (repo / "seed.txt").exists()


def test_isolated_coder_waiting_for_permission_is_failure_with_diff(tmp_path) -> None:
    """If a child still needs user input, background delegate must not report success."""

    from bauhinia_agent.agent.subagent import SubagentRequest, SubagentRunner

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    shell_call = ToolCall(id="s1", name="shell", arguments={"command": "rm seed.txt"})
    provider = FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[shell_call],
                finish_reason="tool_calls",
            )
        ]
    )
    runner = SubagentRunner(
        store=JsonlSessionStore(repo / ".fc_sessions"),
        provider=provider,
        tools=[],
        project_root=repo,
    )

    result = runner.run(
        SubagentRequest(
            role="coder",
            task="run dangerous shell",
            parent_session_id="p1",
            isolate_worktree=True,
        )
    )

    assert result.ok is False
    assert result.error == "waiting_for_user_input"
    assert result.worktree_path is None
    assert result.worktree_branch is None
    worktrees_root = repo / ".git" / "fc-worktrees"
    assert not worktrees_root.exists() or not any(worktrees_root.iterdir())
    assert (repo / "seed.txt").exists()


def test_background_coder_uses_worktree_and_leaves_parent_untouched(tmp_path) -> None:
    """Phase 4: background delegate for coder runs isolated and reports a diff summary."""

    from bauhinia_agent.permissions.manager import PermissionManager
    from bauhinia_agent.permissions.policy import DefaultPermissionPolicy
    from bauhinia_agent.permissions.types import PermissionMode

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    write_call = _write_call("w1", "bg_new.py", "x = 1\n")
    provider = FakeProvider(
        [
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[write_call],
                finish_reason="tool_calls",
            ),
            ChatResponse(provider="fake", model="fake-model", content="coder done"),
        ]
    )
    store = JsonlSessionStore(repo / ".fc_sessions")
    permission_manager = PermissionManager(policy=DefaultPermissionPolicy(repo), mode=PermissionMode.STANDARD)
    manager = BackgroundJobManager()
    try:
        session = AgentSession.create(store=store, session_id="parent_bg_coder", permission_manager=permission_manager)
        loop = AgentLoop(session=session, provider=provider, background_manager=manager)
        _create_task_plan(session, task_id="impl")
        session.append_user_message("start")
        call = _delegate_call(
            "call_delegate",
            role="coder",
            task="create bg_new.py",
            run_in_background=True,
            task_id="impl",
        )
        session.append_assistant_response(
            ChatResponse(
                provider="fake",
                model="fake-model",
                content="",
                tool_calls=[call],
                finish_reason="tool_calls",
            )
        )

        loop.tool_executor.execute_interactive([call])
        tool_result = [message for message in session.rebuild_view().messages if message.role == "tool"][0].parts[0]
        assert tool_result.metadata["ok"] is True
        assert tool_result.metadata["data"]["background_job_id"] == "bg_0001"
        assert tool_result.metadata["data"].get("background_rejected") is None

        assert manager.wait(timeout=10) is True
        loop._append_background_notifications()
        notifications = [message.parts[0].content for message in session.rebuild_view().messages if message.role == "user" and "<task_notification>" in message.parts[0].content]
        assert len(notifications) == 1
        assert "bg_new.py" in notifications[0]
        assert "<task_id>impl</task_id>" in notifications[0]
        # 后台 coder 的改动只在隔离 worktree，父工作区看不到。
        assert not (repo / "bg_new.py").exists()
    finally:
        manager.shutdown()


def test_isolated_coder_without_git_repo_returns_error(tmp_path) -> None:
    """When isolation is requested but the project is not a git repo, fail cleanly."""

    from bauhinia_agent.agent.subagent import SubagentRequest, SubagentRunner

    provider = FakeProvider([])
    runner = SubagentRunner(
        store=JsonlSessionStore(tmp_path),
        provider=provider,
        tools=[],
        project_root=tmp_path,
    )
    result = runner.run(
        SubagentRequest(
            role="coder",
            task="edit",
            parent_session_id="p1",
            isolate_worktree=True,
        )
    )
    assert result.ok is False
    assert result.error == "worktree_unavailable"
    # 没有真正调用 provider（在创建 worktree 前就返回了）。
    assert provider.requests == []
