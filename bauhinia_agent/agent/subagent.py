"""Subagent runner for the Phase 2 delegate tool.

The first implementation keeps the boundary deliberately small:

- child sessions are fresh and metadata-tagged;
- tool access is profile restricted;
- children do not receive the delegate tool, preventing recursion;
- foreground execution returns only a compact summary to the parent;
- background execution is handled by the Phase 1 generic async runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from bauhinia_agent.agent.capability_scope import CapabilityScopeError, scope_tools
from bauhinia_agent.agent.loop_limits import AgentLoopLimits
from bauhinia_agent.agent.session import AgentSession
from bauhinia_agent.agent.worktree import Worktree, WorktreeDiff, WorktreeError, WorktreeManager
from bauhinia_agent.context.identity import new_session_id
from bauhinia_agent.context.store import JsonlSessionStore
from bauhinia_agent.permissions.grants import PermissionGrantStore
from bauhinia_agent.permissions.manager import PermissionManager
from bauhinia_agent.permissions.policy import DefaultPermissionPolicy
from bauhinia_agent.permissions.types import PermissionAction, PermissionGrant, PermissionMode, PermissionScopeType
from bauhinia_agent.providers.base import ChatProvider
from bauhinia_agent.providers.types import MainRequestOptions
from bauhinia_agent.runtime.cancellation import AgentCancelledError, CancellationToken
from bauhinia_agent.skills.models import SkillCatalog
from bauhinia_agent.tools.permission_registry import PermissionAwareToolRegistry
from bauhinia_agent.tools.registry import ToolRegistry
from bauhinia_agent.tools.types import Tool
from bauhinia_agent.utils.sandbox_access import SandboxAccess, SandboxAccessMode

if TYPE_CHECKING:
    from bauhinia_agent.agent.evo_observer import AgentEvoObserver, AgentEvoRunResult

SubagentRole = Literal["researcher", "reviewer", "tester", "coder"]

READ_ONLY_TOOL_NAMES = frozenset(
    {
        "ls",
        "view",
        "grep",
        "glob",
        "tree",
        "read_multi",
        "git_status",
        "git_diff",
        "git_log",
        "think",
        "retrieve_archive",
    }
)
REVIEWER_TOOL_NAMES = frozenset({"view", "grep", "git_status", "git_diff", "git_log", "read_multi", "think", "retrieve_archive"})
TESTER_TOOL_NAMES = READ_ONLY_TOOL_NAMES | frozenset({"diagnostics", "shell", "python_exec"})
CODER_TOOL_NAMES = TESTER_TOOL_NAMES | frozenset({"write", "edit", "delete", "apply_patch"})


@dataclass(frozen=True, slots=True)
class SubagentProfile:
    role: SubagentRole
    description: str
    allowed_tool_names: frozenset[str]
    allow_background: bool = True
    requires_worktree: bool = False


SUBAGENT_PROFILES: dict[str, SubagentProfile] = {
    "researcher": SubagentProfile(
        role="researcher",
        description="Read-only codebase exploration and evidence collection.",
        allowed_tool_names=READ_ONLY_TOOL_NAMES,
    ),
    "reviewer": SubagentProfile(
        role="reviewer",
        description="Read-only review of diffs, call sites, and risks.",
        allowed_tool_names=REVIEWER_TOOL_NAMES,
    ),
    "tester": SubagentProfile(
        role="tester",
        description="Validation-focused investigation with diagnostics and approved execution tools.",
        allowed_tool_names=TESTER_TOOL_NAMES,
    ),
    "coder": SubagentProfile(
        role="coder",
        description=("Implementation work. Background coding runs inside an isolated git worktree " "so it can never mutate the parent working tree."),
        allowed_tool_names=CODER_TOOL_NAMES,
        allow_background=True,
        requires_worktree=True,
    ),
}


@dataclass(slots=True)
class SubagentRequest:
    role: SubagentRole
    task: str
    parent_session_id: str
    parent_task_hash: str | None = None
    parent_summary: str | None = None
    path_hints: list[str] = field(default_factory=list)
    run_in_background: bool = False
    isolate_worktree: bool = False
    parent_run_id: str | None = None
    child_run_id: str | None = None
    max_tool_rounds: int | None = None
    max_output_tokens: int | None = None
    # Runtime-only contract narrowing. ``None`` preserves the legacy profile
    # default; an explicit empty collection grants no capability/effect.
    allowed_tool_names: tuple[str, ...] | None = None
    allowed_effects: tuple[str, ...] | None = None


@dataclass(slots=True)
class SubagentResult:
    ok: bool
    role: SubagentRole
    child_session_id: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None
    worktree_path: str | None = None
    worktree_branch: str | None = None
    diff_summary: str | None = None
    confidence: float = 0.0
    confidence_source: str = "unknown"
    confidence_source_event_id: str | None = None

    def to_data(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "child_session_id": self.child_session_id,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "files_changed": list(self.files_changed),
            "error": self.error,
            "worktree_path": self.worktree_path,
            "worktree_branch": self.worktree_branch,
            "diff_summary": self.diff_summary,
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
            "confidence_source_event_id": self.confidence_source_event_id,
        }


class SubagentRunner:
    """Create and run isolated child AgentLoop sessions for delegate."""

    def __init__(
        self,
        *,
        store: JsonlSessionStore,
        provider: ChatProvider,
        tools: list[Tool],
        project_root: str | Path | None = None,
        agents_md: str = "",
        skill_catalog: SkillCatalog | None = None,
        permission_manager: PermissionManager | None = None,
        sandbox_access: SandboxAccess | None = None,
        request_options: MainRequestOptions | None = None,
        limits: AgentLoopLimits | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.tools = list(tools)
        self.project_root = Path(project_root).resolve() if project_root is not None else None
        self.agents_md = agents_md
        self.skill_catalog = skill_catalog or SkillCatalog()
        self.permission_manager = permission_manager
        self.sandbox_access = sandbox_access or SandboxAccess()
        self.request_options = request_options or MainRequestOptions()
        self.limits = limits or AgentLoopLimits(max_tool_rounds=20, max_provider_calls=40, max_turn_seconds=600)
        self.profile_map = SUBAGENT_PROFILES

    def profile(self, role: str) -> SubagentProfile | None:
        return SUBAGENT_PROFILES.get(str(role))

    def tools_for_role(self, role: str) -> list[Tool]:
        profile = self.profile(role)
        if profile is None:
            return []
        return scope_tools(
            self.tools,
            profile_tool_names=profile.allowed_tool_names,
            allowed_tool_names=None,
            allowed_effects=None,
        )

    def tools_for_request(self, request: SubagentRequest) -> list[Tool]:
        """Return the parent-rooted tools allowed by profile and contract."""

        profile = self.profile(request.role)
        if profile is None:
            return []
        return self._scope_tools(self.tools, request=request, profile=profile)

    def validate_contract_scope(
        self,
        *,
        role: str,
        allowed_tool_names: tuple[str, ...] | None,
        allowed_effects: tuple[str, ...] | None,
    ) -> None:
        """Validate capability/Effect fields without executing or persisting."""

        profile = self.profile(role)
        if profile is None:
            raise CapabilityScopeError(f"unknown subagent role: {role}")
        request = SubagentRequest(
            role=profile.role,
            task="runtime scope validation",
            parent_session_id="session_scope_validation",
            allowed_tool_names=allowed_tool_names,
            allowed_effects=allowed_effects,
        )
        self._scope_tools(self.tools, request=request, profile=profile)

    def run(
        self,
        request: SubagentRequest,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> SubagentResult:
        profile = self.profile(request.role)
        if profile is None:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary=f"Unknown subagent role: {request.role}",
                error="unknown_role",
            )
        if cancellation_token is not None and cancellation_token.is_cancelled:
            return self._cancelled_result(request)
        try:
            # Validate unknown Effects before creating a child session/worktree.
            self._scope_tools(self.tools, request=request, profile=profile)
        except CapabilityScopeError as error:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary=f"Subagent contract was rejected: {error}",
                error="capability_scope_invalid",
            )
        if request.run_in_background and not profile.allow_background:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary=f"{request.role} 不支持后台执行。",
                error="background_not_allowed",
            )

        if self._needs_worktree(request, profile=profile):
            return self._run_isolated(
                request,
                profile=profile,
                cancellation_token=cancellation_token,
            )
        return self._run_inline(
            request,
            profile=profile,
            cancellation_token=cancellation_token,
        )

    def _needs_worktree(self, request: SubagentRequest, *, profile: SubagentProfile) -> bool:
        """Whether this run must execute inside an isolated git worktree.

        Mutation-capable roles isolate when running in the background so a
        background job can never touch the parent working tree.  Callers can also
        force isolation explicitly via ``request.isolate_worktree`` (used by the
        parent ToolExecutor when it backgrounds a coder delegate).
        """

        if request.isolate_worktree:
            return True
        return bool(profile.requires_worktree and request.run_in_background)

    def _run_inline(
        self,
        request: SubagentRequest,
        *,
        profile: SubagentProfile,
        cancellation_token: CancellationToken | None,
    ) -> SubagentResult:
        """Original Phase 2 behaviour: run the child against the parent-rooted tools."""

        child_session = self.create_child_session(request, profile=profile)
        prompt = self._child_prompt(request, profile=profile)
        from bauhinia_agent.agent.loop import AgentLoop

        observer = self._observer_for(child_session, request=request)
        loop = AgentLoop(
            session=child_session,
            provider=self.provider,
            tools=child_session.tool_registry.tools(),
            limits=self._limits_for(request),
            request_options=self._request_options_for(request),
            cancellation_token=cancellation_token,
            background_manager=None,
            enable_delegate_tool=False,
            evolution_observer=observer,
        )
        try:
            response = loop.run_user_turn(prompt)
        except AgentCancelledError:
            return self._cancelled_result(
                request,
                child_session_id=child_session.session_id,
                observer=observer,
            )
        except Exception as exc:  # noqa: BLE001 - delegate must return a tool result, not break parent loop
            if cancellation_token is not None and cancellation_token.is_cancelled:
                return self._cancelled_result(
                    request,
                    child_session_id=child_session.session_id,
                    observer=observer,
                )
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id=child_session.session_id,
                summary=f"Subagent failed: {exc}",
                error=str(exc),
            )
        if cancellation_token is not None and cancellation_token.is_cancelled:
            return self._cancelled_result(
                request,
                child_session_id=child_session.session_id,
                observer=observer,
            )
        content = response.content.strip() or "Subagent finished without text output."
        return self._completed_result(
            request,
            child_session_id=child_session.session_id,
            summary=content,
            observer=observer,
        )

    def _run_isolated(
        self,
        request: SubagentRequest,
        *,
        profile: SubagentProfile,
        cancellation_token: CancellationToken | None,
    ) -> SubagentResult:
        """Phase 4: run a mutation-capable child inside a dedicated git worktree.

        The child gets fresh tools rooted at the worktree and a child
        ``PermissionManager`` whose ``project_root`` is the worktree path, so it
        can never read or mutate the parent working tree.  On completion the
        worktree is left in place and a diff summary is returned for explicit
        parent review; nothing is auto-merged.
        """

        if cancellation_token is not None and cancellation_token.is_cancelled:
            return self._cancelled_result(request)
        if self.project_root is None:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary="无法隔离执行：未知项目根目录。",
                error="worktree_unavailable",
            )
        manager = WorktreeManager(self.project_root)
        if not manager.available():
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id="",
                summary="无法隔离执行：当前项目不是 git 仓库，后台 coder 需要 worktree 隔离。",
                error="worktree_unavailable",
            )

        session_id = new_session_id()
        try:
            worktree = manager.create(session_id)
        except WorktreeError as exc:
            return SubagentResult(
                ok=False,
                role=request.role,
                child_session_id=session_id,
                summary=f"创建隔离 worktree 失败：{exc}",
                error="worktree_create_failed",
            )

        try:
            child_session = self._create_isolated_child_session(request, profile=profile, worktree=worktree, session_id=session_id)
            prompt = self._child_prompt(request, profile=profile, worktree=worktree)
            from bauhinia_agent.agent.loop import AgentLoop

            observer = self._observer_for(child_session, request=request)
            loop = AgentLoop(
                session=child_session,
                provider=self.provider,
                tools=child_session.tool_registry.tools(),
                limits=self._limits_for(request),
                request_options=self._request_options_for(request),
                cancellation_token=cancellation_token,
                background_manager=None,
                enable_delegate_tool=False,
                evolution_observer=observer,
            )
            try:
                response = loop.run_user_turn(prompt)
            except AgentCancelledError:
                diff = manager.diff(worktree)
                return self._discard_failed_worktree(
                    manager,
                    worktree,
                    self._cancelled_result(
                        request,
                        child_session_id=session_id,
                        observer=observer,
                        diff=diff,
                        worktree=worktree,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - never break the parent loop
                diff = manager.diff(worktree)
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    return self._discard_failed_worktree(
                        manager,
                        worktree,
                        self._cancelled_result(
                            request,
                            child_session_id=session_id,
                            observer=observer,
                            diff=diff,
                            worktree=worktree,
                        ),
                    )
                return self._discard_failed_worktree(
                    manager,
                    worktree,
                    SubagentResult(
                        ok=False,
                        role=request.role,
                        child_session_id=session_id,
                        summary=f"隔离 coder 执行失败：{exc}",
                        error=str(exc),
                        files_changed=diff.files_changed,
                        worktree_path=str(worktree.path),
                        worktree_branch=worktree.branch,
                        diff_summary=diff.render(),
                    ),
                )
            diff = manager.diff(worktree)
            if cancellation_token is not None and cancellation_token.is_cancelled:
                return self._discard_failed_worktree(
                    manager,
                    worktree,
                    self._cancelled_result(
                        request,
                        child_session_id=session_id,
                        observer=observer,
                        diff=diff,
                        worktree=worktree,
                    ),
                )
            content = response.content.strip() or "Subagent finished without text output."
            if response.finish_reason == "waiting_for_user_input":
                return self._discard_failed_worktree(
                    manager,
                    worktree,
                    SubagentResult(
                        ok=False,
                        role=request.role,
                        child_session_id=session_id,
                        summary=f"隔离 coder 等待用户输入，无法在后台继续：{content}",
                        error="waiting_for_user_input",
                        files_changed=diff.files_changed,
                        worktree_path=str(worktree.path),
                        worktree_branch=worktree.branch,
                        diff_summary=diff.render(),
                    ),
                )
            summary = self._compose_isolated_summary(content, worktree=worktree, diff=diff)
            return self._completed_result(
                request,
                child_session_id=session_id,
                summary=summary,
                observer=observer,
                files_changed=diff.files_changed,
                worktree_path=str(worktree.path),
                worktree_branch=worktree.branch,
                diff_summary=diff.render(),
            )
        except Exception as exc:  # noqa: BLE001 - defensive: setup failures must not break parent loop
            return self._discard_failed_worktree(
                manager,
                worktree,
                SubagentResult(
                    ok=False,
                    role=request.role,
                    child_session_id=session_id,
                    summary=f"隔离执行初始化失败：{exc}",
                    error=str(exc),
                    worktree_path=str(worktree.path),
                    worktree_branch=worktree.branch,
                ),
            )

    def create_child_session(self, request: SubagentRequest, *, profile: SubagentProfile) -> AgentSession:
        session_id = new_session_id()
        child = AgentSession.create(
            store=self.store,
            session_id=session_id,
            agents_md=self.agents_md,
            skill_catalog=self.skill_catalog,
            tools=self._supplied_tools_for_child(request, profile=profile),
            permission_manager=self.permission_manager,
            sandbox_access=self.sandbox_access,
        )
        self._restrict_child_registry(child, request=request, profile=profile)
        child.writer.append_session_metadata_updated(
            parent_session_id=request.parent_session_id,
            parent_task_hash=request.parent_task_hash,
            parent_run_id=request.parent_run_id,
            child_run_id=request.child_run_id,
            delegate_role=profile.role,
            delegate_task=request.task,
        )
        return child

    def _supplied_tools_for_child(
        self,
        request: SubagentRequest,
        *,
        profile: SubagentProfile,
    ) -> list[Tool]:
        """Tools passed into AgentSession.create, excluding session-reserved tools.

        ``retrieve_archive`` is injected by ``create_session_tool_registry`` for
        each session, so passing the parent session's instance would violate the
        reserved-name guard.
        """

        return [tool for tool in self._scope_tools(self.tools, request=request, profile=profile) if tool.name != "retrieve_archive"]

    def _create_isolated_child_session(
        self,
        request: SubagentRequest,
        *,
        profile: SubagentProfile,
        worktree: Worktree,
        session_id: str,
    ) -> AgentSession:
        """Create a child session whose permissions/tools are rooted at the worktree.

        The child gets its own ``PermissionManager`` (or a private clone of the
        parent's) whose ``project_root`` is the worktree path, so every path,
        shell, and git decision is evaluated against the isolated tree instead of
        the parent working directory.
        """

        permission_manager = self._child_permission_manager(worktree.path)
        # PROJECT sandbox keeps every file tool physically confined to the worktree
        # root even though the policy auto-allows in-tree writes.
        sandbox_access = SandboxAccess(mode=SandboxAccessMode.PROJECT)
        child = AgentSession.create(
            store=self.store,
            session_id=session_id,
            agents_md=self.agents_md,
            skill_catalog=self.skill_catalog,
            tools=self._worktree_child_tools(
                worktree.path,
                request=request,
                profile=profile,
                access=sandbox_access,
                for_registry=True,
            ),
            permission_manager=permission_manager,
            sandbox_access=sandbox_access,
        )
        self._restrict_child_registry(child, request=request, profile=profile)
        # Background isolated coder has no interactive user, so per-write review
        # confirmations would deadlock the job.  The worktree diff is reviewed by the
        # parent instead, so disable the pausing prewrite-review path here.
        child.require_prewrite_review = False
        child.writer.append_session_metadata_updated(
            parent_session_id=request.parent_session_id,
            parent_task_hash=request.parent_task_hash,
            parent_run_id=request.parent_run_id,
            child_run_id=request.child_run_id,
            delegate_role=profile.role,
            delegate_task=request.task,
            worktree_path=str(worktree.path),
            worktree_branch=worktree.branch,
        )
        return child

    def _child_permission_manager(self, root) -> PermissionManager:
        """Build an autonomous permission manager scoped to the worktree root.

        The policy root is the worktree path, so every path/shell/git decision is
        evaluated against the isolated tree.  AGGRESSIVE mode auto-allows in-tree
        writes and a safe validation-command allow-list so the background coder can
        make progress without an interactive user, while sensitive paths, deletes,
        and dangerous shell still require confirmation (and will simply pause the
        job rather than escape the sandbox).  A fresh grant store avoids inheriting
        parent-scoped grants.
        """

        grants = PermissionGrantStore()
        root_value = str(root)
        for action in (PermissionAction.WRITE_PATH, PermissionAction.DELETE_PATH):
            grants.add(
                PermissionGrant(
                    id=f"grant_subagent_{action.value}",
                    effect="allow",
                    action=action,
                    scope_type=PermissionScopeType.PATH_TREE,
                    scope_value=root_value,
                    created_at="runtime",
                    reason="Isolated background coder may mutate only its dedicated worktree.",
                )
            )
        return PermissionManager(
            policy=DefaultPermissionPolicy(root),
            grants=grants,
            mode=PermissionMode.AGGRESSIVE,
        )

    def _worktree_child_tools(
        self,
        root,
        *,
        request: SubagentRequest,
        profile: SubagentProfile,
        access: SandboxAccess,
        for_registry: bool = False,
    ) -> list[Tool]:
        """Build fresh tools rooted at the worktree for the child's role.

        Tools are rebuilt (not reused from the parent) so their internal path
        sandboxes point at the worktree, not the parent cwd.  ``for_registry``
        excludes session-reserved tool names when passing into ``AgentSession``.
        """

        from bauhinia_agent.tools.builtin import create_builtin_registry

        registry = create_builtin_registry(
            root,
            include_mutation_tools=True,
            include_execution_tools=True,
            include_network_tools=True,
            access=access,
        )
        tools = self._scope_tools(registry.tools(), request=request, profile=profile)
        if for_registry:
            tools = [tool for tool in tools if tool.name != "retrieve_archive"]
        return tools

    def _child_prompt(
        self,
        request: SubagentRequest,
        *,
        profile: SubagentProfile,
        worktree: Worktree | None = None,
    ) -> str:
        hints = "\n".join(f"- {hint}" for hint in request.path_hints if str(hint).strip())
        summary = request.parent_summary.strip() if request.parent_summary else "(none provided)"
        if worktree is not None:
            root = str(worktree.path)
            isolation = (
                "You are running inside an ISOLATED git worktree. All edits stay on branch "
                f"{worktree.branch} and never touch the parent working tree. Implement the task, "
                "then summarize what you changed. Do not attempt to merge or push.\n"
            )
        else:
            root = str(self.project_root) if self.project_root is not None else "(current project root)"
            isolation = ""
        return (
            f"You are a BauhiniaAgent subagent with role: {profile.role}.\n"
            f"Role scope: {profile.description}\n"
            f"Project root: {root}\n"
            f"{isolation}"
            "Do not call delegate or spawn nested subagents.\n"
            "Return a compact final report with: summary, evidence, files changed, and risks.\n\n"
            f"Parent summary:\n{summary}\n\n"
            f"Path hints:\n{hints or '(none)'}\n\n"
            f"Task:\n{request.task}"
        )

    def _scope_tools(
        self,
        tools,
        *,
        request: SubagentRequest,
        profile: SubagentProfile,
    ) -> list[Tool]:
        return scope_tools(
            tools,
            profile_tool_names=profile.allowed_tool_names,
            allowed_tool_names=request.allowed_tool_names,
            allowed_effects=request.allowed_effects,
        )

    def _restrict_child_registry(
        self,
        child: AgentSession,
        *,
        request: SubagentRequest,
        profile: SubagentProfile,
    ) -> None:
        """Remove session-injected tools outside the effective contract.

        ``AgentSession.create`` installs task/session helpers in addition to the
        supplied tools.  Rebuilding the same registry interface here is required
        so those helpers cannot bypass the contract intersection.
        """

        tools = self._scope_tools(
            child.tool_registry.tools(),
            request=request,
            profile=profile,
        )
        registry = ToolRegistry(tools)
        if child.permission_manager is None:
            child.tool_registry = registry
        else:
            child.tool_registry = PermissionAwareToolRegistry(
                registry,
                child.permission_manager,
            )

    def _observer_for(
        self,
        child: AgentSession,
        *,
        request: SubagentRequest,
    ) -> AgentEvoObserver | None:
        if request.child_run_id is None:
            return None
        from bauhinia_agent.agent.evo_observer import AgentEvoObserver

        return AgentEvoObserver(
            session=child,
            provider=self.provider,
            run_id=request.child_run_id,
            compile_candidates=False,
        )

    def _completed_result(
        self,
        request: SubagentRequest,
        *,
        child_session_id: str,
        summary: str,
        observer: AgentEvoObserver | None,
        files_changed: list[str] | None = None,
        worktree_path: str | None = None,
        worktree_branch: str | None = None,
        diff_summary: str | None = None,
    ) -> SubagentResult:
        observation = observer.last_result if observer is not None else None
        evidence = list(observation.evidence_ids) if observation is not None else []
        confidence, confidence_source, source_event_id = self._confidence_from(observation)
        ok = True
        error: str | None = None
        if observation is not None and observation.outcome in {
            "failure",
            "cancelled",
            "timeout",
        }:
            ok = False
            error = observation.outcome_category or observation.outcome
        return SubagentResult(
            ok=ok,
            role=request.role,
            child_session_id=child_session_id,
            summary=summary,
            evidence=evidence,
            files_changed=list(files_changed or ()),
            error=error,
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            diff_summary=diff_summary,
            confidence=confidence,
            confidence_source=confidence_source,
            confidence_source_event_id=source_event_id,
        )

    @staticmethod
    def _confidence_from(
        observation: AgentEvoRunResult | None,
    ) -> tuple[float, str, str | None]:
        if observation is None or observation.outcome_event_id is None or observation.outcome in {None, "unknown"} or observation.outcome_category in {None, "unknown"}:
            return 0.0, "unknown", None
        return (
            observation.outcome_confidence,
            "outcome_classified",
            observation.outcome_event_id,
        )

    def _cancelled_result(
        self,
        request: SubagentRequest,
        *,
        child_session_id: str = "",
        observer: AgentEvoObserver | None = None,
        diff: WorktreeDiff | None = None,
        worktree: Worktree | None = None,
    ) -> SubagentResult:
        observation = observer.last_result if observer is not None else None
        return SubagentResult(
            ok=False,
            role=request.role,
            child_session_id=child_session_id,
            summary="Subagent task was cancelled.",
            evidence=(list(observation.evidence_ids) if observation is not None else []),
            files_changed=list(diff.files_changed) if diff is not None else [],
            error="cancelled",
            worktree_path=str(worktree.path) if worktree is not None else None,
            worktree_branch=worktree.branch if worktree is not None else None,
            diff_summary=diff.render() if diff is not None else None,
            confidence=0.0,
            confidence_source="unknown",
            confidence_source_event_id=None,
        )

    @staticmethod
    def _discard_failed_worktree(
        manager: WorktreeManager,
        worktree: Worktree,
        result: SubagentResult,
    ) -> SubagentResult:
        """Best-effort reclaim of an isolated failed/cancelled execution."""

        try:
            manager.discard(worktree)
        except WorktreeError as error:
            cleanup = f"worktree_cleanup_failed: {error}"
            result.error = cleanup if result.error is None else f"{result.error}; {cleanup}"
            if not worktree.path.exists():
                result.worktree_path = None
            return result
        result.worktree_path = None
        result.worktree_branch = None
        return result

    def _limits_for(self, request: SubagentRequest) -> AgentLoopLimits:
        requested = request.max_tool_rounds
        if requested is None:
            return self.limits
        configured = self.limits.max_tool_rounds
        return replace(self.limits, max_tool_rounds=requested if configured is None else min(requested, configured))

    def _request_options_for(self, request: SubagentRequest) -> MainRequestOptions:
        requested = request.max_output_tokens
        if requested is None:
            return self.request_options
        configured = self.request_options.max_tokens
        return replace(self.request_options, max_tokens=requested if configured is None else min(requested, configured))

    def _compose_isolated_summary(self, content: str, *, worktree: Worktree, diff: "WorktreeDiff") -> str:
        parts = [
            content,
            "",
            "--- isolated worktree ---",
            f"path: {worktree.path}",
            f"branch: {worktree.branch}",
            "diff:",
            diff.render(),
        ]
        return "\n".join(parts)
