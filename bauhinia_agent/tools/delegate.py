"""Delegate tool for running restricted BauhiniaAgent subagents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bauhinia_agent.agent.subagent import (
    SUBAGENT_PROFILES,
    SubagentRequest,
    SubagentRunner,
    SubagentRole,
)
from bauhinia_agent.planning.evo import PlanGraphError, TaskContract
from bauhinia_agent.providers.types import ToolDefinition
from bauhinia_agent.runtime.cancellation import current_cancellation_token
from bauhinia_agent.tools.types import Tool, ToolResult, make_error_result, make_text_result
from bauhinia_agent.utils.schema import object_schema

CollaborationDispatcher = Callable[
    [TaskContract, str | None, str | None],
    ToolResult,
]


def create_delegate_tool(
    runner: SubagentRunner,
    *,
    parent_session_id: str,
    parent_task_hash: str | None = None,
    collaboration_dispatcher: CollaborationDispatcher | None = None,
) -> Tool:
    """Create the parent-facing delegate tool.

    Background execution itself is handled by ToolExecutor's generic Phase 1
    `run_in_background` control field.  The delegate executor keeps foreground
    semantics and receives cleaned arguments.
    """

    def delegate(
        role: str | None = None,
        task: str | None = None,
        parent_summary: str | None = None,
        path_hints: list[str] | None = None,
        isolate_worktree: bool = False,
        contract: dict[str, Any] | None = None,
        collaboration_id: str | None = None,
        assignment_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        # ``isolate_worktree`` is an internal control field injected by the parent
        # ToolExecutor when it backgrounds a mutation-capable role; it is not part of
        # the model-visible schema.
        if kwargs:
            return make_error_result("delegate", f"未知参数：{', '.join(sorted(kwargs))}")
        if contract is not None:
            if any(value is not None for value in (role, task, parent_summary, path_hints)) or isolate_worktree:
                return make_error_result(
                    "delegate",
                    "Structured contract delegation cannot be mixed with legacy role/task arguments.",
                    field="contract",
                )
            if collaboration_dispatcher is None:
                return make_error_result(
                    "delegate",
                    "Structured collaboration is unavailable for this Agent run.",
                    collaboration_unavailable=True,
                )
            try:
                resolved_contract = TaskContract.from_dict(contract)
            except (PlanGraphError, TypeError, ValueError) as error:
                return make_error_result(
                    "delegate",
                    f"Invalid task contract: {error}",
                    field="contract",
                )
            return collaboration_dispatcher(
                resolved_contract,
                collaboration_id,
                assignment_id,
            )
        if collaboration_id is not None or assignment_id is not None:
            return make_error_result(
                "delegate",
                "collaboration_id and assignment_id require contract.",
                field="contract",
            )
        normalized_role = str(role).strip()
        if normalized_role not in runner.profile_map:
            return make_error_result("delegate", f"未知子代理角色：{normalized_role}", role=normalized_role)
        normalized_task = str(task or "").strip()
        if not normalized_task:
            return make_error_result("delegate", "task 不能为空")
        hints = [str(item).strip() for item in path_hints or [] if str(item).strip()]
        request = SubagentRequest(
            role=normalized_role,  # type: ignore[arg-type]
            task=normalized_task,
            parent_session_id=parent_session_id,
            parent_task_hash=parent_task_hash,
            parent_summary=parent_summary,
            path_hints=hints,
            run_in_background=False,
            isolate_worktree=bool(isolate_worktree),
        )
        result = runner.run(
            request,
            cancellation_token=current_cancellation_token(),
        )
        if not result.ok:
            return make_error_result("delegate", result.summary, **result.to_data())
        return make_text_result(
            "delegate",
            _format_delegate_result(result.summary, result.child_session_id),
            **result.to_data(),
        )

    return Tool(
        definition=ToolDefinition(
            name="delegate",
            description=(
                "Run a restricted child BauhiniaAgent subagent with a fresh context. Use for independent "
                "research, review, validation, or isolated implementation work. Do not use for nested "
                "delegation. Prefer contract for evidence-governed collaboration; it schedules its own "
                "background job, so do not add run_in_background. Legacy role/task calls remain "
                "available as non-learning compatibility behavior."
            ),
            parameters=object_schema(
                {
                    "role": {
                        "type": "string",
                        "enum": ["researcher", "reviewer", "tester", "coder"],
                        "description": "Subagent profile to run.",
                    },
                    "task": {"type": "string", "description": "Concrete task for the child agent."},
                    "parent_summary": {
                        "type": "string",
                        "description": "Optional compact context from the parent.",
                    },
                    "path_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional workspace paths to inspect.",
                    },
                    "contract": _task_contract_schema(),
                    "collaboration_id": {
                        "type": "string",
                        "description": "Optional stable collaboration identifier.",
                    },
                    "assignment_id": {
                        "type": "string",
                        "description": "Optional stable assignment identifier.",
                    },
                },
            ),
        ),
        executor=delegate,
    )


def role_allows_background(role: str) -> bool:
    profile = SUBAGENT_PROFILES.get(str(role).strip())
    return bool(profile and profile.allow_background)


def role_requires_worktree(role: str) -> bool:
    profile = SUBAGENT_PROFILES.get(str(role).strip())
    return bool(profile and profile.requires_worktree)


def _format_delegate_result(summary: str, child_session_id: str) -> str:
    return f"Subagent {child_session_id} completed.\n\n{summary}"


def _task_contract_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Evidence-governed TaskContract. Use instead of legacy role/task.",
        "properties": {
            "role": {
                "type": "string",
                "enum": [
                    "planner",
                    "researcher",
                    "executor",
                    "verifier",
                    "critic",
                    "curator",
                ],
            },
            "plan_id": {"type": "string"},
            "node_id": {"type": "string"},
            "goal": {"type": "string"},
            "input_snapshot": {"type": "string"},
            "allowed_effects": {"type": "array", "items": {"type": "string"}},
            "expected_evidence": {"type": "array", "items": {"type": "string"}},
            "budget": {
                "type": "object",
                "properties": {
                    "max_tool_calls": {"type": ["integer", "null"]},
                    "max_attempts": {"type": "integer"},
                    "max_tokens": {"type": ["integer", "null"]},
                },
            },
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "resource_claims": {"type": "array", "items": {"type": "string"}},
            "minimum_confidence": {"type": "number"},
            "cancellation_mode": {
                "type": "string",
                "enum": ["cooperative", "terminate"],
            },
            "deadline_at": {"type": ["string", "null"]},
        },
        "required": [
            "role",
            "plan_id",
            "node_id",
            "goal",
            "input_snapshot",
            "allowed_effects",
            "expected_evidence",
            "budget",
        ],
    }
