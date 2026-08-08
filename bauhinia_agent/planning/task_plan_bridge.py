"""Adapter that mirrors the existing session TaskPlan into a P2 PlanGraph.

It is deliberately one way: the TaskPlan service remains the authoritative
session control plane.  P2 recording failures return diagnostics and never
roll back an already accepted TaskPlan mutation.
"""

from __future__ import annotations

import hashlib

from bauhinia_agent.evolution.store import EvoEventStore
from bauhinia_agent.planning.evo import EvoPlanningService, PlanBudget, PlanGraph, PlanNode
from bauhinia_agent.planning.models import TaskPlan


class TaskPlanEvoBridge:
    """Record versions of one session's canonical TaskPlan as Evo plan facts."""

    def __init__(self, *, root: str, session_id: str) -> None:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        self._plan_id = f"plan_session_{digest}"
        self._service = EvoPlanningService(
            store=EvoEventStore(root),
            run_id=f"run_session_{digest}",
            session_id=f"session_{digest}",
        )

    def observe(self, plan: TaskPlan, operation: str) -> str | None:
        """Best-effort mirror; return a structured-text diagnostic on failure."""

        try:
            current = self._service.get(self._plan_id)
            graph = self._graph(plan, version=0 if current is None else current.version + 1)
            if current is None:
                self._service.create(graph)
            else:
                self._service.synchronize(graph, expected_version=current.version, change_summary=f"task_plan:{operation}")
        except Exception as error:  # recorder failures must not change task-plan semantics
            return f"evo_plan_recording_failed: {type(error).__name__}: {error}"
        return None

    def _graph(self, plan: TaskPlan, *, version: int) -> PlanGraph:
        ids = {task.id: self._node_id(task.id) for task in plan.tasks}
        nodes = tuple(
            PlanNode(
                node_id=ids[task.id],
                goal=task.content,
                depends_on=tuple(ids[dependency] for dependency in task.depends_on),
                preconditions=("session task dependencies are completed",) if task.depends_on else (),
                risks=("session task is subject to the existing PermissionManager",),
                budget=PlanBudget(max_attempts=0),
                verification_conditions=("linked session task reaches completed",),
                status=task.status,
            )
            for task in plan.tasks
        )
        return PlanGraph(plan_id=self._plan_id, goal="Session task plan", version=version, nodes=nodes)

    def _node_id(self, task_id: str) -> str:
        digest = hashlib.sha256(f"{self._plan_id}:{task_id}".encode("utf-8")).hexdigest()[:16]
        return f"node_task_{digest}"
