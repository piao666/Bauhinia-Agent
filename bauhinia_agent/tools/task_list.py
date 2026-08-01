"""Read the authoritative task plan and derived projection."""

from __future__ import annotations

from bauhinia_agent.planning.projection import project_plan
from bauhinia_agent.planning.service import TaskPlanService
from bauhinia_agent.providers.types import ToolDefinition
from bauhinia_agent.tools.task_plan_support import format_task_plan_snapshot
from bauhinia_agent.tools.types import Tool, make_text_result
from bauhinia_agent.utils.schema import object_schema


def create_task_list_tool(service: TaskPlanService) -> Tool:
    def task_list():
        plan = service.current()
        if plan is None:
            return make_text_result(
                "task_list",
                "No task plan exists. Create one with task_create.",
                revision=0,
                plan=None,
                projection=None,
            )
        projection = project_plan(plan)
        return make_text_result(
            "task_list",
            format_task_plan_snapshot(plan, projection),
            revision=plan.revision,
            plan=plan.to_dict(),
            projection=projection,
        )

    parameters = object_schema({})
    parameters["additionalProperties"] = False
    return Tool(
        definition=ToolDefinition(
            name="task_list",
            description="Read the authoritative task-plan revision, snapshot, and derived projection.",
            parameters=parameters,
        ),
        executor=task_list,
    )
