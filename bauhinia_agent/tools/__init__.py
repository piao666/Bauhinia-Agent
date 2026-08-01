"""工具定义、注册和执行入口。"""

from bauhinia_agent.tools.builtin import create_builtin_registry
from bauhinia_agent.tools.apply_patch import create_apply_patch_tool
from bauhinia_agent.tools.ask_user import create_ask_user_tool
from bauhinia_agent.tools.delete import create_delete_tool
from bauhinia_agent.tools.diagnostics import create_diagnostics_tool
from bauhinia_agent.tools.edit import create_edit_tool
from bauhinia_agent.tools.fetch import create_fetch_tool
from bauhinia_agent.tools.git_diff import create_git_diff_tool
from bauhinia_agent.tools.git_log import create_git_log_tool
from bauhinia_agent.tools.git_status import create_git_status_tool
from bauhinia_agent.tools.glob import create_glob_tool
from bauhinia_agent.tools.grep import create_grep_tool
from bauhinia_agent.tools.ls import create_ls_tool
from bauhinia_agent.tools.python_exec import create_python_exec_tool
from bauhinia_agent.tools.read_multi import create_read_multi_tool
from bauhinia_agent.tools.shell import create_shell_tool
from bauhinia_agent.tools.task_boundary import create_task_boundary_tool
from bauhinia_agent.tools.registry import ToolRegistry
from bauhinia_agent.tools.think import create_think_tool
from bauhinia_agent.tools.tree import create_tree_tool
from bauhinia_agent.tools.types import Tool, ToolExecutor, ToolResult
from bauhinia_agent.tools.view import create_view_tool
from bauhinia_agent.tools.web_search import create_web_search_tool
from bauhinia_agent.tools.write import create_write_tool

__all__ = [
    "Tool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "create_apply_patch_tool",
    "create_ask_user_tool",
    "create_builtin_registry",
    "create_delete_tool",
    "create_delegate_tool",
    "create_diagnostics_tool",
    "create_edit_tool",
    "create_fetch_tool",
    "create_git_diff_tool",
    "create_git_log_tool",
    "create_git_status_tool",
    "create_glob_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_python_exec_tool",
    "create_read_multi_tool",
    "create_shell_tool",
    "create_task_boundary_tool",
    "create_think_tool",
    "create_task_create_tool",
    "create_task_update_tool",
    "create_task_revise_tool",
    "create_task_list_tool",
    "create_tree_tool",
    "create_view_tool",
    "create_web_search_tool",
    "create_write_tool",
]


def __getattr__(name: str):
    """Lazily expose tools that depend on agent-layer runtime objects.

    Importing ``bauhinia_agent.tools`` is part of low-level context/writer startup.
    ``delegate`` depends on ``bauhinia_agent.agent.subagent`` and therefore cannot be
    imported eagerly here without creating a package-level cycle.
    """

    if name == "create_delegate_tool":
        from bauhinia_agent.tools.delegate import create_delegate_tool

        return create_delegate_tool
    task_factories = {
        "create_task_create_tool": ("bauhinia_agent.tools.task_create", "create_task_create_tool"),
        "create_task_update_tool": ("bauhinia_agent.tools.task_update", "create_task_update_tool"),
        "create_task_revise_tool": ("bauhinia_agent.tools.task_revise", "create_task_revise_tool"),
        "create_task_list_tool": ("bauhinia_agent.tools.task_list", "create_task_list_tool"),
    }
    target = task_factories.get(name)
    if target is not None:
        from importlib import import_module

        return getattr(import_module(target[0]), target[1])
    raise AttributeError(name)
