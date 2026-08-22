"""Runtime-only capability narrowing for delegated Agent sessions.

The planning contract names effects independently from the provider-facing tool
schema.  Until tools expose a first-class Effect field, this module provides one
small fail-closed adapter over the existing tool name and permission metadata.
It never grants a capability: callers still intersect the result with the
subagent profile and the normal permission engine remains authoritative.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from bauhinia_agent.permissions.types import PermissionAction
from bauhinia_agent.tools.types import Tool

KNOWN_TOOL_EFFECTS = frozenset({"read", "write", "execute", "network", "external"})

# Names are needed for tools whose permission action is deliberately broader
# than their actual operation (for example read-only git inspection).
_NAMED_TOOL_EFFECTS: dict[str, frozenset[str]] = {
    "ls": frozenset({"read"}),
    "view": frozenset({"read"}),
    "grep": frozenset({"read"}),
    "glob": frozenset({"read"}),
    "tree": frozenset({"read"}),
    "read_multi": frozenset({"read"}),
    "git_status": frozenset({"read"}),
    "git_diff": frozenset({"read"}),
    "git_log": frozenset({"read"}),
    "think": frozenset({"read"}),
    "retrieve_archive": frozenset({"read"}),
    "diagnostics": frozenset({"execute", "write", "network"}),
    # Arbitrary interpreters can mutate files and initiate network traffic even
    # when the submitted command looks observational.  The normal Permission
    # Engine still evaluates the concrete call; the contract must declare every
    # capability that the interpreter can exercise.
    "shell": frozenset({"execute", "write", "network"}),
    "python_exec": frozenset({"execute", "write", "network"}),
    "write": frozenset({"write"}),
    "edit": frozenset({"write"}),
    "delete": frozenset({"write"}),
    "apply_patch": frozenset({"write"}),
    "fetch": frozenset({"network"}),
    "web_search": frozenset({"network"}),
    "ask_user": frozenset({"external"}),
}

_PERMISSION_EFFECTS: dict[PermissionAction, frozenset[str]] = {
    PermissionAction.READ_PATH: frozenset({"read"}),
    PermissionAction.READ_ENV: frozenset({"read"}),
    PermissionAction.WRITE_PATH: frozenset({"write"}),
    PermissionAction.DELETE_PATH: frozenset({"write"}),
    PermissionAction.EXECUTE_SHELL: frozenset({"execute", "write", "network"}),
    PermissionAction.NETWORK_REQUEST: frozenset({"network"}),
    PermissionAction.MCP_TOOL: frozenset({"external"}),
}


class CapabilityScopeError(ValueError):
    """A runtime contract cannot be represented safely."""


def effects_for_tool(tool: Tool) -> frozenset[str] | None:
    """Return every possible runtime Effect, or ``None`` when unknown.

    Unknown tools intentionally stay unknown.  They are removed whenever an
    effect-restricted contract is active instead of being guessed as read-only.
    """

    named = _NAMED_TOOL_EFFECTS.get(tool.name)
    if named is not None:
        return named
    permission = tool.permission
    if permission is None:
        return None
    return _PERMISSION_EFFECTS.get(permission.action)


def effects_for_capability(
    capability: str,
    *,
    available_tools: Iterable[Tool] = (),
) -> frozenset[str] | None:
    """Resolve one capability Effect without guessing an unknown tool."""

    named = _NAMED_TOOL_EFFECTS.get(capability)
    if named is not None:
        return named
    tool = next((item for item in available_tools if item.name == capability), None)
    return None if tool is None else effects_for_tool(tool)


def validate_contract_scope(
    *,
    profile_tool_names: Collection[str],
    allowed_tool_names: Collection[str] | None,
    allowed_effects: Collection[str] | None,
    available_tools: Iterable[Tool] = (),
) -> None:
    """Validate the field-level relationship of a formal runtime contract."""

    profile_names = frozenset(profile_tool_names)
    effects = None if allowed_effects is None else frozenset(str(effect).strip().lower() for effect in allowed_effects)
    if effects is not None:
        unknown = effects.difference(KNOWN_TOOL_EFFECTS)
        if unknown:
            raise CapabilityScopeError(f"unknown high-risk Effect: {sorted(unknown)[0]}")
    if allowed_tool_names is None:
        return

    capabilities = frozenset(allowed_tool_names)
    unavailable = capabilities.difference(profile_names)
    if unavailable:
        raise CapabilityScopeError(f"capability is unavailable for profile: {sorted(unavailable)[0]}")
    if effects is None:
        return
    available = tuple(available_tools)
    for capability in sorted(capabilities):
        required_effects = effects_for_capability(
            capability,
            available_tools=available,
        )
        if required_effects is None:
            raise CapabilityScopeError(f"capability has unknown high-risk Effect: {capability}")
        missing = required_effects.difference(effects)
        if missing:
            raise CapabilityScopeError(f"capability {capability!r} requires Effects {sorted(required_effects)!r}; " f"missing {sorted(missing)!r} from allowed_effects")


def scope_tools(
    tools: Iterable[Tool],
    *,
    profile_tool_names: Collection[str],
    allowed_tool_names: Collection[str] | None,
    allowed_effects: Collection[str] | None,
) -> list[Tool]:
    """Intersect profile, contract capability, and Effect restrictions.

    ``None`` preserves the legacy profile default.  An explicitly empty
    collection means no capability/effect is allowed.  Unknown requested
    Effects reject the contract; unknown tool Effects are silently withheld.
    """

    materialized_tools = tuple(tools)
    validate_contract_scope(
        profile_tool_names=profile_tool_names,
        allowed_tool_names=allowed_tool_names,
        allowed_effects=allowed_effects,
        available_tools=materialized_tools,
    )
    profile_names = frozenset(profile_tool_names)
    contract_names = profile_names if allowed_tool_names is None else frozenset(allowed_tool_names)
    permitted_names = profile_names.intersection(contract_names)

    effect_filter: frozenset[str] | None
    if allowed_effects is None:
        effect_filter = None
    else:
        effect_filter = frozenset(str(effect).strip().lower() for effect in allowed_effects)

    selected: list[Tool] = []
    for tool in materialized_tools:
        if tool.name == "delegate" or tool.name not in permitted_names:
            continue
        if effect_filter is not None:
            required_effects = effects_for_tool(tool)
            if required_effects is None or not required_effects.issubset(effect_filter):
                continue
        selected.append(tool)
    return selected
