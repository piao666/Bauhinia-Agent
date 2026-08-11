"""Versioned P2 planning contracts backed by append-only Evo events.

The existing :mod:`bauhinia_agent.planning` TaskPlan remains the session-level
task control plane.  This module adds the evidence-oriented graph and decision
contracts used by the Evo slow loop; it never executes a tool or makes a
permission decision itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Callable, Literal, Mapping, Protocol, cast

from bauhinia_agent.evolution.events import (
    DecisionRecordedPayload,
    EvoEvent,
    EvoReferences,
    PlanCreatedPayload,
    PlanNodeUpdatedPayload,
)
from bauhinia_agent.evolution.identifiers import new_evo_id, require_evo_id
from bauhinia_agent.evolution.store import EvoEventStore

PlanNodeStatus = Literal["pending", "in_progress", "completed", "failed", "blocked", "cancelled", "superseded"]
ReplanTrigger = Literal[
    "verification_failed",
    "tool_error",
    "permission_denied",
    "precondition_invalid",
    "context_conflict",
    "budget_exhausted",
    "user_changed_goal",
    "cancelled",
]
ReplanOutcome = Literal["continue", "narrow", "replace", "request_help", "terminate"]
PlanningRole = Literal["planner", "researcher", "executor", "verifier", "critic", "curator"]
CancellationMode = Literal["cooperative", "terminate"]

_PLANNING_ROLES = frozenset({"planner", "researcher", "executor", "verifier", "critic", "curator"})
_CANCELLATION_MODES = frozenset({"cooperative", "terminate"})

_NODE_STATUSES = frozenset({"pending", "in_progress", "completed", "failed", "blocked", "cancelled", "superseded"})
_REPLAN_TRIGGERS = frozenset(
    {
        "verification_failed",
        "tool_error",
        "permission_denied",
        "precondition_invalid",
        "context_conflict",
        "budget_exhausted",
        "user_changed_goal",
        "cancelled",
    }
)
_REPLAN_OUTCOMES = frozenset({"continue", "narrow", "replace", "request_help", "terminate"})
_TRANSITIONS: dict[PlanNodeStatus, frozenset[PlanNodeStatus]] = {
    "pending": frozenset({"in_progress", "completed", "failed", "blocked", "cancelled", "superseded"}),
    "in_progress": frozenset({"completed", "failed", "blocked", "cancelled", "superseded"}),
    "completed": frozenset(),
    "failed": frozenset({"pending", "cancelled", "superseded"}),
    "blocked": frozenset({"pending", "cancelled", "superseded"}),
    "cancelled": frozenset({"pending"}),
    "superseded": frozenset(),
}


class PlanGraphError(ValueError):
    """A graph, decision, or planning hand-off violates the P2 contract."""


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanGraphError(f"{field} must be a non-blank string")
    return value


def _identifier(value: object, *, field: str, kind: str | None = None) -> str:
    try:
        return require_evo_id(value, field=field, kind=cast(object, kind))  # type: ignore[arg-type]
    except ValueError as error:
        raise PlanGraphError(str(error)) from error


def _string_tuple(value: object, *, field: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise PlanGraphError(f"{field} must be a list of non-blank strings")
    result = tuple(_text(item, field=f"{field}[]") for item in value)
    if len(result) != len(set(result)):
        raise PlanGraphError(f"{field} must not contain duplicates")
    return result


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanGraphError(f"{field} must be a non-negative integer")
    return value


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise PlanGraphError("confidence must be between 0 and 1")
    return float(value)


def _utc_deadline(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise PlanGraphError(f"{field} must be a UTC ISO-8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise PlanGraphError(f"{field} must include UTC timezone")
    return text


@dataclass(frozen=True, slots=True)
class PlanBudget:
    """Per-node upper bounds; ``None`` means the existing loop owns that limit."""

    max_tool_calls: int | None = None
    max_attempts: int = 0
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        for field in ("max_tool_calls", "max_tokens"):
            value = getattr(self, field)
            if value is not None:
                _non_negative_int(value, field=field)
        _non_negative_int(self.max_attempts, field="max_attempts")

    def to_dict(self) -> dict[str, int | None]:
        return {"max_tool_calls": self.max_tool_calls, "max_attempts": self.max_attempts, "max_tokens": self.max_tokens}

    @classmethod
    def from_dict(cls, raw: object) -> "PlanBudget":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise PlanGraphError("budget must be an object")
        unknown = set(raw).difference({"max_tool_calls", "max_attempts", "max_tokens"})
        if unknown:
            raise PlanGraphError(f"budget has unknown field: {sorted(unknown)[0]}")
        values: dict[str, int | None] = {}
        for field in ("max_tool_calls", "max_tokens"):
            value = raw.get(field)
            values[field] = None if value is None else _non_negative_int(value, field=field)
        return cls(max_attempts=_non_negative_int(raw.get("max_attempts", 0), field="max_attempts"), **values)


@dataclass(frozen=True, slots=True)
class PlanNode:
    """A graph node with executable preconditions and observable acceptance criteria."""

    node_id: str
    goal: str
    depends_on: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    budget: PlanBudget = PlanBudget()
    verification_conditions: tuple[str, ...] = ()
    status: PlanNodeStatus = "pending"
    attempts: int = 0
    evidence_refs: tuple[str, ...] = ()
    parent_node_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.node_id, field="node_id", kind="node")
        _text(self.goal, field="goal")
        _string_tuple(self.depends_on, field="depends_on")
        _string_tuple(self.preconditions, field="preconditions")
        _string_tuple(self.risks, field="risks")
        _string_tuple(self.verification_conditions, field="verification_conditions")
        _string_tuple(self.evidence_refs, field="evidence_refs")
        if self.status not in _NODE_STATUSES:
            raise PlanGraphError(f"unknown node status: {self.status!r}")
        _non_negative_int(self.attempts, field="attempts")
        if self.attempts > self.budget.max_attempts:
            raise PlanGraphError("attempts exceeds budget.max_attempts")
        if self.parent_node_id is not None:
            _identifier(self.parent_node_id, field="parent_node_id", kind="node")

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "goal": self.goal,
            "depends_on": list(self.depends_on),
            "preconditions": list(self.preconditions),
            "risks": list(self.risks),
            "budget": self.budget.to_dict(),
            "verification_conditions": list(self.verification_conditions),
            "status": self.status,
            "attempts": self.attempts,
            "evidence_refs": list(self.evidence_refs),
            "parent_node_id": self.parent_node_id,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "PlanNode":
        if not isinstance(raw, Mapping):
            raise PlanGraphError("node must be an object")
        known = {
            "node_id",
            "goal",
            "depends_on",
            "preconditions",
            "risks",
            "budget",
            "verification_conditions",
            "status",
            "attempts",
            "evidence_refs",
            "parent_node_id",
        }
        unknown = set(raw).difference(known)
        if unknown:
            raise PlanGraphError(f"node has unknown field: {sorted(unknown)[0]}")
        status = raw.get("status", "pending")
        if not isinstance(status, str):
            raise PlanGraphError("status must be a string")
        parent = raw.get("parent_node_id")
        if parent is not None and not isinstance(parent, str):
            raise PlanGraphError("parent_node_id must be a string or null")
        return cls(
            node_id=_identifier(raw.get("node_id"), field="node_id", kind="node"),
            goal=_text(raw.get("goal"), field="goal"),
            depends_on=_string_tuple(raw.get("depends_on"), field="depends_on"),
            preconditions=_string_tuple(raw.get("preconditions"), field="preconditions"),
            risks=_string_tuple(raw.get("risks"), field="risks"),
            budget=PlanBudget.from_dict(raw.get("budget")),
            verification_conditions=_string_tuple(raw.get("verification_conditions"), field="verification_conditions"),
            status=cast(PlanNodeStatus, status),
            attempts=_non_negative_int(raw.get("attempts", 0), field="attempts"),
            evidence_refs=_string_tuple(raw.get("evidence_refs"), field="evidence_refs"),
            parent_node_id=parent,
        )


@dataclass(frozen=True, slots=True)
class PlanGraph:
    """An immutable, versioned DAG. Replans always produce a new version."""

    plan_id: str
    goal: str
    version: int
    nodes: tuple[PlanNode, ...]

    def __post_init__(self) -> None:
        _identifier(self.plan_id, field="plan_id", kind="plan")
        _text(self.goal, field="goal")
        _non_negative_int(self.version, field="version")
        self.validate()

    def validate(self) -> None:
        by_id = {node.node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise PlanGraphError("plan nodes must have unique node_id values")
        for node in self.nodes:
            if node.node_id in node.depends_on:
                raise PlanGraphError(f"node {node.node_id} cannot depend on itself")
            for dependency in node.depends_on:
                if dependency not in by_id:
                    raise PlanGraphError(f"node {node.node_id} depends on missing node {dependency}")
            if node.parent_node_id is not None and node.parent_node_id not in by_id:
                raise PlanGraphError(f"node {node.node_id} has missing parent {node.parent_node_id}")
            if node.status == "in_progress" and not all(by_id[item].status == "completed" for item in node.depends_on):
                raise PlanGraphError(f"node {node.node_id} is not ready to enter in_progress")
        remaining = {node.node_id: set(node.depends_on) for node in self.nodes}
        while remaining:
            ready = {node_id for node_id, dependencies in remaining.items() if not dependencies}
            if not ready:
                raise PlanGraphError("plan graph contains a dependency cycle")
            remaining = {node_id: dependencies.difference(ready) for node_id, dependencies in remaining.items() if node_id not in ready}

    def node(self, node_id: str) -> PlanNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise PlanGraphError(f"unknown plan node: {node_id}")

    def replace_nodes(self, nodes: tuple[PlanNode, ...], *, expected_version: int) -> "PlanGraph":
        if expected_version != self.version:
            raise PlanGraphError(f"plan version conflict: expected {expected_version}, actual {self.version}")
        next_graph = PlanGraph(plan_id=self.plan_id, goal=self.goal, version=self.version + 1, nodes=nodes)
        previous = {node.node_id: node for node in self.nodes}
        for node in next_graph.nodes:
            before = previous.get(node.node_id)
            if before is not None and before.status != node.status and node.status not in _TRANSITIONS[before.status]:
                raise PlanGraphError(f"node {node.node_id} cannot transition from {before.status} to {node.status}")
        return next_graph

    def to_dict(self) -> dict[str, object]:
        return {"plan_id": self.plan_id, "goal": self.goal, "version": self.version, "nodes": [node.to_dict() for node in self.nodes]}

    @classmethod
    def from_dict(cls, raw: object) -> "PlanGraph":
        if not isinstance(raw, Mapping):
            raise PlanGraphError("plan graph must be an object")
        unknown = set(raw).difference({"plan_id", "goal", "version", "nodes"})
        if unknown:
            raise PlanGraphError(f"plan graph has unknown field: {sorted(unknown)[0]}")
        nodes = raw.get("nodes")
        if not isinstance(nodes, list):
            raise PlanGraphError("nodes must be a list")
        return cls(
            plan_id=_identifier(raw.get("plan_id"), field="plan_id", kind="plan"),
            goal=_text(raw.get("goal"), field="goal"),
            version=_non_negative_int(raw.get("version"), field="version"),
            nodes=tuple(PlanNode.from_dict(item) for item in nodes),
        )


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Auditable decision summary. It intentionally has no private-reasoning field."""

    decision_id: str
    plan_id: str
    node_id: str
    subgoal: str
    evidence_refs: tuple[str, ...]
    assumptions: tuple[str, ...]
    options_considered: tuple[str, ...]
    selected_action: str
    rationale_summary: str
    confidence: float
    expected_observation: str
    verification_method: str
    outcome: str | None = None
    next_decision: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.decision_id, field="decision_id")
        _identifier(self.plan_id, field="plan_id", kind="plan")
        _identifier(self.node_id, field="node_id", kind="node")
        for field in ("subgoal", "selected_action", "rationale_summary", "expected_observation", "verification_method"):
            _text(getattr(self, field), field=field)
        if len(self.rationale_summary) > 2000:
            raise PlanGraphError("rationale_summary must be at most 2000 characters")
        _string_tuple(self.evidence_refs, field="evidence_refs")
        _string_tuple(self.assumptions, field="assumptions")
        _string_tuple(self.options_considered, field="options_considered")
        _confidence(self.confidence)
        for field in ("outcome", "next_decision"):
            value = getattr(self, field)
            if value is not None:
                _text(value, field=field)

    def to_payload(self) -> DecisionRecordedPayload:
        return DecisionRecordedPayload(
            subgoal=self.subgoal,
            evidence_refs=self.evidence_refs,
            assumptions=self.assumptions,
            options_considered=self.options_considered,
            selected_action=self.selected_action,
            rationale_summary=self.rationale_summary,
            confidence=self.confidence,
            expected_observation=self.expected_observation,
            verification_method=self.verification_method,
            outcome=self.outcome,
            next_decision=self.next_decision,
            extensions={"decision_id": self.decision_id},
        )


@dataclass(frozen=True, slots=True)
class ReplanRequest:
    plan_id: str
    expected_version: int
    trigger: ReplanTrigger
    outcome: ReplanOutcome
    evidence_refs: tuple[str, ...]
    rationale_summary: str
    next_graph: PlanGraph

    def __post_init__(self) -> None:
        _identifier(self.plan_id, field="plan_id", kind="plan")
        _non_negative_int(self.expected_version, field="expected_version")
        if self.trigger not in _REPLAN_TRIGGERS:
            raise PlanGraphError(f"unknown replan trigger: {self.trigger!r}")
        if self.outcome not in _REPLAN_OUTCOMES:
            raise PlanGraphError(f"unknown replan outcome: {self.outcome!r}")
        _string_tuple(self.evidence_refs, field="evidence_refs")
        _text(self.rationale_summary, field="rationale_summary")
        if self.next_graph.plan_id != self.plan_id or self.next_graph.version != self.expected_version + 1:
            raise PlanGraphError("replan next_graph must retain plan_id and increment version by one")


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Role hand-off without importing the Agent loop or a concrete tool registry."""

    role: PlanningRole
    plan_id: str
    node_id: str
    goal: str
    input_snapshot: str
    allowed_effects: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    budget: PlanBudget
    capabilities: tuple[str, ...] = ()
    resource_claims: tuple[str, ...] = ()
    minimum_confidence: float = 0.5
    cancellation_mode: CancellationMode = "cooperative"
    deadline_at: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _PLANNING_ROLES:
            raise PlanGraphError(f"unknown planning role: {self.role!r}")
        _identifier(self.plan_id, field="plan_id", kind="plan")
        _identifier(self.node_id, field="node_id", kind="node")
        _text(self.goal, field="goal")
        _text(self.input_snapshot, field="input_snapshot")
        _string_tuple(self.allowed_effects, field="allowed_effects")
        _string_tuple(self.expected_evidence, field="expected_evidence")
        _string_tuple(self.capabilities, field="capabilities")
        claims = _string_tuple(self.resource_claims, field="resource_claims")
        for claim in claims:
            if not claim.startswith(("read:", "write:")) or not claim.partition(":")[2].strip():
                raise PlanGraphError("resource_claims entries must use read:<resource> or write:<resource>")
        _confidence(self.minimum_confidence)
        if self.cancellation_mode not in _CANCELLATION_MODES:
            raise PlanGraphError(f"unknown cancellation mode: {self.cancellation_mode!r}")
        if self.deadline_at is not None:
            _utc_deadline(self.deadline_at, field="deadline_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "plan_id": self.plan_id,
            "node_id": self.node_id,
            "goal": self.goal,
            "input_snapshot": self.input_snapshot,
            "allowed_effects": list(self.allowed_effects),
            "expected_evidence": list(self.expected_evidence),
            "budget": self.budget.to_dict(),
            "capabilities": list(self.capabilities),
            "resource_claims": list(self.resource_claims),
            "minimum_confidence": self.minimum_confidence,
            "cancellation_mode": self.cancellation_mode,
            "deadline_at": self.deadline_at,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "TaskContract":
        if not isinstance(raw, Mapping):
            raise PlanGraphError("task contract must be an object")
        known = {
            "role",
            "plan_id",
            "node_id",
            "goal",
            "input_snapshot",
            "allowed_effects",
            "expected_evidence",
            "budget",
            "capabilities",
            "resource_claims",
            "minimum_confidence",
            "cancellation_mode",
            "deadline_at",
        }
        unknown = set(raw).difference(known)
        if unknown:
            raise PlanGraphError(f"task contract has unknown field: {sorted(unknown)[0]}")
        role = raw.get("role")
        cancellation_mode = raw.get("cancellation_mode", "cooperative")
        if not isinstance(role, str):
            raise PlanGraphError("role must be a string")
        if not isinstance(cancellation_mode, str):
            raise PlanGraphError("cancellation_mode must be a string")
        deadline = raw.get("deadline_at")
        if deadline is not None and not isinstance(deadline, str):
            raise PlanGraphError("deadline_at must be a string or null")
        return cls(
            role=cast(PlanningRole, role),
            plan_id=_identifier(raw.get("plan_id"), field="plan_id", kind="plan"),
            node_id=_identifier(raw.get("node_id"), field="node_id", kind="node"),
            goal=_text(raw.get("goal"), field="goal"),
            input_snapshot=_text(raw.get("input_snapshot"), field="input_snapshot"),
            allowed_effects=_string_tuple(raw.get("allowed_effects"), field="allowed_effects"),
            expected_evidence=_string_tuple(raw.get("expected_evidence"), field="expected_evidence"),
            budget=PlanBudget.from_dict(raw.get("budget")),
            capabilities=_string_tuple(raw.get("capabilities"), field="capabilities"),
            resource_claims=_string_tuple(raw.get("resource_claims"), field="resource_claims"),
            minimum_confidence=_confidence(raw.get("minimum_confidence", 0.5)),
            cancellation_mode=cast(CancellationMode, cancellation_mode),
            deadline_at=deadline,
        )


class PermissionPreflight(Protocol):
    def __call__(self, contract: TaskContract) -> bool: ...


class ExecuteAction(Protocol):
    def __call__(self, contract: TaskContract) -> object: ...


class VerifyAction(Protocol):
    def __call__(self, contract: TaskContract, execution_result: object) -> tuple[bool, tuple[str, ...]]: ...


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    contract: TaskContract
    result: object
    permission_checked: bool


class PlanningExecutionGate:
    """Application-level hand-off gate; permission and verification stay external.

    The gate deliberately accepts callbacks so it can be wired to the existing
    PermissionManager and ToolExecutor instead of becoming a second executor.
    """

    def __init__(self, *, permission_preflight: PermissionPreflight, execute: ExecuteAction, verify: VerifyAction) -> None:
        self._permission_preflight = permission_preflight
        self._execute = execute
        self._verify = verify

    def execute(self, contract: TaskContract) -> ExecutionReceipt:
        if contract.role != "executor":
            raise PlanGraphError("only an executor contract may execute an action")
        if not self._permission_preflight(contract):
            raise PlanGraphError("permission preflight denied execution")
        return ExecutionReceipt(contract=contract, result=self._execute(contract), permission_checked=True)

    def verify(self, receipt: ExecutionReceipt, verifier: TaskContract) -> tuple[bool, tuple[str, ...]]:
        if verifier.role != "verifier":
            raise PlanGraphError("only a verifier contract may accept execution evidence")
        if verifier.plan_id != receipt.contract.plan_id or verifier.node_id != receipt.contract.node_id:
            raise PlanGraphError("verifier contract must target the executed plan node")
        return self._verify(verifier, receipt.result)


class EvoPlanningService:
    """Persist PlanGraph, DecisionRecord, and Replan facts to P1's event store."""

    def __init__(self, *, store: EvoEventStore, run_id: str, session_id: str | None = None) -> None:
        self._store = store
        self._run_id = _identifier(run_id, field="run_id", kind="run")
        self._session_id = None if session_id is None else _identifier(session_id, field="session_id", kind="session")

    def create(self, graph: PlanGraph) -> PlanGraph:
        if self.get(graph.plan_id) is not None:
            raise PlanGraphError(f"plan already exists: {graph.plan_id}")
        self._append(
            event_type="PlanCreated",
            refs=EvoReferences(run_id=self._run_id, session_id=self._session_id, plan_id=graph.plan_id),
            payload=PlanCreatedPayload(goal=graph.goal, node_ids=tuple(node.node_id for node in graph.nodes), extensions={"graph": graph.to_dict()}),
        )
        return graph

    def record_decision(self, decision: DecisionRecord) -> str:
        graph = self.get(decision.plan_id)
        if graph is None:
            raise PlanGraphError(f"cannot record a decision for unknown plan: {decision.plan_id}")
        graph.node(decision.node_id)
        result = self._append(
            event_type="DecisionRecorded",
            refs=EvoReferences(run_id=self._run_id, session_id=self._session_id, plan_id=decision.plan_id, node_id=decision.node_id),
            payload=decision.to_payload(),
        )
        return result.event.event_id

    def update_node(self, *, plan_id: str, expected_version: int, node: PlanNode, change_summary: str) -> PlanGraph:
        graph = self._require(plan_id)
        if graph.version != expected_version:
            raise PlanGraphError(f"plan version conflict: expected {expected_version}, actual {graph.version}")
        existing = graph.node(node.node_id)
        if existing == node:
            return graph
        nodes = tuple(node if item.node_id == node.node_id else item for item in graph.nodes)
        next_graph = graph.replace_nodes(nodes, expected_version=expected_version)
        self._append_node_update(graph=next_graph, node=node, change_summary=change_summary)
        return next_graph

    def synchronize(self, graph: PlanGraph, *, expected_version: int, change_summary: str) -> PlanGraph:
        """Persist externally scheduled graph changes without inventing a Replan.

        The existing TaskPlan service uses this boundary after it has already
        accepted a session-scoped task mutation.  Failure here must therefore be
        reported to the caller as a diagnostic rather than altering that task
        mutation's result.
        """

        current = self._require(graph.plan_id)
        if current.version != expected_version:
            raise PlanGraphError(f"plan version conflict: expected {expected_version}, actual {current.version}")
        current.replace_nodes(graph.nodes, expected_version=expected_version)
        changed = [node for node in graph.nodes if _node_changed(current, node)]
        if not changed:
            return current
        for node in changed:
            self._append_node_update(graph=graph, node=node, change_summary=change_summary)
        return graph

    def replan(self, request: ReplanRequest) -> PlanGraph:
        current = self._require(request.plan_id)
        if current.version != request.expected_version:
            raise PlanGraphError(f"plan version conflict: expected {request.expected_version}, actual {current.version}")
        current.replace_nodes(request.next_graph.nodes, expected_version=request.expected_version)
        changed = [node for node in request.next_graph.nodes if _node_changed(current, node)]
        if not changed:
            raise PlanGraphError("replan must change at least one node")
        decision = DecisionRecord(
            decision_id=new_evo_id("event"),
            plan_id=request.plan_id,
            node_id=changed[0].node_id,
            subgoal=current.goal,
            evidence_refs=request.evidence_refs,
            assumptions=(),
            options_considered=("continue", "narrow", "replace", "request_help", "terminate"),
            selected_action=request.outcome,
            rationale_summary=request.rationale_summary,
            confidence=1.0,
            expected_observation=f"plan graph advances to version {request.next_graph.version}",
            verification_method="inspect append-only Evo events",
            next_decision="resume only from the replacement plan version",
        )
        parent_event_id = self.record_decision(decision)
        for node in changed:
            self._append_node_update(
                graph=request.next_graph,
                node=node,
                change_summary=f"replan:{request.trigger}:{request.outcome}",
                parent_event_id=parent_event_id,
                replan={"trigger": request.trigger, "outcome": request.outcome, "evidence_refs": list(request.evidence_refs)},
            )
        return request.next_graph

    def get(self, plan_id: str) -> PlanGraph | None:
        _identifier(plan_id, field="plan_id", kind="plan")
        graph: PlanGraph | None = None
        for event in self._store.list_events():
            if event.refs.plan_id != plan_id:
                continue
            if event.event_type == "PlanCreated" and isinstance(event.payload, PlanCreatedPayload):
                raw = event.payload.extensions.get("graph")
                graph = PlanGraph.from_dict(raw)
            elif event.event_type == "PlanNodeUpdated" and isinstance(event.payload, PlanNodeUpdatedPayload) and graph is not None:
                raw_node = event.payload.extensions.get("node")
                version = event.payload.extensions.get("graph_version")
                if raw_node is None or not isinstance(version, int):
                    continue
                node = PlanNode.from_dict(raw_node)
                by_id = {item.node_id: item for item in graph.nodes}
                by_id[node.node_id] = node
                graph = PlanGraph(
                    plan_id=graph.plan_id,
                    goal=graph.goal,
                    version=version,
                    nodes=tuple(by_id[item.node_id] if item.node_id in by_id else item for item in graph.nodes)
                    + tuple(item for item in by_id.values() if item.node_id not in {old.node_id for old in graph.nodes}),
                )
        return graph

    def _require(self, plan_id: str) -> PlanGraph:
        graph = self.get(plan_id)
        if graph is None:
            raise PlanGraphError(f"unknown plan: {plan_id}")
        return graph

    def _append_node_update(
        self,
        *,
        graph: PlanGraph,
        node: PlanNode,
        change_summary: str,
        parent_event_id: str | None = None,
        replan: dict[str, object] | None = None,
    ) -> None:
        extensions: dict[str, object] = {"graph_version": graph.version, "node": node.to_dict()}
        if replan is not None:
            extensions["replan"] = replan
        self._append(
            event_type="PlanNodeUpdated",
            refs=EvoReferences(
                run_id=self._run_id,
                session_id=self._session_id,
                plan_id=graph.plan_id,
                node_id=node.node_id,
                parent_event_id=parent_event_id,
            ),
            payload=PlanNodeUpdatedPayload(
                status=node.status,
                change_summary=_text(change_summary, field="change_summary"),
                attempt=node.attempts,
                verification_refs=node.evidence_refs,
                extensions=extensions,
            ),
        )

    def _append(self, *, event_type: str, refs: EvoReferences, payload: object):
        return self._store.append(EvoEvent(event_id=new_evo_id("event"), event_type=event_type, refs=refs, payload=cast(object, payload)))


def _node_changed(previous: PlanGraph, next_node: PlanNode) -> bool:
    try:
        return previous.node(next_node.node_id) != next_node
    except PlanGraphError:
        return True
