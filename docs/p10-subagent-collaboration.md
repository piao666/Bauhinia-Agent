# P10 Sub-agent collaboration protocol

P10 adds an evidence-governed collaboration layer without replacing the existing `SubagentRunner`, `BackgroundJobManager`, `PermissionManager`, Tool Registry, Agent Loop, or Session store.

## Roles and runtime mapping

| Domain role | Existing runtime profile | Default boundary |
| --- | --- | --- |
| Planner | researcher | read-only planning evidence |
| Researcher | researcher | read-only exploration |
| Executor | coder | mutation only through existing permission checks; background work uses an isolated worktree |
| Verifier | tester | read plus approved validation execution |
| Critic | reviewer | read-only review |
| Curator | reviewer | read-only evidence curation |

Every assignment uses `planning.TaskContract`. The contract records goal, input snapshot, capabilities, allowed Effects, resource claims, expected evidence, budget, minimum confidence, UTC deadline, and cancellation mode. Runtime capabilities are a subset of the mapped existing profile; an unknown Effect fails closed.

## Append-only facts

The v1 Evo envelope recognizes four additional event types:

- `CollaborationTaskDelegated`: redacted Task Contract and runtime role.
- `CollaborationTaskResultRecorded`: status, child Run/Session, evidence references, confidence, claim fingerprints, changed files, and learning eligibility.
- `CollaborationConflictDetected`: write-resource or conclusion conflict with retained branches.
- `CollaborationRunAggregated`: parent-visible child Run references, result/conflict events, independent evidence-group count, and eligible result IDs.

Example delegation payload:

```json
{
  "collaboration_id": "collab_example",
  "assignment_id": "assignment_review",
  "runtime_role": "reviewer",
  "contract": {
    "role": "critic",
    "plan_id": "plan_example",
    "node_id": "node_review",
    "goal": "Review the API change",
    "input_snapshot": "tree@abc123",
    "allowed_effects": ["read"],
    "expected_evidence": ["diff and focused test result"],
    "budget": {"max_tool_calls": 5, "max_attempts": 1, "max_tokens": 2000},
    "capabilities": ["git_diff", "view"],
    "resource_claims": ["read:src/api.py"],
    "minimum_confidence": 0.7,
    "cancellation_mode": "cooperative",
    "deadline_at": "2026-08-09T12:00:00Z"
  }
}
```

Unknown future fields remain available through payload extensions. Adding these event types is backward compatible and does not change the v1 envelope or require a destructive migration.

## Evidence and conflict rules

- A result is a learning input only when it succeeded, has evidence references, meets its contract confidence threshold, matches the assigned role, and was persisted successfully.
- Claims with the same provenance key or any overlapping evidence reference form one independent evidence group, even when several agents repeat them.
- Different conclusions for one claim key are retained as branches and remain pending for a Verifier or Curator.
- Overlapping `write:` resource claims, or a `write:` claim overlapping a `read:` claim, are not dispatched concurrently.
- Experience Compiler may link the parent aggregate event for lineage, but child Run IDs are not copied into a single-run Candidate as additional independent support.
- Recorder failures produce diagnostics and never convert a failed/cancelled child execution into a successful learning result.
