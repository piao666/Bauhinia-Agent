"""Deterministic Memory retrieval, strict Context Packs, and usage feedback."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from bauhinia_agent.context.token_budget import estimate_text_tokens
from bauhinia_agent.evolution import (
    ContextPackRecordedPayload,
    EvoEvent,
    EvoReferences,
    EvoStoreError,
    MemoryUsedPayload,
    OutcomeClassifiedPayload,
    OutcomeIntegrityError,
    attest_outcome_event,
    new_evo_id,
    redact_text,
    require_evo_id,
    resolve_evidence_records,
)
from bauhinia_agent.memory.models import MemoryRecord
from bauhinia_agent.memory.service import MemoryService

_WORDS = re.compile(r"[\w-]+", re.UNICODE)


class TokenEstimator(Protocol):
    estimator_id: str

    def estimate(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class HeuristicTokenEstimator:
    """Provider-neutral estimator shared with the existing Context layer."""

    estimator_id: str = "heuristic-char-v1"

    def estimate(self, text: str) -> int:
        return estimate_text_tokens(text)


@dataclass(frozen=True, slots=True)
class MemoryAccessAuthorization:
    """Explicit application-boundary authorization for restricted Memory."""

    project_id: str
    user_id: str | None = None
    session_id: str | None = None
    allow_restricted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, str) or not self.project_id.strip():
            raise ValueError("authorization project_id must be non-blank")
        for field_name in ("user_id", "session_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"authorization {field_name} must be non-blank or null")


class _EstimatorAdapter:
    """Normalize legacy callables and validate estimator boundary results."""

    def __init__(self, estimator: TokenEstimator | Callable[[str], int]) -> None:
        estimate = getattr(estimator, "estimate", None)
        if callable(estimate):
            self._estimate = estimate
            estimator_id = getattr(estimator, "estimator_id", None)
        elif callable(estimator):
            self._estimate = estimator
            module = getattr(estimator, "__module__", "unknown")
            name = getattr(estimator, "__qualname__", getattr(estimator, "__name__", "callable"))
            estimator_id = f"callable:{module}.{name}:v1"
        else:
            raise TypeError("estimator must implement estimate(text) or be callable")
        if not isinstance(estimator_id, str) or not estimator_id.strip():
            raise ValueError("TokenEstimator.estimator_id must be a non-blank string")
        self.estimator_id = estimator_id.strip()

    def estimate(self, text: str) -> int:
        value = self._estimate(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("TokenEstimator.estimate must return a non-negative integer")
        return value


@dataclass(frozen=True, slots=True)
class QuerySignature:
    goal: str
    repository_features: tuple[str, ...] = ()
    error_type: str | None = None
    tool_environment: tuple[str, ...] = ()
    plan_node_id: str | None = None
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must be a non-blank string")
        for field_name in (
            "repository_features",
            "tool_environment",
            "constraints",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} must be a tuple of non-blank strings")
        if self.error_type is not None and (not isinstance(self.error_type, str) or not self.error_type.strip()):
            raise ValueError("error_type must be a non-blank string or null")
        if self.plan_node_id is not None:
            require_evo_id(self.plan_node_id, field="plan_node_id", kind="node")

    @property
    def terms(self) -> frozenset[str]:
        values = (
            self.goal,
            *self.repository_features,
            *self.tool_environment,
            *self.constraints,
            self.error_type or "",
        )
        return frozenset(_WORDS.findall(" ".join(values).casefold()))

    @property
    def signature_hash(self) -> str:
        canonical = json.dumps(
            {
                "goal": self.goal,
                "repository_features": self.repository_features,
                "error_type": self.error_type,
                "tool_environment": self.tool_environment,
                "plan_node_id": self.plan_node_id,
                "constraints": self.constraints,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    record: MemoryRecord
    score: int
    reasons: tuple[str, ...]
    token_cost: int
    matched_terms: tuple[str, ...] = ()
    exact_error_match: bool = False
    confidence_score: float = 0.0
    freshness_score: float = 0.0
    scope_specificity: int = 0
    rank: int = 0


@dataclass(frozen=True, slots=True)
class ContextPackItem:
    hit: RetrievalHit
    packed_content: str
    original_token_cost: int
    packed_token_cost: int
    truncated: bool
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class ContextOmission:
    memory_id: str
    reason: str
    details: str | None = None


@dataclass(frozen=True, slots=True)
class ContextPackDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ContextPack:
    context_pack_id: str
    query_signature_hash: str
    items: tuple[ContextPackItem, ...]
    omissions: tuple[ContextOmission, ...]
    token_budget: int
    used_tokens: int
    estimator_id: str
    run_id: str | None = None
    plan_id: str | None = None
    node_id: str | None = None
    recorded_event_id: str | None = None
    diagnostic: ContextPackDiagnostic | None = None

    @property
    def hits(self) -> tuple[RetrievalHit, ...]:
        """Compatibility view used by P4 callers before packed item metadata."""

        return tuple(item.hit for item in self.items)

    @property
    def omitted(self) -> tuple[tuple[str, str], ...]:
        """Compatibility view of omission ID and reason pairs."""

        return tuple((item.memory_id, item.reason) for item in self.omissions)

    @property
    def rendered(self) -> str:
        return _render_pack(self.items)


@dataclass(frozen=True, slots=True)
class MemoryUseDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class MemoryUseResult:
    persisted: bool
    event: EvoEvent[MemoryUsedPayload] | None = None
    diagnostic: MemoryUseDiagnostic | None = None


class MemoryRetriever:
    def __init__(
        self,
        service: MemoryService,
        *,
        clock: Callable[[], datetime] | None = None,
        estimator: TokenEstimator | Callable[[str], int] | None = None,
    ) -> None:
        self._service = service
        self._clock = clock or (lambda: datetime.now(UTC))
        self._estimator = _EstimatorAdapter(estimator or HeuristicTokenEstimator())

    def retrieve(
        self,
        signature: QuerySignature,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        at: datetime | None = None,
        include_restricted: bool = False,
        authorization: MemoryAccessAuthorization | None = None,
    ) -> list[RetrievalHit]:
        self._authorize_restricted(
            include_restricted=include_restricted,
            authorization=authorization,
            user_id=user_id,
            session_id=session_id,
        )
        hits, _ = self._candidates(
            signature,
            user_id=user_id,
            session_id=session_id,
            at=at,
            include_restricted=include_restricted,
        )
        return list(hits)

    def pack(
        self,
        signature: QuerySignature,
        *,
        token_budget: int,
        user_id: str | None = None,
        session_id: str | None = None,
        at: datetime | None = None,
        include_restricted: bool = False,
        authorization: MemoryAccessAuthorization | None = None,
        run_id: str | None = None,
        plan_id: str | None = None,
        node_id: str | None = None,
    ) -> ContextPack:
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 0:
            raise ValueError("token_budget must be a non-negative integer")
        if run_id is None and any(value is not None for value in (plan_id, node_id)):
            raise ValueError("plan_id and node_id require run_id")
        if run_id is not None:
            require_evo_id(run_id, field="run_id", kind="run")
        if plan_id is not None:
            require_evo_id(plan_id, field="plan_id", kind="plan")
        if node_id is not None:
            require_evo_id(node_id, field="node_id", kind="node")
        if signature.plan_node_id is not None and node_id != signature.plan_node_id:
            raise ValueError("node_id must match QuerySignature.plan_node_id")
        self._authorize_restricted(
            include_restricted=include_restricted,
            authorization=authorization,
            user_id=user_id,
            session_id=session_id,
        )

        hits, candidate_omissions = self._candidates(
            signature,
            user_id=user_id,
            session_id=session_id,
            at=at,
            include_restricted=include_restricted,
        )
        items: list[ContextPackItem] = []
        omissions = list(candidate_omissions)
        for hit in hits:
            full = _pack_item(hit, hit.record.content, self._estimator)
            if self._estimate_items((*items, full)) <= token_budget:
                items.append(full)
                continue
            truncated = self._largest_fitting_item(
                hit,
                existing=tuple(items),
                token_budget=token_budget,
            )
            if truncated is None:
                omissions.append(
                    ContextOmission(
                        hit.record.memory_id,
                        "token_budget",
                        "Even the item metadata and matched text exceeded the budget.",
                    )
                )
            else:
                items.append(truncated)

        used_tokens = self._estimate_items(tuple(items))
        if used_tokens > token_budget:
            raise RuntimeError("Context Pack estimator invariant was violated")
        pack = ContextPack(
            context_pack_id=new_evo_id("context_pack"),
            query_signature_hash=signature.signature_hash,
            items=tuple(items),
            omissions=tuple(omissions),
            token_budget=token_budget,
            used_tokens=used_tokens,
            estimator_id=self._estimator.estimator_id,
            run_id=run_id,
            plan_id=plan_id,
            node_id=node_id,
        )
        return pack if run_id is None else self._record_pack(pack)

    def record_use(
        self,
        pack: ContextPack,
        memory_id: str,
        *,
        run_id: str,
        reason: str,
        usage_status: str = "used",
        outcome_event_id: str | None = None,
        verification_evidence_refs: tuple[str, ...] = (),
        feedback_status: str = "unknown",
    ) -> MemoryUseResult:
        require_evo_id(memory_id, field="memory_id", kind="memory")
        require_evo_id(run_id, field="run_id", kind="run")
        if pack.recorded_event_id is None or pack.run_id is None:
            raise ValueError("Memory use requires a recorded Context Pack")
        if pack.run_id != run_id:
            raise ValueError("Memory use run_id must match the Context Pack Run")
        if usage_status not in {"used", "not_used"}:
            raise ValueError("usage_status must be used or not_used")
        if feedback_status not in {"helpful", "harmful", "neutral", "unknown"}:
            raise ValueError("unsupported feedback_status")
        if usage_status != "used" and feedback_status != "unknown":
            raise ValueError("only used Memory can receive outcome feedback")
        reason_text = _text(reason, field="reason")

        events = self._service.store.list_events()
        context_event = next(
            (event for event in events if event.event_id == pack.recorded_event_id and event.event_type == "ContextPackRecorded" and isinstance(event.payload, ContextPackRecordedPayload)),
            None,
        )
        if context_event is None or context_event.refs.run_id != run_id or context_event.refs.context_pack_id != pack.context_pack_id:
            raise ValueError("recorded Context Pack cannot be resolved in this Run")
        context_payload = context_event.payload
        try:
            selected_index = context_payload.selected_memory_ids.index(memory_id)
        except ValueError:
            selected_index = None
        try:
            omitted_index = context_payload.omitted_memory_ids.index(memory_id)
        except ValueError:
            omitted_index = None
        if usage_status == "used" and selected_index is None:
            raise ValueError("used Memory must be present in the recorded Context Pack")
        if usage_status == "not_used" and omitted_index is None:
            raise ValueError("not_used Memory must be present in recorded Context Pack omissions")

        if outcome_event_id is None:
            if verification_evidence_refs:
                raise ValueError("verification Evidence requires outcome_event_id")
            if feedback_status != "unknown":
                raise ValueError("non-unknown feedback requires Outcome and verification Evidence")
        else:
            require_evo_id(outcome_event_id, field="outcome_event_id", kind="event")
            outcome = next(
                (event for event in events if event.event_id == outcome_event_id and event.event_type == "OutcomeClassified" and isinstance(event.payload, OutcomeClassifiedPayload)),
                None,
            )
            if outcome is None or outcome.refs.run_id != run_id:
                raise ValueError("Outcome must exist in the current Context Pack Run")
            if context_event.sequence is None or outcome.sequence is None or outcome.sequence <= context_event.sequence:
                raise ValueError("Outcome must occur after the recorded Context Pack")
            if not verification_evidence_refs:
                raise ValueError("Outcome feedback requires verification Evidence")
            try:
                attest_outcome_event(events, outcome)
            except OutcomeIntegrityError as error:
                raise ValueError("Outcome does not match the canonical prior Evidence for this Run") from error
            evidence_records = resolve_evidence_records(
                events,
                verification_evidence_refs,
                run_id=run_id,
                require_verified=True,
                deterministic_only=True,
                require_exit_code=True,
            )
            if not set(verification_evidence_refs).issubset(outcome.payload.evidence_refs):
                raise ValueError("verification Evidence must be referenced by the Outcome")
            evidence_events = {event.refs.evidence_id: event for event in events if event.event_type == "EvidenceRecorded" and event.refs.evidence_id in verification_evidence_refs}
            if any(
                evidence_events[record.evidence_id].sequence is None
                or evidence_events[record.evidence_id].sequence <= context_event.sequence
                or evidence_events[record.evidence_id].sequence >= outcome.sequence
                for record in evidence_records
            ):
                raise ValueError("verification Evidence must occur after the Context Pack and before the Outcome")
            verification_succeeded = all(record.payload.exit_code == 0 for record in evidence_records)
            if feedback_status == "helpful" and (outcome.payload.outcome != "success" or not verification_succeeded):
                raise ValueError("helpful feedback requires a canonical successful Outcome and passing verification Evidence")
            if feedback_status == "harmful" and (outcome.payload.outcome == "success" and verification_succeeded):
                raise ValueError("harmful feedback requires a non-success Outcome or failing verification Evidence")

        rank = None if selected_index is None else context_payload.selected_ranks[selected_index]
        packed_token_cost = None if selected_index is None else context_payload.selected_packed_token_costs[selected_index]
        truncated = False if selected_index is None else context_payload.selected_truncated[selected_index]
        omission_reason = None if omitted_index is None else context_payload.omitted_reasons[omitted_index]
        payload = MemoryUsedPayload(
            reason=redact_text(reason_text)[0],
            retrieval_rank=rank,
            helpfulness=None if feedback_status == "unknown" else feedback_status,
            context_pack_id=pack.context_pack_id,
            usage_status=usage_status,
            packed_token_cost=packed_token_cost,
            truncated=truncated,
            outcome_event_id=outcome_event_id,
            verification_evidence_refs=verification_evidence_refs,
            feedback_status=feedback_status,
            extensions={
                "canonical_context_pack_event_id": context_event.event_id,
                "omission_reason": omission_reason,
            },
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="MemoryUsed",
            refs=EvoReferences(
                run_id=run_id,
                plan_id=context_event.refs.plan_id,
                node_id=context_event.refs.node_id,
                memory_id=memory_id,
                context_pack_id=context_event.refs.context_pack_id,
                parent_event_id=context_event.event_id,
            ),
            payload=payload,
        )
        try:
            appended = self._service.store.append(event)
        except EvoStoreError as error:
            return MemoryUseResult(
                False,
                diagnostic=MemoryUseDiagnostic("memory_use_recording_failed", str(error)),
            )
        except Exception as error:  # noqa: BLE001 - feedback persistence cannot change the Run
            return MemoryUseResult(
                False,
                diagnostic=MemoryUseDiagnostic(
                    "memory_use_recording_failed",
                    f"unexpected Memory use recorder failure: {error}",
                ),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = MemoryUseDiagnostic(
                appended.diagnostic.code,
                appended.diagnostic.message,
            )
        return MemoryUseResult(True, appended.event, diagnostic)

    def _candidates(
        self,
        signature: QuerySignature,
        *,
        user_id: str | None,
        session_id: str | None,
        at: datetime | None,
        include_restricted: bool,
    ) -> tuple[tuple[RetrievalHit, ...], tuple[ContextOmission, ...]]:
        retrieved_at = self._resolve_time(at)
        terms = signature.terms
        if not terms:
            return (), ()
        hits: list[RetrievalHit] = []
        omissions: list[ContextOmission] = []
        projection = self._service.projection()
        conflict_counts: Counter[str] = Counter()
        for entry in projection.entries:
            record = entry.record
            if (
                record.conflict_group is None
                or entry.diagnostics
                or entry.effective_status != "active"
                or _freshness(record, retrieved_at) <= 0
                or (record.sensitivity == "restricted" and not include_restricted)
                or not record.is_readable_by(
                    project_id=self._service.project_id,
                    user_id=user_id,
                    session_id=session_id,
                )
            ):
                continue
            conflict_counts[record.conflict_group] += 1
        for entry in projection.entries:
            record = entry.record
            if not record.is_readable_by(
                project_id=self._service.project_id,
                user_id=user_id,
                session_id=session_id,
            ):
                continue
            matched = _matched_terms(terms, record.content)
            if not matched:
                continue
            if entry.diagnostics:
                omissions.append(ContextOmission(record.memory_id, "projection_diagnostic"))
                continue
            if entry.effective_status != "active":
                omissions.append(ContextOmission(record.memory_id, f"status:{entry.effective_status}"))
                continue
            freshness = _freshness(record, retrieved_at)
            if freshness <= 0:
                omissions.append(ContextOmission(record.memory_id, "expired"))
                continue
            if record.sensitivity == "restricted" and not include_restricted:
                omissions.append(ContextOmission(record.memory_id, "restricted"))
                continue
            exact_error = bool(signature.error_type and signature.error_type.casefold() in record.content.casefold())
            scope_specificity = int(record.scope.user_id is not None) + 2 * int(record.scope.session_id is not None)
            reasons = [
                "project_scope",
                "keyword:" + ",".join(matched),
                f"confidence:{record.confidence:.3f}",
                f"freshness:{freshness:.3f}",
                f"scope_specificity:{scope_specificity}",
            ]
            if exact_error:
                reasons.append("exact_error_type")
            placeholder = RetrievalHit(
                record=record,
                score=len(matched) + (1000 if exact_error else 0),
                reasons=tuple(reasons),
                token_cost=0,
                matched_terms=matched,
                exact_error_match=exact_error,
                confidence_score=record.confidence,
                freshness_score=freshness,
                scope_specificity=scope_specificity,
            )
            hits.append(
                replace(
                    placeholder,
                    token_cost=self._estimator.estimate(_render_item(placeholder, record.content)),
                )
            )

        unresolved_groups = {group for group, count in conflict_counts.items() if count > 1}
        resolved_hits: list[RetrievalHit] = []
        for hit in hits:
            group = hit.record.conflict_group
            if group is not None and group in unresolved_groups:
                omissions.append(
                    ContextOmission(
                        hit.record.memory_id,
                        "conflict_group",
                        f"conflict_group={group}",
                    )
                )
                continue
            resolved_hits.append(hit)
        resolved_hits.sort(
            key=lambda hit: (
                -int(hit.exact_error_match),
                -len(hit.matched_terms),
                -hit.confidence_score,
                -hit.freshness_score,
                -hit.scope_specificity,
                hit.record.memory_id,
            )
        )
        ranked = tuple(replace(hit, rank=index) for index, hit in enumerate(resolved_hits, start=1))
        return ranked, tuple(omissions)

    def _largest_fitting_item(
        self,
        hit: RetrievalHit,
        *,
        existing: tuple[ContextPackItem, ...],
        token_budget: int,
    ) -> ContextPackItem | None:
        content = hit.record.content
        if not content or token_budget == 0:
            return None
        anchor, minimum_length = _match_anchor(content, hit.matched_terms)
        low = max(1, minimum_length)
        high = len(content) - 1
        best: ContextPackItem | None = None
        while low <= high:
            length = (low + high) // 2
            start, end = _window(
                len(content),
                anchor,
                length,
                minimum_length,
            )
            candidate = _pack_item(
                hit,
                content[start:end],
                self._estimator,
                start_offset=start,
                end_offset=end,
            )
            if self._estimate_items((*existing, candidate)) <= token_budget:
                best = candidate
                low = length + 1
            else:
                high = length - 1
        return best

    def _estimate_items(self, items: tuple[ContextPackItem, ...]) -> int:
        return self._estimator.estimate(_render_pack(items))

    def _record_pack(self, pack: ContextPack) -> ContextPack:
        if pack.run_id is None:
            return pack
        payload = ContextPackRecordedPayload(
            context_pack_schema_version="v1",
            context_pack_id=pack.context_pack_id,
            query_signature_hash=pack.query_signature_hash,
            token_budget=pack.token_budget,
            used_tokens=pack.used_tokens,
            estimator_id=pack.estimator_id,
            selected_memory_ids=tuple(item.hit.record.memory_id for item in pack.items),
            selected_ranks=tuple(item.hit.rank for item in pack.items),
            selected_original_token_costs=tuple(item.original_token_cost for item in pack.items),
            selected_packed_token_costs=tuple(item.packed_token_cost for item in pack.items),
            selected_truncated=tuple(item.truncated for item in pack.items),
            selected_start_offsets=tuple(item.start_offset for item in pack.items),
            selected_end_offsets=tuple(item.end_offset for item in pack.items),
            omitted_memory_ids=tuple(item.memory_id for item in pack.omissions),
            omitted_reasons=tuple(item.reason for item in pack.omissions),
            extensions={"recording_version": "p4-audit-fix-1"},
        )
        events = self._service.store.list_events()
        parent_event_id = next(
            (event.event_id for event in reversed(events) if event.refs.run_id == pack.run_id),
            None,
        )
        event = EvoEvent(
            event_id=new_evo_id("event"),
            event_type="ContextPackRecorded",
            refs=EvoReferences(
                run_id=pack.run_id,
                plan_id=pack.plan_id,
                node_id=pack.node_id,
                context_pack_id=pack.context_pack_id,
                parent_event_id=parent_event_id,
            ),
            payload=payload,
        )
        try:
            appended = self._service.store.append(event)
        except EvoStoreError as error:
            return replace(
                pack,
                diagnostic=ContextPackDiagnostic(
                    "context_pack_recording_failed",
                    str(error),
                ),
            )
        except Exception as error:  # noqa: BLE001 - recording failure cannot remove retrieved context
            return replace(
                pack,
                diagnostic=ContextPackDiagnostic(
                    "context_pack_recording_failed",
                    f"unexpected Context Pack recorder failure: {error}",
                ),
            )
        diagnostic = None
        if appended.diagnostic is not None:
            diagnostic = ContextPackDiagnostic(
                appended.diagnostic.code,
                appended.diagnostic.message,
            )
        return replace(
            pack,
            recorded_event_id=appended.event.event_id,
            diagnostic=diagnostic,
        )

    def _resolve_time(self, value: datetime | None) -> datetime:
        resolved = self._clock() if value is None else value
        if not isinstance(resolved, datetime) or resolved.tzinfo is None:
            raise ValueError("retrieval time must be timezone-aware")
        return resolved.astimezone(UTC)

    def _authorize_restricted(
        self,
        *,
        include_restricted: bool,
        authorization: MemoryAccessAuthorization | None,
        user_id: str | None,
        session_id: str | None,
    ) -> None:
        if not include_restricted:
            return
        if authorization is None or not authorization.allow_restricted:
            raise ValueError("restricted Memory requires an explicit authorization context")
        if authorization.project_id != self._service.project_id:
            raise ValueError("restricted Memory authorization is for another project")
        if user_id is not None and authorization.user_id != user_id:
            raise ValueError("restricted Memory authorization user does not match query")
        if session_id is not None and authorization.session_id != session_id:
            raise ValueError("restricted Memory authorization session does not match query")


def _freshness(record: MemoryRecord, at: datetime) -> float:
    total = (record.expires_at - record.created_at).total_seconds()
    remaining = (record.expires_at - at).total_seconds()
    if remaining <= 0:
        return 0.0
    return min(1.0, max(0.0, remaining / total))


def _matched_terms(terms: frozenset[str], content: str) -> tuple[str, ...]:
    r"""Match normalized keywords without assuming whitespace-delimited scripts.

    ``\w`` tokenization works for identifiers and space-delimited languages, but
    a Chinese phrase embedded next to other Chinese text becomes one larger token.
    Preserve exact-token matching and add deterministic substring matching so the
    provider-neutral fallback remains usable for CJK and mixed-script Memory.
    """

    folded = content.casefold()
    content_terms = frozenset(_WORDS.findall(folded))
    return tuple(sorted(term for term in terms if term in content_terms or (any(ord(character) > 127 for character in term) and term in folded)))


def _pack_item(
    hit: RetrievalHit,
    content: str,
    estimator: TokenEstimator,
    *,
    start_offset: int = 0,
    end_offset: int | None = None,
) -> ContextPackItem:
    resolved_end = len(hit.record.content) if end_offset is None else end_offset
    rendered = _render_item(hit, content)
    return ContextPackItem(
        hit=hit,
        packed_content=content,
        original_token_cost=hit.token_cost,
        packed_token_cost=estimator.estimate(rendered),
        truncated=start_offset != 0 or resolved_end != len(hit.record.content),
        start_offset=start_offset,
        end_offset=resolved_end,
    )


def _render_item(hit: RetrievalHit, content: str) -> str:
    return f"[memory id={hit.record.memory_id} layer={hit.record.layer} " f"confidence={hit.confidence_score:.3f} freshness={hit.freshness_score:.3f}]\n" f"{content}"


def _render_pack(items: tuple[ContextPackItem, ...]) -> str:
    return "\n\n".join(_render_item(item.hit, item.packed_content) for item in items)


def _match_anchor(content: str, matched_terms: tuple[str, ...]) -> tuple[int, int]:
    folded_parts: list[str] = []
    folded_to_original: list[int] = []
    for index, character in enumerate(content):
        piece = character.casefold()
        folded_parts.append(piece)
        folded_to_original.extend(index for _ in piece)
    folded = "".join(folded_parts)
    located: list[tuple[int, int]] = []
    for term in matched_terms:
        normalized = term.casefold()
        offset = folded.find(normalized)
        if offset < 0 or not normalized:
            continue
        original_start = folded_to_original[offset]
        original_end = folded_to_original[offset + len(normalized) - 1] + 1
        located.append((original_start, original_end - original_start))
    if not located:
        return 0, 1
    anchor, term_length = min(located, key=lambda item: (item[0], -item[1]))
    return anchor, max(1, term_length)


def _window(
    content_length: int,
    anchor: int,
    length: int,
    anchor_length: int,
) -> tuple[int, int]:
    start = max(0, anchor - max(0, length - anchor_length) // 2)
    end = min(content_length, start + length)
    if end < anchor + anchor_length:
        end = min(content_length, anchor + anchor_length)
        start = max(0, end - length)
    start = max(0, end - length)
    return start, end


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string")
    return value.strip()
