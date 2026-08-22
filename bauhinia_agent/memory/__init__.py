"""P3 project-isolated memory domain contracts.

The package intentionally has no store, retrieval index, or Agent Loop wiring.
"""

from bauhinia_agent.memory.models import (
    MEMORY_LAYER_RULES,
    MemoryChangeKind,
    MemoryLifecycleChange,
    MemoryLayer,
    MemoryLayerRule,
    MemoryModelError,
    MemoryOrigin,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemorySensitivity,
    MemoryStatus,
)
from bauhinia_agent.memory.projection import (
    MemoryProjection,
    MemoryProjectionDiagnostic,
    MemoryProjectionEntry,
    build_memory_projection,
)
from bauhinia_agent.memory.service import (
    MemoryActorKind,
    MemoryDiagnostic,
    MemoryLifecycleResult,
    MemoryService,
    MemoryWriteDisabledError,
)
from bauhinia_agent.memory.retrieval import (
    ContextOmission,
    ContextPack,
    ContextPackDiagnostic,
    ContextPackItem,
    HeuristicTokenEstimator,
    MemoryAccessAuthorization,
    MemoryRetriever,
    MemoryUseDiagnostic,
    MemoryUseResult,
    QuerySignature,
    RetrievalHit,
    TokenEstimator,
)

__all__ = [
    "MEMORY_LAYER_RULES",
    "MemoryChangeKind",
    "MemoryLifecycleChange",
    "MemoryLayer",
    "MemoryLayerRule",
    "MemoryModelError",
    "MemoryOrigin",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryScope",
    "MemorySensitivity",
    "MemoryStatus",
    "MemoryActorKind",
    "MemoryDiagnostic",
    "MemoryLifecycleResult",
    "MemoryProjection",
    "MemoryProjectionDiagnostic",
    "MemoryProjectionEntry",
    "MemoryService",
    "MemoryWriteDisabledError",
    "ContextPack",
    "ContextPackDiagnostic",
    "ContextPackItem",
    "ContextOmission",
    "HeuristicTokenEstimator",
    "MemoryAccessAuthorization",
    "MemoryRetriever",
    "MemoryUseDiagnostic",
    "MemoryUseResult",
    "QuerySignature",
    "RetrievalHit",
    "TokenEstimator",
    "build_memory_projection",
]
