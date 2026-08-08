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
from bauhinia_agent.memory.service import MemoryService, MemoryWriteDisabledError
from bauhinia_agent.memory.retrieval import ContextPack, MemoryRetriever, QuerySignature, RetrievalHit

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
    "MemoryService",
    "MemoryWriteDisabledError",
    "ContextPack",
    "MemoryRetriever",
    "QuerySignature",
    "RetrievalHit",
]
