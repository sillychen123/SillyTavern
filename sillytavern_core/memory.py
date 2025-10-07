"""Memory management helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence

from .models import MemoryEntry


@dataclass
class CompressionResult:
    retained: List[MemoryEntry]
    archived: List[MemoryEntry]
    summary: str


class MemoryCompressor:
    """Compress conversational memories based on weight and recency."""

    def __init__(self, retention_limit: int = 50, archive_threshold: float = 0.5) -> None:
        self.retention_limit = retention_limit
        self.archive_threshold = archive_threshold

    def _score(self, memory: MemoryEntry, now: datetime) -> float:
        age_seconds = max((now - memory.timestamp).total_seconds(), 1.0)
        recency_factor = 1.0 / age_seconds
        return memory.weight + recency_factor

    def compress(self, memories: Sequence[MemoryEntry]) -> CompressionResult:
        now = datetime.utcnow()
        scored = sorted(memories, key=lambda mem: self._score(mem, now), reverse=True)
        retained = scored[: self.retention_limit]
        archived = [mem for mem in scored[self.retention_limit :] if mem.weight < self.archive_threshold]
        summary = self._summarize(retained)
        return CompressionResult(retained=retained, archived=archived, summary=summary)

    def _summarize(self, memories: Iterable[MemoryEntry]) -> str:
        highlights = [mem.content for mem in memories][:5]
        if not highlights:
            return ""
        return "Key memories: " + "; ".join(highlights)


def boost_memory_weight(memories: Iterable[MemoryEntry], keyword: str, boost: float = 0.5) -> List[MemoryEntry]:
    keyword_lower = keyword.lower()
    result: List[MemoryEntry] = []
    for memory in memories:
        updated = memory.copy_with()
        if keyword_lower in memory.content.lower():
            updated = updated.copy_with(weight=memory.weight + boost)
        result.append(updated)
    return result
