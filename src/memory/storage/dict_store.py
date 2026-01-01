"""
In-memory dictionary store for short-term memories.

Fast, ephemeral storage with optional TTL expiration.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from memory.schema.memory_entry import MemoryEntry, MemoryTier
from memory.storage.base import BaseStore


class DictStore(BaseStore):
    """
    In-memory dictionary-based storage for short-term memories.

    Features:
    - Fast O(1) access by ID
    - Optional TTL expiration
    - Simple filtering
    - No persistence (ephemeral)
    """

    tier = MemoryTier.SHORT_TERM

    def __init__(
        self,
        ttl_hours: int = 72,
        max_entries: int = 10000,
    ):
        self._memories: dict[UUID, MemoryEntry] = {}
        self._hash_index: dict[str, UUID] = {}  # content_hash -> id
        self._ttl_hours = ttl_hours
        self._max_entries = max_entries
        self._cleanup_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        """Start the cleanup task."""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def close(self) -> None:
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def store(self, memory: MemoryEntry) -> MemoryEntry:
        """Store a memory entry."""
        # Enforce max entries by removing oldest if needed
        if len(self._memories) >= self._max_entries:
            await self._evict_oldest()

        memory.tier = MemoryTier.SHORT_TERM
        self._memories[memory.id] = memory
        self._hash_index[memory.content_hash] = memory.id
        return memory

    async def get(self, memory_id: UUID) -> MemoryEntry | None:
        """Retrieve a memory by ID."""
        memory = self._memories.get(memory_id)
        if memory:
            memory.record_access()
        return memory

    async def update(self, memory: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry."""
        if memory.id not in self._memories:
            raise KeyError(f"Memory {memory.id} not found")

        # Update hash index if content changed
        old_memory = self._memories[memory.id]
        if old_memory.content_hash in self._hash_index:
            del self._hash_index[old_memory.content_hash]

        memory.updated_at = datetime.now()
        self._memories[memory.id] = memory
        self._hash_index[memory.content_hash] = memory.id
        return memory

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        if memory_id in self._memories:
            memory = self._memories[memory_id]
            if memory.content_hash in self._hash_index:
                del self._hash_index[memory.content_hash]
            del self._memories[memory_id]
            return True
        return False

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """
        Simple cosine similarity search.

        Note: For short-term memory, we do a linear scan.
        For production, consider using a more efficient index.
        """
        if not embedding:
            return [], []

        results: list[tuple[MemoryEntry, float]] = []

        for memory in self._memories.values():
            if not self._matches_filters(memory, filters):
                continue

            if memory.embedding:
                score = self._cosine_similarity(embedding, memory.embedding)
                results.append((memory, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply limit
        results = results[:limit]

        memories = [r[0] for r in results]
        scores = [r[1] for r in results]

        return memories, scores

    async def text_search(
        self,
        query_text: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """Simple substring text search."""
        query_lower = query_text.lower()
        results: list[tuple[MemoryEntry, float]] = []

        for memory in self._memories.values():
            if not self._matches_filters(memory, filters):
                continue

            content_lower = memory.content.lower()
            if query_lower in content_lower:
                # Simple relevance: how much of content is the query
                score = len(query_text) / max(len(memory.content), 1)
                results.append((memory, min(score * 2, 1.0)))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]

        memories = [r[0] for r in results]
        scores = [r[1] for r in results]

        return memories, scores

    async def find_by_hash(self, content_hash: str) -> MemoryEntry | None:
        """Find a memory by its content hash."""
        memory_id = self._hash_index.get(content_hash)
        if memory_id:
            return self._memories.get(memory_id)
        return None

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count memories matching filters."""
        if not filters:
            return len(self._memories)

        return sum(1 for m in self._memories.values() if self._matches_filters(m, filters))

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """List all memories with pagination."""
        memories = list(self._memories.values())

        if filters:
            memories = [m for m in memories if self._matches_filters(m, filters)]

        # Sort by created_at descending
        memories.sort(key=lambda m: m.created_at, reverse=True)

        return memories[offset : offset + limit]

    # Private methods
    def _matches_filters(self, memory: MemoryEntry, filters: dict[str, Any] | None) -> bool:
        """Check if memory matches all filters."""
        if not filters:
            return True

        for key, value in filters.items():
            if key == "truth_category" and isinstance(value, list):
                if memory.truth_category.value not in value:
                    return False
            elif key == "memory_type" and isinstance(value, list):
                if memory.memory_type.value not in value:
                    return False
            elif key == "domains" and isinstance(value, list):
                if not any(d in memory.domains for d in value):
                    return False
            elif key == "tags" and isinstance(value, list):
                if not any(t in memory.tags for t in value):
                    return False
            elif key == "confidence_gte":
                if memory.confidence.overall < value:
                    return False
            elif key == "created_after":
                if memory.created_at < value:
                    return False
            elif key == "created_before":
                if memory.created_at > value:
                    return False

        return True

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def _evict_oldest(self) -> None:
        """Remove oldest memory to make room for new ones."""
        if not self._memories:
            return

        oldest = min(self._memories.values(), key=lambda m: m.last_accessed)
        await self.delete(oldest.id)

    async def _periodic_cleanup(self) -> None:
        """Periodically remove expired memories."""
        while True:
            await asyncio.sleep(3600)  # Check every hour

            cutoff = datetime.now() - timedelta(hours=self._ttl_hours)
            expired = [mid for mid, m in self._memories.items() if m.created_at < cutoff and m.is_active]

            for mid in expired:
                await self.delete(mid)
