"""
Unified storage manager for the three-tier memory system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from memory.schema.memory_entry import MemoryEntry, MemoryTier
from memory.storage.base import BaseStore
from memory.storage.dict_store import DictStore
from memory.storage.lancedb_store import LanceDBStore
from memory.storage.sqlite_store import SQLiteStore


class StorageManager:
    """
    Unified manager for all three storage tiers.

    Handles:
    - Initialization of all stores
    - Routing operations to correct tier
    - Cross-tier migrations
    - Unified querying
    """

    def __init__(
        self,
        base_path: str | Path = "~/memory/data",
        short_term_ttl_hours: int = 72,
        embedding_dim: int = 384,
    ):
        self._base_path = Path(base_path).expanduser()
        self._embedding_dim = embedding_dim

        # Initialize stores
        self._short_term = DictStore(ttl_hours=short_term_ttl_hours)
        self._long_term = LanceDBStore(
            db_path=self._base_path / "long_term" / "lancedb",
            embedding_dim=embedding_dim,
        )
        self._persistent = SQLiteStore(
            db_path=self._base_path / "persistent" / "core.sqlite"
        )

        self._stores: dict[MemoryTier, BaseStore] = {
            MemoryTier.SHORT_TERM: self._short_term,
            MemoryTier.LONG_TERM: self._long_term,
            MemoryTier.PERSISTENT: self._persistent,
        }

        self._initialized = False

    @property
    def short_term(self) -> DictStore:
        """Access short-term store directly."""
        return self._short_term

    @property
    def long_term(self) -> LanceDBStore:
        """Access long-term store directly."""
        return self._long_term

    @property
    def persistent(self) -> SQLiteStore:
        """Access persistent store directly."""
        return self._persistent

    async def initialize(self) -> None:
        """Initialize all storage backends."""
        if self._initialized:
            return

        self._base_path.mkdir(parents=True, exist_ok=True)

        await self._short_term.initialize()
        await self._long_term.initialize()
        await self._persistent.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Close all storage backends."""
        await self._short_term.close()
        await self._long_term.close()
        await self._persistent.close()
        self._initialized = False

    def get_store(self, tier: MemoryTier) -> BaseStore:
        """Get the store for a specific tier."""
        return self._stores[tier]

    # CRUD operations (routed by tier)
    async def store(self, memory: MemoryEntry) -> MemoryEntry:
        """Store a memory in its designated tier."""
        store = self._stores[memory.tier]
        return await store.store(memory)

    async def get(self, memory_id: UUID, tier: MemoryTier | None = None) -> MemoryEntry | None:
        """
        Retrieve a memory by ID.

        If tier is specified, only search that tier.
        Otherwise, search all tiers (persistent first, then long-term, then short-term).
        """
        if tier:
            return await self._stores[tier].get(memory_id)

        # Search all tiers
        for t in [MemoryTier.PERSISTENT, MemoryTier.LONG_TERM, MemoryTier.SHORT_TERM]:
            memory = await self._stores[t].get(memory_id)
            if memory:
                return memory
        return None

    async def update(self, memory: MemoryEntry) -> MemoryEntry:
        """Update a memory in its current tier."""
        store = self._stores[memory.tier]
        return await store.update(memory)

    async def delete(self, memory_id: UUID, tier: MemoryTier | None = None) -> bool:
        """
        Delete a memory by ID.

        If tier is specified, only delete from that tier.
        Otherwise, try all tiers.
        """
        if tier:
            return await self._stores[tier].delete(memory_id)

        # Try all tiers
        for t in [MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM, MemoryTier.PERSISTENT]:
            if await self._stores[t].delete(memory_id):
                return True
        return False

    # Cross-tier operations
    async def migrate_tier(
        self,
        memory: MemoryEntry,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
    ) -> MemoryEntry:
        """
        Migrate a memory from one tier to another.

        1. Store in new tier
        2. Delete from old tier
        3. Update memory's tier field
        """
        # Update tier
        memory.tier = to_tier

        # Store in new tier
        await self._stores[to_tier].store(memory)

        # Delete from old tier
        await self._stores[from_tier].delete(memory.id)

        return memory

    # Search operations (across all tiers)
    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 20,
        tiers: list[MemoryTier] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """
        Perform vector search across specified tiers.

        Results are merged and sorted by score.
        """
        tiers = tiers or list(MemoryTier)
        all_memories: list[MemoryEntry] = []
        all_scores: list[float] = []

        for tier in tiers:
            store = self._stores[tier]
            memories, scores = await store.vector_search(
                embedding=embedding,
                limit=limit,
                filters=filters,
            )
            all_memories.extend(memories)
            all_scores.extend(scores)

        # Sort by score descending
        sorted_pairs = sorted(zip(all_memories, all_scores), key=lambda x: x[1], reverse=True)
        sorted_pairs = sorted_pairs[:limit]

        return [m for m, s in sorted_pairs], [s for m, s in sorted_pairs]

    async def text_search(
        self,
        query_text: str,
        limit: int = 20,
        tiers: list[MemoryTier] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """
        Perform text search across specified tiers.

        Results are merged and sorted by score.
        """
        tiers = tiers or list(MemoryTier)
        all_memories: list[MemoryEntry] = []
        all_scores: list[float] = []

        for tier in tiers:
            store = self._stores[tier]
            memories, scores = await store.text_search(
                query_text=query_text,
                limit=limit,
                filters=filters,
            )
            all_memories.extend(memories)
            all_scores.extend(scores)

        # Sort by score descending
        sorted_pairs = sorted(zip(all_memories, all_scores), key=lambda x: x[1], reverse=True)
        sorted_pairs = sorted_pairs[:limit]

        return [m for m, s in sorted_pairs], [s for m, s in sorted_pairs]

    async def find_by_hash(self, content_hash: str) -> MemoryEntry | None:
        """Find a memory by content hash across all tiers."""
        for tier in [MemoryTier.PERSISTENT, MemoryTier.LONG_TERM, MemoryTier.SHORT_TERM]:
            memory = await self._stores[tier].find_by_hash(content_hash)
            if memory:
                return memory
        return None

    async def count(
        self,
        tier: MemoryTier | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """
        Count memories by tier.

        If tier is specified, return count for that tier only.
        Otherwise, return counts for all tiers.
        """
        if tier:
            count = await self._stores[tier].count(filters)
            return {tier.value: count}

        counts = {}
        for t in MemoryTier:
            counts[t.value] = await self._stores[t].count(filters)
        counts["total"] = sum(counts.values())
        return counts

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the memory system."""
        counts = await self.count()

        return {
            "counts": counts,
            "tiers": {
                "short_term": {
                    "backend": "dict",
                    "count": counts.get("short_term", 0),
                },
                "long_term": {
                    "backend": "lancedb",
                    "count": counts.get("long_term", 0),
                    "path": str(self._base_path / "long_term" / "lancedb"),
                },
                "persistent": {
                    "backend": "sqlite",
                    "count": counts.get("persistent", 0),
                    "path": str(self._base_path / "persistent" / "core.sqlite"),
                },
            },
            "embedding_dim": self._embedding_dim,
            "base_path": str(self._base_path),
        }
