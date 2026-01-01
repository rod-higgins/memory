"""
Base storage interface for all memory backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from memory.schema.memory_entry import MemoryEntry, MemoryTier


class BaseStore(ABC):
    """
    Abstract base class for memory storage backends.

    All storage implementations must implement these methods.
    """

    tier: MemoryTier

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create tables, indexes, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and cleanup resources."""
        pass

    # CRUD operations
    @abstractmethod
    async def store(self, memory: MemoryEntry) -> MemoryEntry:
        """Store a memory entry. Returns the stored entry with any updates."""
        pass

    @abstractmethod
    async def get(self, memory_id: UUID) -> MemoryEntry | None:
        """Retrieve a memory by ID."""
        pass

    @abstractmethod
    async def update(self, memory: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry."""
        pass

    @abstractmethod
    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory by ID. Returns True if deleted, False if not found."""
        pass

    # Search operations
    @abstractmethod
    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """
        Perform vector similarity search.

        Returns tuple of (memories, scores).
        """
        pass

    @abstractmethod
    async def text_search(
        self,
        query_text: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """
        Perform full-text search.

        Returns tuple of (memories, scores).
        """
        pass

    # Utility methods
    @abstractmethod
    async def find_by_hash(self, content_hash: str) -> MemoryEntry | None:
        """Find a memory by its content hash (for deduplication)."""
        pass

    @abstractmethod
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count memories matching the given filters."""
        pass

    @abstractmethod
    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """List all memories with pagination and optional filters."""
        pass

    # Bulk operations
    async def store_many(self, memories: list[MemoryEntry]) -> list[MemoryEntry]:
        """Store multiple memories. Default implementation calls store() in loop."""
        return [await self.store(m) for m in memories]

    async def delete_many(self, memory_ids: list[UUID]) -> int:
        """Delete multiple memories. Returns count of deleted entries."""
        count = 0
        for mid in memory_ids:
            if await self.delete(mid):
                count += 1
        return count
