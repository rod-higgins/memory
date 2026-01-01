"""
LanceDB vector store for long-term memories.

High-performance semantic search with <10ms latency.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import lancedb
from lancedb.pydantic import LanceModel, Vector

from memory.schema.memory_entry import (
    ConfidenceScore,
    MemoryEntry,
    MemorySource,
    MemoryTier,
    MemoryType,
    SourceType,
    TruthCategory,
)
from memory.storage.base import BaseStore


class MemoryLanceModel(LanceModel):
    """LanceDB model for memory entries."""

    id: str
    content: str
    summary: str | None = None
    content_hash: str

    # Classification (stored as strings for LanceDB)
    tier: str
    truth_category: str
    memory_type: str

    # Confidence
    confidence_overall: float
    confidence_source_reliability: float
    confidence_recency: float
    corroboration_count: int
    contradiction_count: int

    # Semantic - vector dimension depends on embedding model
    vector: Vector(384)  # type: ignore[valid-type]

    # Organization (stored as JSON strings)
    tags_json: str
    domains_json: str
    entities_json: str

    # Relationships (stored as JSON strings)
    related_memories_json: str
    contradicts_json: str
    supports_json: str
    supersedes: str | None = None

    # Sources (stored as JSON string)
    sources_json: str

    # Temporal
    created_at: str
    updated_at: str
    last_accessed: str
    access_count: int

    # Lifecycle
    is_active: bool
    promotion_history_json: str

    # Metadata
    metadata_json: str


class LanceDBStore(BaseStore):
    """
    LanceDB-based storage for long-term memories.

    Features:
    - Fast vector similarity search (<10ms on 1M+ vectors)
    - Hybrid search (vector + metadata filtering)
    - File-based persistence
    - HNSW indexing
    """

    tier = MemoryTier.LONG_TERM

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "memories",
        embedding_dim: int = 384,
    ):
        self._db_path = Path(db_path).expanduser()
        self._table_name = table_name
        self._embedding_dim = embedding_dim
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    async def initialize(self) -> None:
        """Initialize LanceDB connection and table."""
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))

        # Check if table exists
        try:
            if self._table_name in self._db.table_names():
                self._table = self._db.open_table(self._table_name)
            else:
                # Create empty table with schema
                self._table = self._db.create_table(
                    self._table_name,
                    schema=MemoryLanceModel,
                    mode="overwrite",
                )
        except Exception as e:
            # If schema-only creation fails, create with empty data
            try:
                self._table = self._db.create_table(
                    self._table_name,
                    data=[],
                    schema=MemoryLanceModel,
                    mode="overwrite",
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize LanceDB: {e}, then {e2}") from e2

    async def close(self) -> None:
        """Close LanceDB connection."""
        # LanceDB handles cleanup automatically
        self._db = None
        self._table = None

    async def store(self, memory: MemoryEntry) -> MemoryEntry:
        """Store a memory entry."""
        if self._table is None:
            raise RuntimeError("Store not initialized")

        memory.tier = MemoryTier.LONG_TERM
        lance_model = self._to_lance_model(memory)

        self._table.add([lance_model.model_dump()])
        return memory

    async def get(self, memory_id: UUID) -> MemoryEntry | None:
        """Retrieve a memory by ID."""
        if self._table is None:
            raise RuntimeError("Store not initialized")

        results = self._table.search().where(f"id = '{str(memory_id)}'").limit(1).to_list()

        if results:
            memory = self._from_lance_dict(results[0])
            memory.record_access()
            await self.update(memory)
            return memory
        return None

    async def update(self, memory: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry."""
        if self._table is None:
            raise RuntimeError("Store not initialized")

        memory.updated_at = datetime.now()

        # LanceDB doesn't have native update, so we delete and re-add
        self._table.delete(f"id = '{str(memory.id)}'")
        lance_model = self._to_lance_model(memory)
        self._table.add([lance_model.model_dump()])

        return memory

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        if self._table is None:
            raise RuntimeError("Store not initialized")

        # Check if exists first
        results = self._table.search().where(f"id = '{str(memory_id)}'").limit(1).to_list()

        if results:
            self._table.delete(f"id = '{str(memory_id)}'")
            return True
        return False

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """Perform vector similarity search."""
        if self._table is None:
            return [], []  # Not initialized yet

        query = self._table.search(embedding).limit(limit)

        # Apply filters
        if filters:
            where_clause = self._build_where_clause(filters)
            if where_clause:
                query = query.where(where_clause)

        results = query.to_list()

        memories = [self._from_lance_dict(r) for r in results]
        scores = [r.get("_distance", 0.0) for r in results]

        # Convert distance to similarity (LanceDB uses L2 distance by default)
        scores = [1.0 / (1.0 + d) for d in scores]

        return memories, scores

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """Search by embedding vector. Alias for vector_search."""
        memories, _ = await self.vector_search(query_embedding, limit, filters)
        return memories

    async def text_search(
        self,
        query_text: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """Perform full-text search (using LIKE for simplicity)."""
        if self._table is None:
            return [], []  # Not initialized yet

        # Build where clause for text search
        text_clause = f"content LIKE '%{query_text}%'"

        if filters:
            filter_clause = self._build_where_clause(filters)
            if filter_clause:
                text_clause = f"({text_clause}) AND ({filter_clause})"

        results = self._table.search().where(text_clause).limit(limit).to_list()

        memories = [self._from_lance_dict(r) for r in results]
        # Simple relevance scoring based on content length ratio
        scores = [min(len(query_text) / max(len(m.content), 1) * 2, 1.0) for m in memories]

        return memories, scores

    async def find_by_hash(self, content_hash: str) -> MemoryEntry | None:
        """Find a memory by its content hash."""
        if self._table is None:
            return None  # Not initialized yet

        try:
            results = self._table.search().where(f"content_hash = '{content_hash}'").limit(1).to_list()

            if results:
                return self._from_lance_dict(results[0])
        except Exception:
            pass  # Table may be empty or have issues
        return None

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count memories matching filters."""
        if self._table is None:
            return 0  # Not initialized yet

        try:
            if filters:
                where_clause = self._build_where_clause(filters)
                if where_clause:
                    return len(self._table.search().where(where_clause).to_list())

            return self._table.count_rows()
        except Exception:
            return 0

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """List all memories with pagination."""
        if self._table is None:
            raise RuntimeError("Store not initialized")

        query = self._table.search()

        if filters:
            where_clause = self._build_where_clause(filters)
            if where_clause:
                query = query.where(where_clause)

        # LanceDB doesn't have native offset, so we fetch more and slice
        results = query.limit(offset + limit).to_list()

        return [self._from_lance_dict(r) for r in results[offset:]]

    # Conversion methods
    def _to_lance_model(self, memory: MemoryEntry) -> MemoryLanceModel:
        """Convert MemoryEntry to LanceDB model."""
        # Use zero vector if no embedding
        vector = memory.embedding or [0.0] * self._embedding_dim

        return MemoryLanceModel(
            id=str(memory.id),
            content=memory.content,
            summary=memory.summary,
            content_hash=memory.content_hash,
            tier=memory.tier.value,
            truth_category=memory.truth_category.value,
            memory_type=memory.memory_type.value,
            confidence_overall=memory.confidence.overall,
            confidence_source_reliability=memory.confidence.source_reliability,
            confidence_recency=memory.confidence.recency,
            corroboration_count=memory.confidence.corroboration_count,
            contradiction_count=memory.confidence.contradiction_count,
            vector=vector,
            tags_json=json.dumps(memory.tags),
            domains_json=json.dumps(memory.domains),
            entities_json=json.dumps(memory.entities),
            related_memories_json=json.dumps([str(u) for u in memory.related_memories]),
            contradicts_json=json.dumps([str(u) for u in memory.contradicts]),
            supports_json=json.dumps([str(u) for u in memory.supports]),
            supersedes=str(memory.supersedes) if memory.supersedes else None,
            sources_json=json.dumps([s.model_dump() for s in memory.sources], default=str),
            created_at=memory.created_at.isoformat(),
            updated_at=memory.updated_at.isoformat(),
            last_accessed=memory.last_accessed.isoformat(),
            access_count=memory.access_count,
            is_active=memory.is_active,
            promotion_history_json=json.dumps(memory.promotion_history),
            metadata_json=json.dumps(memory.metadata),
        )

    def _from_lance_dict(self, data: dict[str, Any]) -> MemoryEntry:
        """Convert LanceDB result dict to MemoryEntry."""
        # Parse sources
        sources_data = json.loads(data.get("sources_json", "[]"))
        sources = []
        for s in sources_data:
            if isinstance(s.get("timestamp"), str):
                s["timestamp"] = datetime.fromisoformat(s["timestamp"])
            s["source_type"] = SourceType(s["source_type"])
            sources.append(MemorySource(**s))

        # Parse UUIDs
        related = [UUID(u) for u in json.loads(data.get("related_memories_json", "[]"))]
        contradicts = [UUID(u) for u in json.loads(data.get("contradicts_json", "[]"))]
        supports = [UUID(u) for u in json.loads(data.get("supports_json", "[]"))]
        supersedes = UUID(data["supersedes"]) if data.get("supersedes") else None

        return MemoryEntry(
            id=UUID(data["id"]),
            content=data["content"],
            summary=data.get("summary"),
            tier=MemoryTier(data["tier"]),
            truth_category=TruthCategory(data["truth_category"]),
            memory_type=MemoryType(data["memory_type"]),
            sources=sources,
            confidence=ConfidenceScore(
                overall=data["confidence_overall"],
                source_reliability=data["confidence_source_reliability"],
                recency=data["confidence_recency"],
                corroboration_count=data["corroboration_count"],
                contradiction_count=data["contradiction_count"],
            ),
            embedding=data.get("vector"),
            tags=json.loads(data.get("tags_json", "[]")),
            domains=json.loads(data.get("domains_json", "[]")),
            entities=json.loads(data.get("entities_json", "[]")),
            related_memories=related,
            contradicts=contradicts,
            supports=supports,
            supersedes=supersedes,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            is_active=data["is_active"],
            promotion_history=json.loads(data.get("promotion_history_json", "[]")),
            metadata=json.loads(data.get("metadata_json", "{}")),
        )

    def _build_where_clause(self, filters: dict[str, Any]) -> str:
        """Build SQL-like where clause from filters."""
        clauses = []

        for key, value in filters.items():
            if key == "truth_category" and isinstance(value, list):
                values = ", ".join(f"'{v}'" for v in value)
                clauses.append(f"truth_category IN ({values})")
            elif key == "memory_type" and isinstance(value, list):
                values = ", ".join(f"'{v}'" for v in value)
                clauses.append(f"memory_type IN ({values})")
            elif key == "confidence_gte":
                clauses.append(f"confidence_overall >= {value}")
            elif key == "is_active":
                clauses.append(f"is_active = {str(value).lower()}")

        return " AND ".join(clauses) if clauses else ""
