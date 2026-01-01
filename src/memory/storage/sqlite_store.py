"""
SQLite store for persistent memories.

Human-inspectable, ACID-compliant archive for core truths and identity.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

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


class SQLiteStore(BaseStore):
    """
    SQLite-based storage for persistent memories.

    Features:
    - Human-inspectable database
    - ACID transactions
    - Full-text search via FTS5
    - Portable single-file database
    """

    tier = MemoryTier.PERSISTENT

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path).expanduser()
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize SQLite database and tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")

        # Create main table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT,
                content_hash TEXT UNIQUE,

                tier TEXT NOT NULL,
                truth_category TEXT NOT NULL,
                memory_type TEXT NOT NULL,

                confidence_overall REAL,
                confidence_source_reliability REAL,
                confidence_recency REAL,
                corroboration_count INTEGER,
                contradiction_count INTEGER,

                embedding BLOB,
                embedding_model TEXT,

                tags_json TEXT,
                domains_json TEXT,
                entities_json TEXT,

                related_memories_json TEXT,
                contradicts_json TEXT,
                supports_json TEXT,
                supersedes TEXT,

                sources_json TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,

                is_active INTEGER DEFAULT 1,
                promotion_history_json TEXT,
                metadata_json TEXT
            )
        """)

        # Create FTS5 virtual table for full-text search
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                summary,
                tags_json,
                domains_json,
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Create triggers to keep FTS in sync
        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, summary, tags_json, domains_json)
                VALUES (new.rowid, new.content, new.summary, new.tags_json, new.domains_json);
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags_json, domains_json)
                VALUES ('delete', old.rowid, old.content, old.summary, old.tags_json, old.domains_json);
            END
        """)

        self._conn.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags_json, domains_json)
                VALUES ('delete', old.rowid, old.content, old.summary, old.tags_json, old.domains_json);
                INSERT INTO memories_fts(rowid, content, summary, tags_json, domains_json)
                VALUES (new.rowid, new.content, new.summary, new.tags_json, new.domains_json);
            END
        """)

        # Create indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_truth_category ON memories(truth_category)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_is_active ON memories(is_active)")

        self._conn.commit()

    async def close(self) -> None:
        """Close SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def store(self, memory: MemoryEntry) -> MemoryEntry:
        """Store a memory entry."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        memory.tier = MemoryTier.PERSISTENT

        # Serialize embedding to bytes if present
        embedding_blob = None
        if memory.embedding:
            embedding_blob = json.dumps(memory.embedding).encode()

        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories (
                id, content, summary, content_hash,
                tier, truth_category, memory_type,
                confidence_overall, confidence_source_reliability, confidence_recency,
                corroboration_count, contradiction_count,
                embedding, embedding_model,
                tags_json, domains_json, entities_json,
                related_memories_json, contradicts_json, supports_json, supersedes,
                sources_json,
                created_at, updated_at, last_accessed, access_count,
                is_active, promotion_history_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(memory.id),
                memory.content,
                memory.summary,
                memory.content_hash,
                memory.tier.value,
                memory.truth_category.value,
                memory.memory_type.value,
                memory.confidence.overall,
                memory.confidence.source_reliability,
                memory.confidence.recency,
                memory.confidence.corroboration_count,
                memory.confidence.contradiction_count,
                embedding_blob,
                memory.embedding_model,
                json.dumps(memory.tags),
                json.dumps(memory.domains),
                json.dumps(memory.entities),
                json.dumps([str(u) for u in memory.related_memories]),
                json.dumps([str(u) for u in memory.contradicts]),
                json.dumps([str(u) for u in memory.supports]),
                str(memory.supersedes) if memory.supersedes else None,
                json.dumps([s.model_dump() for s in memory.sources], default=str),
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                1 if memory.is_active else 0,
                json.dumps(memory.promotion_history),
                json.dumps(memory.metadata),
            ),
        )
        self._conn.commit()
        return memory

    async def get(self, memory_id: UUID) -> MemoryEntry | None:
        """Retrieve a memory by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        cursor = self._conn.execute("SELECT * FROM memories WHERE id = ?", (str(memory_id),))
        row = cursor.fetchone()

        if row:
            memory = self._row_to_memory(row)
            memory.record_access()
            await self.update(memory)
            return memory
        return None

    async def update(self, memory: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        memory.updated_at = datetime.now()

        embedding_blob = None
        if memory.embedding:
            embedding_blob = json.dumps(memory.embedding).encode()

        self._conn.execute(
            """
            UPDATE memories SET
                content = ?, summary = ?, content_hash = ?,
                tier = ?, truth_category = ?, memory_type = ?,
                confidence_overall = ?, confidence_source_reliability = ?, confidence_recency = ?,
                corroboration_count = ?, contradiction_count = ?,
                embedding = ?, embedding_model = ?,
                tags_json = ?, domains_json = ?, entities_json = ?,
                related_memories_json = ?, contradicts_json = ?, supports_json = ?, supersedes = ?,
                sources_json = ?,
                updated_at = ?, last_accessed = ?, access_count = ?,
                is_active = ?, promotion_history_json = ?, metadata_json = ?
            WHERE id = ?
            """,
            (
                memory.content,
                memory.summary,
                memory.content_hash,
                memory.tier.value,
                memory.truth_category.value,
                memory.memory_type.value,
                memory.confidence.overall,
                memory.confidence.source_reliability,
                memory.confidence.recency,
                memory.confidence.corroboration_count,
                memory.confidence.contradiction_count,
                embedding_blob,
                memory.embedding_model,
                json.dumps(memory.tags),
                json.dumps(memory.domains),
                json.dumps(memory.entities),
                json.dumps([str(u) for u in memory.related_memories]),
                json.dumps([str(u) for u in memory.contradicts]),
                json.dumps([str(u) for u in memory.supports]),
                str(memory.supersedes) if memory.supersedes else None,
                json.dumps([s.model_dump() for s in memory.sources], default=str),
                memory.updated_at.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                1 if memory.is_active else 0,
                json.dumps(memory.promotion_history),
                json.dumps(memory.metadata),
                str(memory.id),
            ),
        )
        self._conn.commit()
        return memory

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (str(memory_id),))
        self._conn.commit()
        return cursor.rowcount > 0

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[MemoryEntry], list[float]]:
        """
        Vector similarity search.

        Note: SQLite doesn't have native vector search.
        This implementation loads all embeddings and computes similarity in Python.
        For production with large datasets, use sqlite-vec extension.
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        # Build query with filters
        query = "SELECT * FROM memories WHERE is_active = 1"
        params: list[Any] = []

        if filters:
            filter_sql, filter_params = self._build_filter_sql(filters)
            if filter_sql:
                query += f" AND {filter_sql}"
                params.extend(filter_params)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        # Compute similarities
        results: list[tuple[MemoryEntry, float]] = []
        for row in rows:
            memory = self._row_to_memory(row)
            if memory.embedding:
                score = self._cosine_similarity(embedding, memory.embedding)
                results.append((memory, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
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
        """Perform full-text search using FTS5."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        # Use FTS5 MATCH query
        fts_query = f'"{query_text}"'  # Exact phrase search

        query = """
            SELECT m.*, bm25(memories_fts) as score
            FROM memories m
            JOIN memories_fts ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ? AND m.is_active = 1
        """
        params: list[Any] = [fts_query]

        if filters:
            filter_sql, filter_params = self._build_filter_sql(filters)
            if filter_sql:
                query += f" AND {filter_sql}"
                params.extend(filter_params)

        query += " ORDER BY score LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        memories = [self._row_to_memory(row) for row in rows]
        # Normalize BM25 scores (lower is better, so invert)
        scores = [1.0 / (1.0 + abs(row["score"])) for row in rows]

        return memories, scores

    async def find_by_hash(self, content_hash: str) -> MemoryEntry | None:
        """Find a memory by its content hash."""
        if not self._conn:
            return None  # Not initialized yet

        try:
            cursor = self._conn.execute("SELECT * FROM memories WHERE content_hash = ?", (content_hash,))
            row = cursor.fetchone()

            if row:
                return self._row_to_memory(row)
        except Exception:
            pass
        return None

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count memories matching filters."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        query = "SELECT COUNT(*) FROM memories WHERE is_active = 1"
        params: list[Any] = []

        if filters:
            filter_sql, filter_params = self._build_filter_sql(filters)
            if filter_sql:
                query += f" AND {filter_sql}"
                params.extend(filter_params)

        cursor = self._conn.execute(query, params)
        return cursor.fetchone()[0]

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """List all memories with pagination."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        query = "SELECT * FROM memories WHERE is_active = 1"
        params: list[Any] = []

        if filters:
            filter_sql, filter_params = self._build_filter_sql(filters)
            if filter_sql:
                query += f" AND {filter_sql}"
                params.extend(filter_params)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    # Helper methods
    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert SQLite row to MemoryEntry."""
        # Parse embedding from blob
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"].decode())

        # Parse sources
        sources_data = json.loads(row["sources_json"] or "[]")
        sources = []
        for s in sources_data:
            if isinstance(s.get("timestamp"), str):
                s["timestamp"] = datetime.fromisoformat(s["timestamp"])
            s["source_type"] = SourceType(s["source_type"])
            sources.append(MemorySource(**s))

        # Parse UUIDs
        related = [UUID(u) for u in json.loads(row["related_memories_json"] or "[]")]
        contradicts = [UUID(u) for u in json.loads(row["contradicts_json"] or "[]")]
        supports = [UUID(u) for u in json.loads(row["supports_json"] or "[]")]
        supersedes = UUID(row["supersedes"]) if row["supersedes"] else None

        return MemoryEntry(
            id=UUID(row["id"]),
            content=row["content"],
            summary=row["summary"],
            tier=MemoryTier(row["tier"]),
            truth_category=TruthCategory(row["truth_category"]),
            memory_type=MemoryType(row["memory_type"]),
            sources=sources,
            confidence=ConfidenceScore(
                overall=row["confidence_overall"] or 0.5,
                source_reliability=row["confidence_source_reliability"] or 0.5,
                recency=row["confidence_recency"] or 0.5,
                corroboration_count=row["corroboration_count"] or 0,
                contradiction_count=row["contradiction_count"] or 0,
            ),
            embedding=embedding,
            embedding_model=row["embedding_model"] or "",
            tags=json.loads(row["tags_json"] or "[]"),
            domains=json.loads(row["domains_json"] or "[]"),
            entities=json.loads(row["entities_json"] or "[]"),
            related_memories=related,
            contradicts=contradicts,
            supports=supports,
            supersedes=supersedes,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"] or 0,
            is_active=bool(row["is_active"]),
            promotion_history=json.loads(row["promotion_history_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _build_filter_sql(self, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        """Build SQL WHERE clause from filters."""
        clauses = []
        params: list[Any] = []

        for key, value in filters.items():
            if key == "truth_category" and isinstance(value, list):
                placeholders = ", ".join("?" * len(value))
                clauses.append(f"truth_category IN ({placeholders})")
                params.extend(value)
            elif key == "memory_type" and isinstance(value, list):
                placeholders = ", ".join("?" * len(value))
                clauses.append(f"memory_type IN ({placeholders})")
                params.extend(value)
            elif key == "confidence_gte":
                clauses.append("confidence_overall >= ?")
                params.append(value)
            elif key == "created_after":
                clauses.append("created_at >= ?")
                params.append(value.isoformat() if hasattr(value, "isoformat") else value)
            elif key == "created_before":
                clauses.append("created_at <= ?")
                params.append(value.isoformat() if hasattr(value, "isoformat") else value)

        return " AND ".join(clauses), params

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
