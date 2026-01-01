"""
Smart Knowledge Store for PLM.

This module stores ONLY:
1. Embeddings for semantic search
2. Distilled summaries (not raw content)
3. Categorized facts and insights
4. Relationships between knowledge

NOT stored:
- Raw emails/documents
- Full text content
- Redundant data
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx


class KnowledgeType(str, Enum):
    """Types of knowledge entries."""

    FACT = "fact"  # Verified factual information
    SKILL = "skill"  # Known capability or expertise
    PREFERENCE = "preference"  # User preference
    EXPERIENCE = "experience"  # Work/life experience
    CONTACT = "contact"  # Person/organization
    PROJECT = "project"  # Project or work item
    INSIGHT = "insight"  # Derived insight


@dataclass
class KnowledgeEntry:
    """A distilled knowledge entry."""

    id: UUID = field(default_factory=uuid4)
    knowledge_type: KnowledgeType = KnowledgeType.FACT

    # Distilled content (NOT raw data)
    summary: str = ""  # Max 500 chars
    category: str = ""  # e.g., "technology", "business", "personal"
    subcategory: str = ""  # e.g., "drupal", "aws", "contracts"

    # Structured data
    entities: list[str] = field(default_factory=list)  # Named entities
    tags: list[str] = field(default_factory=list)

    # Semantic search
    embedding: list[float] = field(default_factory=list)

    # Metadata
    confidence: float = 0.8
    source_count: int = 1  # How many sources support this
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)


class OllamaEmbeddings:
    """Generate embeddings using Ollama."""

    def __init__(self, model: str = "tinyllama"):
        self.model = model
        self.base_url = "http://localhost:11434"

    async def embed(self, text: str, retries: int = 3) -> list[float]:
        """Generate embedding for text with retry logic."""
        import asyncio

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text[:1000]},
                    )
                    response.raise_for_status()
                    return response.json()["embedding"]
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue
                raise e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            emb = await self.embed(text)
            embeddings.append(emb)
        return embeddings


class KnowledgeStore:
    """
    Smart storage for distilled knowledge.

    Stores embeddings and summaries, NOT raw content.
    """

    def __init__(self, db_path: str = "~/memory/data/knowledge.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = OllamaEmbeddings()
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the knowledge store."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        # Create optimized schema
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                knowledge_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                category TEXT,
                subcategory TEXT,
                entities_json TEXT,
                tags_json TEXT,
                embedding BLOB,
                confidence REAL DEFAULT 0.8,
                source_count INTEGER DEFAULT 1,
                created_at TEXT,
                last_accessed TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(knowledge_type);
            CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category);

            -- Virtual table for full-text search on summaries
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                id,
                summary,
                category,
                tags,
                content='knowledge',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(id, summary, category, tags)
                VALUES (new.id, new.summary, new.category, new.tags_json);
            END;
        """)
        self._conn.commit()

    async def close(self) -> None:
        """Close the connection."""
        if self._conn:
            self._conn.close()

    async def store(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Store a knowledge entry."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        # Generate embedding if not present
        if not entry.embedding:
            entry.embedding = await self.embeddings.embed(entry.summary)

        self._conn.execute(
            """
            INSERT OR REPLACE INTO knowledge
            (id, knowledge_type, summary, category, subcategory,
             entities_json, tags_json, embedding, confidence,
             source_count, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(entry.id),
                entry.knowledge_type.value,
                entry.summary,
                entry.category,
                entry.subcategory,
                json.dumps(entry.entities),
                json.dumps(entry.tags),
                json.dumps(entry.embedding),
                entry.confidence,
                entry.source_count,
                entry.created_at.isoformat(),
                entry.last_accessed.isoformat(),
            ),
        )
        self._conn.commit()
        return entry

    async def search(
        self,
        query: str,
        limit: int = 10,
        knowledge_type: KnowledgeType | None = None,
    ) -> list[tuple[KnowledgeEntry, float]]:
        """
        Semantic search using embeddings.
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        # Generate query embedding
        query_embedding = await self.embeddings.embed(query)

        # Get all entries (for now - would use vector index in production)
        sql = "SELECT * FROM knowledge"
        params = []

        if knowledge_type:
            sql += " WHERE knowledge_type = ?"
            params.append(knowledge_type.value)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        # Calculate similarity scores
        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            if entry.embedding:
                score = self._cosine_similarity(query_embedding, entry.embedding)
                results.append((entry, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def text_search(self, query: str, limit: int = 10) -> list[KnowledgeEntry]:
        """Full-text search on summaries."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        cursor = self._conn.execute(
            """
            SELECT k.* FROM knowledge k
            JOIN knowledge_fts fts ON k.id = fts.id
            WHERE knowledge_fts MATCH ?
            LIMIT ?
        """,
            (query, limit),
        )

        return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def get_by_category(
        self,
        category: str,
        subcategory: str | None = None,
    ) -> list[KnowledgeEntry]:
        """Get knowledge by category."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        sql = "SELECT * FROM knowledge WHERE category = ?"
        params = [category]

        if subcategory:
            sql += " AND subcategory = ?"
            params.append(subcategory)

        cursor = self._conn.execute(sql, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        if not self._conn:
            return {}

        cursor = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT category) as categories,
                AVG(length(summary)) as avg_summary_length
            FROM knowledge
        """)
        row = cursor.fetchone()

        # Count by type
        by_type = {}
        cursor = self._conn.execute("""
            SELECT knowledge_type, COUNT(*) as count
            FROM knowledge
            GROUP BY knowledge_type
        """)
        for r in cursor.fetchall():
            by_type[r["knowledge_type"]] = r["count"]

        return {
            "total_entries": row["total"],
            "categories": row["categories"],
            "avg_summary_length": row["avg_summary_length"],
            "by_type": by_type,
        }

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        """Convert database row to KnowledgeEntry."""
        return KnowledgeEntry(
            id=UUID(row["id"]),
            knowledge_type=KnowledgeType(row["knowledge_type"]),
            summary=row["summary"],
            category=row["category"] or "",
            subcategory=row["subcategory"] or "",
            entities=json.loads(row["entities_json"] or "[]"),
            tags=json.loads(row["tags_json"] or "[]"),
            embedding=json.loads(row["embedding"]) if row["embedding"] else [],
            confidence=row["confidence"] or 0.8,
            source_count=row["source_count"] or 1,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else datetime.now(),
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
