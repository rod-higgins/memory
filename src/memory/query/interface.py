"""
Unified query interface for memory retrieval.

This is the intelligence layer that analyzes queries and retrieves
relevant memories from all tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from memory.llm.embeddings import EmbeddingProvider, get_embedding_provider
from memory.schema.memory_entry import (
    MemoryEntry,
    MemoryTier,
    MemoryType,
    TruthCategory,
)
from memory.storage.manager import StorageManager


@dataclass
class QueryResult:
    """Result from a memory query."""

    memories: list[MemoryEntry]
    scores: list[float]
    query_text: str
    query_type: str
    execution_time_ms: float
    total_searched: int = 0

    def top(self, n: int = 5) -> list[MemoryEntry]:
        """Get top N memories."""
        return self.memories[:n]

    def by_type(self, memory_type: MemoryType) -> list[MemoryEntry]:
        """Filter by memory type."""
        return [m for m in self.memories if m.memory_type == memory_type]

    def by_tier(self, tier: MemoryTier) -> list[MemoryEntry]:
        """Filter by tier."""
        return [m for m in self.memories if m.tier == tier]


@dataclass
class QueryContext:
    """Context for shaping memory retrieval."""

    # What we're looking for
    intent: str | None = None  # "recall", "validate", "explore"
    domains: list[str] = field(default_factory=list)

    # Filters
    memory_types: list[MemoryType] | None = None
    truth_categories: list[TruthCategory] | None = None
    tiers: list[MemoryTier] | None = None
    min_confidence: float = 0.0

    # Recency
    since: datetime | None = None
    accessed_since: datetime | None = None

    # Relationships
    related_to: UUID | None = None
    exclude_ids: list[UUID] = field(default_factory=list)


class MemoryQuery:
    """
    Intelligent memory query interface.

    Provides semantic search, hybrid search, and specialized queries
    that understand the user's intent and retrieve relevant context.
    """

    def __init__(
        self,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._storage = storage
        self._embeddings = embedding_provider
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the query interface."""
        if self._initialized:
            return

        if not self._embeddings:
            try:
                self._embeddings = get_embedding_provider()
            except ImportError:
                pass  # Embeddings not available, will use keyword search

        self._initialized = True

    async def semantic_search(
        self,
        query: str,
        limit: int = 20,
        context: QueryContext | None = None,
    ) -> QueryResult:
        """
        Perform semantic search using embeddings.

        Finds memories that are conceptually similar to the query,
        even if they don't share exact keywords.
        """
        start = datetime.now()

        # Generate query embedding
        if not self._embeddings:
            # Fall back to keyword search if no embeddings
            return await self.keyword_search(query, limit, context)

        embedding = await self._embeddings.embed(query)

        # Build filters
        filters = self._build_filters(context)

        # Determine which tiers to search
        tiers = context.tiers if context else None

        # Execute search
        memories, scores = await self._storage.vector_search(
            embedding=embedding,
            limit=limit,
            tiers=tiers,
            filters=filters,
        )

        # Filter by exclusions
        if context and context.exclude_ids:
            filtered = [(m, s) for m, s in zip(memories, scores) if m.id not in context.exclude_ids]
            memories = [m for m, s in filtered]
            scores = [s for m, s in filtered]

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return QueryResult(
            memories=memories,
            scores=scores,
            query_text=query,
            query_type="semantic",
            execution_time_ms=elapsed,
        )

    async def keyword_search(
        self,
        query: str,
        limit: int = 20,
        context: QueryContext | None = None,
    ) -> QueryResult:
        """
        Perform keyword/text search.

        Finds memories containing the exact query terms.
        """
        start = datetime.now()

        # Build filters
        filters = self._build_filters(context)

        # Determine which tiers to search
        tiers = context.tiers if context else None

        # Execute search
        memories, scores = await self._storage.text_search(
            query_text=query,
            limit=limit,
            tiers=tiers,
            filters=filters,
        )

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return QueryResult(
            memories=memories,
            scores=scores,
            query_text=query,
            query_type="keyword",
            execution_time_ms=elapsed,
        )

    async def hybrid_search(
        self,
        query: str,
        limit: int = 20,
        semantic_weight: float = 0.7,
        context: QueryContext | None = None,
    ) -> QueryResult:
        """
        Perform hybrid search combining semantic and keyword.

        Best for most queries - catches both conceptual and exact matches.
        """
        start = datetime.now()

        # Run both searches
        semantic_result = await self.semantic_search(query, limit=limit * 2, context=context)
        keyword_result = await self.keyword_search(query, limit=limit * 2, context=context)

        # Combine and score
        memory_scores: dict[UUID, tuple[MemoryEntry, float]] = {}

        for mem, score in zip(semantic_result.memories, semantic_result.scores):
            memory_scores[mem.id] = (mem, score * semantic_weight)

        keyword_weight = 1.0 - semantic_weight
        for mem, score in zip(keyword_result.memories, keyword_result.scores):
            if mem.id in memory_scores:
                existing_mem, existing_score = memory_scores[mem.id]
                memory_scores[mem.id] = (existing_mem, existing_score + score * keyword_weight)
            else:
                memory_scores[mem.id] = (mem, score * keyword_weight)

        # Sort by combined score
        sorted_items = sorted(
            memory_scores.values(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        memories = [m for m, s in sorted_items]
        scores = [s for m, s in sorted_items]

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return QueryResult(
            memories=memories,
            scores=scores,
            query_text=query,
            query_type="hybrid",
            execution_time_ms=elapsed,
        )

    async def recall(
        self,
        topic: str,
        limit: int = 10,
    ) -> QueryResult:
        """
        Recall memories about a topic.

        Optimized for retrieving known information - prefers
        high-confidence, validated memories.
        """
        context = QueryContext(
            intent="recall",
            min_confidence=0.5,
            tiers=[MemoryTier.PERSISTENT, MemoryTier.LONG_TERM],
        )
        return await self.hybrid_search(topic, limit=limit, context=context)

    async def get_preferences(
        self,
        domain: str | None = None,
        limit: int = 20,
    ) -> QueryResult:
        """
        Get user preferences, optionally filtered by domain.
        """
        context = QueryContext(
            intent="recall",
            memory_types=[MemoryType.PREFERENCE],
            domains=[domain] if domain else [],
        )

        query = f"{domain} preferences" if domain else "preferences"
        return await self.hybrid_search(query, limit=limit, context=context)

    async def get_beliefs(
        self,
        domain: str | None = None,
        limit: int = 20,
    ) -> QueryResult:
        """
        Get user beliefs, optionally filtered by domain.
        """
        context = QueryContext(
            intent="recall",
            memory_types=[MemoryType.BELIEF],
            domains=[domain] if domain else [],
        )

        query = f"{domain} beliefs" if domain else "beliefs"
        return await self.hybrid_search(query, limit=limit, context=context)

    async def get_skills(
        self,
        domain: str | None = None,
        limit: int = 20,
    ) -> QueryResult:
        """
        Get user skills and expertise.
        """
        context = QueryContext(
            intent="recall",
            memory_types=[MemoryType.SKILL],
            domains=[domain] if domain else [],
        )

        query = f"{domain} skills expertise" if domain else "skills expertise"
        return await self.hybrid_search(query, limit=limit, context=context)

    async def get_facts(
        self,
        topic: str,
        include_contextual: bool = True,
        limit: int = 20,
    ) -> QueryResult:
        """
        Get verified facts about a topic.
        """
        categories = [TruthCategory.ABSOLUTE]
        if include_contextual:
            categories.append(TruthCategory.CONTEXTUAL)

        context = QueryContext(
            intent="recall",
            memory_types=[MemoryType.FACT],
            truth_categories=categories,
            min_confidence=0.6,
        )

        return await self.hybrid_search(topic, limit=limit, context=context)

    async def get_recent(
        self,
        days: int = 7,
        limit: int = 20,
    ) -> QueryResult:
        """
        Get recently accessed or created memories.
        """
        since = datetime.now() - timedelta(days=days)
        QueryContext(
            accessed_since=since,
        )

        # For recent, we just list - no query needed
        start = datetime.now()

        memories = await self._storage.short_term.list_all(limit=limit)

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return QueryResult(
            memories=memories,
            scores=[1.0] * len(memories),
            query_text=f"recent {days} days",
            query_type="recent",
            execution_time_ms=elapsed,
        )

    async def get_by_domain(
        self,
        domain: str,
        limit: int = 20,
    ) -> QueryResult:
        """
        Get all memories in a specific domain.
        """
        context = QueryContext(
            domains=[domain],
        )

        return await self.hybrid_search(domain, limit=limit, context=context)

    async def find_contradictions(
        self,
        memory: MemoryEntry,
        limit: int = 5,
    ) -> QueryResult:
        """
        Find memories that might contradict a given memory.
        """
        # Search for related content
        result = await self.semantic_search(
            memory.content,
            limit=limit * 3,
            context=QueryContext(exclude_ids=[memory.id]),
        )

        # Filter to those in same domains
        potential = [
            (m, s) for m, s in zip(result.memories, result.scores) if any(d in memory.domains for d in m.domains)
        ][:limit]

        return QueryResult(
            memories=[m for m, s in potential],
            scores=[s for m, s in potential],
            query_text=f"contradictions for {memory.id}",
            query_type="contradiction",
            execution_time_ms=result.execution_time_ms,
        )

    async def get_context_for_query(
        self,
        user_query: str,
        max_memories: int = 10,
        max_tokens: int = 2000,
    ) -> QueryResult:
        """
        Get optimized context for an LLM query.

        This is the main entry point for memory augmentation.
        Returns memories most relevant to the user's query,
        optimized for token budget.

        Prioritizes:
        - User-authored content (sent emails, created documents)
        - High-confidence memories
        - Memories with meaningful summaries
        """
        # Hybrid search for maximum coverage
        result = await self.hybrid_search(
            user_query,
            limit=max_memories * 3,  # Get extra to filter and prioritize
        )

        # Score and prioritize memories
        scored_memories: list[tuple[MemoryEntry, float]] = []

        for mem, base_score in zip(result.memories, result.scores):
            adjusted_score = base_score

            # Boost user-authored content (SENT emails, created documents)
            if "sent" in mem.tags or "sent_email" in mem.tags:
                adjusted_score *= 1.5
            if "starred" in mem.tags or "important" in mem.tags:
                adjusted_score *= 1.3

            # Boost high-confidence memories
            if mem.confidence and hasattr(mem.confidence, "overall"):
                adjusted_score *= 0.7 + mem.confidence.overall * 0.3

            # Boost memories with summaries (indicates processed content)
            if mem.summary and len(mem.summary) > 20:
                adjusted_score *= 1.2

            # Penalize very long raw content (likely unprocessed)
            if len(mem.content) > 3000 and not mem.summary:
                adjusted_score *= 0.7

            # Boost business/work content
            if any(d in ["business", "work", "project"] for d in mem.domains):
                adjusted_score *= 1.1

            scored_memories.append((mem, adjusted_score))

        # Sort by adjusted score
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Select within token budget
        selected: list[tuple[MemoryEntry, float]] = []
        token_count = 0

        for mem, score in scored_memories:
            # Use summary if available, otherwise truncate content
            content_for_tokens = mem.summary if mem.summary else mem.content[:500]
            mem_tokens = len(content_for_tokens) // 4

            if token_count + mem_tokens > max_tokens:
                if not selected:  # Always include at least one
                    selected.append((mem, score))
                break

            selected.append((mem, score))
            token_count += mem_tokens

            if len(selected) >= max_memories:
                break

        return QueryResult(
            memories=[m for m, s in selected],
            scores=[s for m, s in selected],
            query_text=user_query,
            query_type="context",
            execution_time_ms=result.execution_time_ms,
        )

    def _build_filters(self, context: QueryContext | None) -> dict[str, Any] | None:
        """Build storage filters from query context."""
        if not context:
            return None

        filters: dict[str, Any] = {}

        if context.memory_types:
            filters["memory_type"] = [t.value for t in context.memory_types]

        if context.truth_categories:
            filters["truth_category"] = [t.value for t in context.truth_categories]

        if context.min_confidence > 0:
            filters["confidence_gte"] = context.min_confidence

        return filters if filters else None
