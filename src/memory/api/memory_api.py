"""
High-level Memory API.

This is the main entry point for using the memory system.
It provides a simple, unified interface for all memory operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from memory.export.formats import ExportFormat, MemoryExporter
from memory.llm.augmentation import (
    AugmentedPrompt,
    MemoryAugmenter,
)
from memory.llm.embeddings import get_embedding_provider
from memory.llm.slm import get_slm_provider
from memory.query.interface import MemoryQuery, QueryResult
from memory.schema.memory_entry import MemoryEntry, MemoryTier, MemoryType, TruthCategory
from memory.storage.manager import StorageManager


class MemoryAPI:
    """
    Unified API for the Personal Memory System.

    This class provides all the functionality needed to:
    - Store and retrieve memories
    - Search across all memory tiers
    - Get context for LLM prompts
    - Export memories in various formats

    Usage:
        api = MemoryAPI()
        await api.initialize()

        # Store a memory
        await api.remember("I prefer Python for data processing")

        # Search memories
        results = await api.search("programming preferences")

        # Get context for an LLM
        context = await api.get_context("How should I structure this Python project?")

        # Augment a prompt
        augmented = await api.augment_prompt("Help me with Drupal")
    """

    def __init__(
        self,
        base_path: str | Path = "~/memory/data",
        use_local_embeddings: bool = True,
        use_local_slm: bool = True,
    ):
        self._base_path = Path(base_path).expanduser()
        self._use_local_embeddings = use_local_embeddings
        self._use_local_slm = use_local_slm

        self._storage: StorageManager | None = None
        self._query: MemoryQuery | None = None
        self._augmenter: MemoryAugmenter | None = None
        self._exporter = MemoryExporter()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        # Initialize storage
        self._storage = StorageManager(base_path=self._base_path)
        await self._storage.initialize()

        # Initialize embeddings
        embeddings = None
        if self._use_local_embeddings:
            try:
                embeddings = get_embedding_provider()
            except ImportError:
                pass  # Embeddings not available

        # Initialize SLM - prefer Ollama for intelligent synthesis
        slm = None
        if self._use_local_slm:
            # Try Ollama first with available models
            try:
                ollama_slm = get_slm_provider(provider="ollama", model="tinyllama")
                # Check if Ollama is running
                if hasattr(ollama_slm, "is_available"):
                    is_avail = await ollama_slm.is_available()
                    if is_avail:
                        slm = ollama_slm
            except Exception:
                pass

            # Fall back to mock if Ollama unavailable
            if slm is None:
                slm = get_slm_provider(provider="mock")

        # Initialize query interface
        self._query = MemoryQuery(self._storage, embeddings)
        await self._query.initialize()

        # Initialize augmenter
        self._augmenter = MemoryAugmenter(self._storage, embeddings, slm)
        await self._augmenter.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Close all connections."""
        if self._storage:
            await self._storage.close()
        self._initialized = False

    # === Storage Operations ===

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        truth_category: TruthCategory = TruthCategory.INFERRED,
        domains: list[str] | None = None,
        tags: list[str] | None = None,
        tier: MemoryTier | None = None,
    ) -> MemoryEntry:
        """
        Store a new memory.

        Args:
            content: The memory content
            memory_type: Type of memory (FACT, BELIEF, PREFERENCE, etc.)
            truth_category: Truth classification
            domains: Knowledge domains
            tags: Optional tags
            tier: Memory tier (SHORT_TERM, LONG_TERM, PERSISTENT). Defaults to PERSISTENT.

        Returns:
            The stored MemoryEntry
        """
        if not self._storage:
            raise RuntimeError("API not initialized")

        # Default to PERSISTENT for durability (short-term is ephemeral/in-memory only)
        if tier is None:
            tier = MemoryTier.PERSISTENT

        memory = MemoryEntry(
            content=content,
            memory_type=memory_type,
            truth_category=truth_category,
            domains=domains or [],
            tags=tags or [],
            tier=tier,
        )

        return await self._storage.store(memory)

    async def get(self, memory_id: str | UUID) -> MemoryEntry | None:
        """Get a memory by ID."""
        if not self._storage:
            raise RuntimeError("API not initialized")

        if isinstance(memory_id, str):
            memory_id = UUID(memory_id)

        return await self._storage.get(memory_id)

    async def forget(self, memory_id: str | UUID) -> bool:
        """Delete a memory."""
        if not self._storage:
            raise RuntimeError("API not initialized")

        if isinstance(memory_id, str):
            memory_id = UUID(memory_id)

        return await self._storage.delete(memory_id)

    # === Search Operations ===

    async def search(
        self,
        query: str,
        limit: int = 10,
        semantic: bool = True,
    ) -> QueryResult:
        """
        Search memories.

        Args:
            query: Search query
            limit: Maximum results
            semantic: Use semantic (vector) search if True

        Returns:
            QueryResult with matching memories
        """
        if not self._query:
            raise RuntimeError("API not initialized")

        try:
            if semantic:
                return await self._query.hybrid_search(query, limit=limit)
            else:
                return await self._query.keyword_search(query, limit=limit)
        except (ImportError, RuntimeError):
            # Fall back to keyword search if embeddings unavailable
            return await self._query.keyword_search(query, limit=limit)

    async def recall(self, topic: str, limit: int = 5) -> QueryResult:
        """Recall validated memories about a topic."""
        if not self._query:
            raise RuntimeError("API not initialized")

        return await self._query.recall(topic, limit=limit)

    async def get_preferences(self, domain: str | None = None) -> QueryResult:
        """Get user preferences."""
        if not self._query:
            raise RuntimeError("API not initialized")

        return await self._query.get_preferences(domain)

    async def get_beliefs(self, domain: str | None = None) -> QueryResult:
        """Get user beliefs."""
        if not self._query:
            raise RuntimeError("API not initialized")

        return await self._query.get_beliefs(domain)

    async def get_skills(self, domain: str | None = None) -> QueryResult:
        """Get user skills and expertise."""
        if not self._query:
            raise RuntimeError("API not initialized")

        return await self._query.get_skills(domain)

    # === LLM Integration ===

    async def get_context(
        self,
        query: str,
        format: str = "claude",
        max_memories: int = 10,
    ) -> str:
        """
        Get PLM context for LLM injection.

        This queries the SYNTHESIZED knowledge store, not raw memories.
        The knowledge store contains distilled insights, skills, and facts
        extracted from the user's digital life.

        Args:
            query: The user's query to match against
            format: Output format (claude, xml, json, markdown, system_prompt, compact)
            max_memories: Maximum knowledge entries to include

        Returns:
            Formatted context string ready for injection
        """
        # Try to get context from synthesized knowledge store first
        try:
            from memory.knowledge.store import KnowledgeStore

            knowledge_store = KnowledgeStore()
            await knowledge_store.initialize()

            # Semantic search on knowledge
            results = await knowledge_store.search(query, limit=max_memories)

            if results:
                # Build context from knowledge entries
                context_parts = []
                for entry, score in results:
                    if score > 0.3:  # Only include relevant matches
                        context_parts.append(f"- {entry.summary}")
                        if entry.entities:
                            context_parts.append(f"  Related: {', '.join(entry.entities[:5])}")

                if context_parts:
                    knowledge_context = "\n".join(context_parts)
                    await knowledge_store.close()

                    # Format based on requested format
                    if format == "claude":
                        return f"""<user_profile>
Based on the user's professional background and digital history:

{knowledge_context}
</user_profile>"""
                    else:
                        return knowledge_context

            await knowledge_store.close()
        except Exception:
            pass  # Fall back to augmenter

        # Fall back to memory augmenter if knowledge store empty/unavailable
        if not self._augmenter:
            raise RuntimeError("API not initialized")

        export_format = ExportFormat(format)
        return await self._augmenter.get_context_injection(
            query,
            format=export_format,
            max_memories=max_memories,
        )

    async def augment_prompt(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> AugmentedPrompt:
        """
        Augment a prompt with memory context.

        Returns full augmentation info including which memories were used.

        Args:
            user_prompt: The user's message
            system_prompt: Optional existing system prompt

        Returns:
            AugmentedPrompt with all details
        """
        if not self._augmenter:
            raise RuntimeError("API not initialized")

        return await self._augmenter.augment_prompt(user_prompt, system_prompt)

    # === Export ===

    async def export_all(
        self,
        format: str = "json",
        tier: MemoryTier | None = None,
    ) -> str:
        """Export all memories."""
        if not self._storage:
            raise RuntimeError("API not initialized")

        if tier:
            memories = await self._storage.get_store(tier).list_all(limit=10000)
        else:
            all_memories: list[MemoryEntry] = []
            for t in MemoryTier:
                memories_in_tier = await self._storage.get_store(t).list_all(limit=10000)
                all_memories.extend(memories_in_tier)
            memories = all_memories

        export_format = ExportFormat(format)
        return self._exporter.export(memories, format=export_format)

    # === Ingestion ===

    async def ingest(
        self,
        source: str,
        path: str | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        """
        Ingest data from a source.

        Args:
            source: Source identifier (e.g., 'gmail', 'chrome_history')
            path: Optional path to source data
            **options: Source-specific options

        Returns:
            Dict with ingestion results
        """
        from memory.ingestion.sources.registry import SourceRegistry

        registry = SourceRegistry()

        # Find the source adapter
        source_adapter = registry.get_source(source)

        if source_adapter is None:
            # Try to use a generic adapter based on source type
            source_info = None
            for s in registry.list_sources():
                if s.get("id") == source:
                    source_info = s
                    break

            if not source_info:
                raise ValueError(f"Unknown source: {source}")

            # For now, return a placeholder result
            # Real implementation would use proper adapters
            return {
                "source": source,
                "count": 0,
                "message": f"Source '{source}' requires configuration. Please set up a connection first.",
            }

        # Run ingestion
        try:
            limit = options.get("limit", 100)
            memories = await source_adapter.ingest(path=path, limit=limit)

            # Store the memories
            count = 0
            for memory in memories:
                await self.remember(
                    memory.content,
                    memory_type=memory.memory_type,
                    domains=memory.domains,
                    tags=memory.tags,
                )
                count += 1

            return {
                "source": source,
                "count": count,
                "message": f"Ingested {count} memories from {source}",
            }
        except Exception as e:
            return {
                "source": source,
                "count": 0,
                "error": str(e),
            }

    # === Stats ===

    async def stats(self) -> dict[str, Any]:
        """Get system statistics."""
        if not self._storage:
            raise RuntimeError("API not initialized")

        return await self._storage.get_stats()

    async def get_stats(self) -> dict[str, Any]:
        """Get formatted statistics for the web UI."""
        if not self._storage:
            raise RuntimeError("API not initialized")

        # Get all memories for detailed stats
        all_memories: list[MemoryEntry] = []
        for tier in MemoryTier:
            try:
                store = self._storage.get_store(tier)
                memories = await store.list_all(limit=10000)
                all_memories.extend(memories)
            except Exception:
                pass

        # Calculate by_tier from actual tier field (not store location)
        by_tier: dict[str, int] = {"short_term": 0, "long_term": 0, "persistent": 0}
        for m in all_memories:
            tier_val = m.tier.value if hasattr(m.tier, "value") else str(m.tier)
            by_tier[tier_val] = by_tier.get(tier_val, 0) + 1

        # Calculate by_type
        by_type: dict[str, int] = {}
        for m in all_memories:
            t = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
            by_type[t] = by_type.get(t, 0) + 1

        # Calculate by_truth_category
        by_truth: dict[str, int] = {}
        for m in all_memories:
            t = m.truth_category.value if hasattr(m.truth_category, "value") else str(m.truth_category)
            by_truth[t] = by_truth.get(t, 0) + 1

        # Calculate top domains
        domain_counts: dict[str, int] = {}
        for m in all_memories:
            for d in m.domains:
                domain_counts[d] = domain_counts.get(d, 0) + 1
        top_domains = sorted(
            [{"domain": k, "count": v} for k, v in domain_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:10]

        # Recent activity (last 10 memories)
        sorted_memories = sorted(all_memories, key=lambda x: x.created_at, reverse=True)
        recent = [
            {
                "id": str(m.id),
                "content": m.content[:100],
                "type": m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in sorted_memories[:10]
        ]

        return {
            "total_memories": len(all_memories),
            "by_tier": by_tier,
            "by_type": by_type,
            "by_truth_category": by_truth,
            "top_domains": top_domains,
            "recent_activity": recent,
        }

    async def list_memories(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
        return_total: bool = False,
    ) -> list[MemoryEntry] | tuple[list[MemoryEntry], int]:
        """List memories with optional filtering.

        If return_total is True, returns (memories, total_count) tuple.
        """
        if not self._storage:
            raise RuntimeError("API not initialized")

        all_memories: list[MemoryEntry] = []

        # Get from all tiers or filter by tier
        tiers_to_check = [MemoryTier.PERSISTENT, MemoryTier.LONG_TERM, MemoryTier.SHORT_TERM]
        if filters and "tier" in filters:
            tier_value = filters["tier"]
            try:
                tiers_to_check = [MemoryTier(tier_value)]
            except ValueError:
                pass

        for tier in tiers_to_check:
            try:
                store = self._storage.get_store(tier)
                memories = await store.list_all(limit=10000)
                all_memories.extend(memories)
            except Exception:
                pass

        # Apply filters
        if filters:
            if "memory_type" in filters:
                mt = filters["memory_type"]
                all_memories = [m for m in all_memories if m.memory_type.value == mt]
            if "domain" in filters:
                domain = filters["domain"]
                all_memories = [m for m in all_memories if domain in (m.domains or [])]

        # Sort by created_at descending
        all_memories.sort(key=lambda x: x.created_at, reverse=True)

        # Get total before pagination
        total_count = len(all_memories)

        # Apply pagination
        paginated = all_memories[offset : offset + limit]

        if return_total:
            return paginated, total_count
        return paginated

    async def get_memory(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        return await self.get(memory_id)

    async def get_identity(self) -> dict[str, Any]:
        """Get user identity profile."""
        from ..schema.identity import IdentityProfile

        # Try to load from config file
        config_path = self._base_path / "persistent" / "data_sources.json"
        if config_path.exists():
            try:
                profile = IdentityProfile.from_config_file(str(config_path))
                return profile.model_dump()
            except Exception:
                pass

        # Return empty profile
        return IdentityProfile.create_default().model_dump()

    async def update_identity(self, data: dict[str, Any]) -> dict[str, Any]:
        """Update user identity profile."""
        # This would update the config file
        # For now, just return the data
        return data

    # === Learning ===

    async def learn_from_interaction(
        self,
        user_message: str,
        assistant_response: str,
    ) -> list[MemoryEntry]:
        """
        Learn from an interaction and create memories.

        Call this after each LLM interaction to build memory.
        """
        if not self._augmenter:
            raise RuntimeError("API not initialized")

        return await self._augmenter.create_memory_from_interaction(
            user_message,
            assistant_response,
        )

    async def give_feedback(
        self,
        memory_id: str,
        feedback: str,  # "correct", "incorrect", "outdated"
    ) -> MemoryEntry | None:
        """
        Provide feedback on a memory.

        This helps the system learn and improve accuracy.
        """
        if not self._augmenter:
            raise RuntimeError("API not initialized")

        return await self._augmenter.learn_from_feedback(memory_id, feedback)


# Convenience functions for quick usage


async def get_memory_context(
    query: str,
    format: str = "claude",
    base_path: str = "~/memory/data",
) -> str:
    """
    Quick function to get memory context for a query.

    Usage:
        context = await get_memory_context("How should I structure my Drupal project?")
        # Use context in your LLM prompt
    """
    api = MemoryAPI(base_path=base_path)
    await api.initialize()
    try:
        return await api.get_context(query, format=format)
    finally:
        await api.close()


async def remember(
    content: str,
    memory_type: str = "fact",
    base_path: str = "~/memory/data",
) -> MemoryEntry:
    """
    Quick function to store a memory.

    Usage:
        await remember("I prefer tabs over spaces", memory_type="preference")
    """
    api = MemoryAPI(base_path=base_path)
    await api.initialize()
    try:
        return await api.remember(content, MemoryType(memory_type))
    finally:
        await api.close()
