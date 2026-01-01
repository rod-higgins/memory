"""
Enrichment pipeline for raw memories.

Adds embeddings, entity extraction, truth categorization, and summaries.
"""

from __future__ import annotations

from memory.llm.embeddings import EmbeddingProvider, SentenceTransformerProvider
from memory.llm.slm import MockSLM, SLMProvider
from memory.schema.memory_entry import ConfidenceScore, MemoryEntry


class EnrichmentPipeline:
    """
    Enriches raw memories with additional context and embeddings.

    Features:
    - Embedding generation (local)
    - Truth categorization (via SLM)
    - Entity extraction (via SLM)
    - Domain classification (via SLM)
    - Summary generation (via SLM)
    - Confidence scoring
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        slm_provider: SLMProvider | None = None,
        use_slm: bool = True,
    ):
        self._embedder = embedding_provider or SentenceTransformerProvider()
        self._slm = slm_provider or MockSLM()
        self._use_slm = use_slm

    async def enrich(
        self,
        memory: MemoryEntry,
        generate_embedding: bool = True,
        use_slm_categorization: bool = True,
        generate_summary: bool = True,
    ) -> MemoryEntry:
        """
        Apply all enrichment steps to a memory.

        Args:
            memory: Raw memory entry
            generate_embedding: Whether to generate embeddings
            use_slm_categorization: Whether to use SLM for categorization
            generate_summary: Whether to generate summary for long content

        Returns:
            Enriched memory entry
        """
        # 1. Generate embedding
        if generate_embedding:
            memory.embedding = await self._embedder.embed(memory.content)
            memory.embedding_model = self._embedder.model_name

        # 2. Categorize truth type (via SLM if enabled)
        if use_slm_categorization and self._use_slm:
            memory.truth_category = await self._slm.categorize_truth(memory.content)
            memory.memory_type = await self._slm.classify_memory_type(memory.content)

        # 3. Extract entities (via SLM)
        if self._use_slm:
            entities = await self._slm.extract_entities(memory.content)
            memory.entities = list(set(memory.entities + entities))

        # 4. Extract/enhance domains (via SLM)
        if self._use_slm:
            domains = await self._slm.extract_domains(memory.content)
            memory.domains = list(set(memory.domains + domains))

        # 5. Generate summary for long content
        if generate_summary and len(memory.content) > 500 and self._use_slm:
            memory.summary = await self._slm.generate_summary(memory.content)

        # 6. Compute initial confidence score
        if memory.sources:
            memory.confidence = ConfidenceScore.from_source_type(
                memory.sources[0].source_type,
                memory.created_at,
            )
        else:
            memory.confidence = ConfidenceScore()
            memory.confidence.compute_overall()

        # 7. Auto-tag based on content
        memory.tags = list(set(memory.tags + self._auto_tag(memory)))

        return memory

    async def enrich_batch(
        self,
        memories: list[MemoryEntry],
        generate_embeddings: bool = True,
        use_slm_categorization: bool = True,
        batch_size: int = 32,
    ) -> list[MemoryEntry]:
        """
        Enrich multiple memories efficiently.

        Uses batch embedding for efficiency.
        """
        enriched = []

        # Batch embed for efficiency
        if generate_embeddings:
            contents = [m.content for m in memories]

            # Process in batches
            for i in range(0, len(contents), batch_size):
                batch_contents = contents[i : i + batch_size]
                batch_memories = memories[i : i + batch_size]

                embeddings = await self._embedder.embed_batch(batch_contents)

                for memory, embedding in zip(batch_memories, embeddings):
                    memory.embedding = embedding
                    memory.embedding_model = self._embedder.model_name

        # Enrich each memory (SLM calls are sequential for now)
        for memory in memories:
            await self.enrich(
                memory,
                generate_embedding=False,  # Already done above
                use_slm_categorization=use_slm_categorization,
            )
            enriched.append(memory)

        return enriched

    def _auto_tag(self, memory: MemoryEntry) -> list[str]:
        """Generate automatic tags based on memory content and metadata."""
        tags = []

        # Tag based on source
        if memory.sources:
            source_type = memory.sources[0].source_type
            tags.append(f"source:{source_type.value}")

        # Tag based on truth category
        tags.append(f"truth:{memory.truth_category.value}")

        # Tag based on memory type
        tags.append(f"type:{memory.memory_type.value}")

        # Tag based on confidence level
        if memory.confidence.overall >= 0.8:
            tags.append("high_confidence")
        elif memory.confidence.overall >= 0.5:
            tags.append("medium_confidence")
        else:
            tags.append("low_confidence")

        return tags


class SimpleEnrichmentPipeline:
    """
    Simplified enrichment without SLM (for initial testing).

    Uses heuristics instead of model inference.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._embedder = embedding_provider or SentenceTransformerProvider()

    async def enrich(self, memory: MemoryEntry) -> MemoryEntry:
        """Enrich with embeddings and heuristic classification."""
        # Generate embedding
        memory.embedding = await self._embedder.embed(memory.content)
        memory.embedding_model = self._embedder.model_name

        # Compute confidence
        if memory.sources:
            memory.confidence = ConfidenceScore.from_source_type(
                memory.sources[0].source_type,
                memory.created_at,
            )

        return memory

    async def enrich_batch(
        self,
        memories: list[MemoryEntry],
        batch_size: int = 32,
    ) -> list[MemoryEntry]:
        """Batch enrich memories."""
        contents = [m.content for m in memories]

        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i : i + batch_size]
            batch_memories = memories[i : i + batch_size]

            embeddings = await self._embedder.embed_batch(batch_contents)

            for memory, embedding in zip(batch_memories, embeddings):
                memory.embedding = embedding
                memory.embedding_model = self._embedder.model_name

                if memory.sources:
                    memory.confidence = ConfidenceScore.from_source_type(
                        memory.sources[0].source_type,
                        memory.created_at,
                    )

        return memories
