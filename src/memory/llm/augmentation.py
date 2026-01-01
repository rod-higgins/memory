"""
Memory-Augmented LLM Interface.

This is the core intelligence layer that bridges the memory system
with any LLM. It analyzes user queries, retrieves relevant memories,
and injects personalized context into prompts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memory.export.formats import ExportConfig, ExportFormat, MemoryExporter
from memory.llm.embeddings import EmbeddingProvider, get_embedding_provider
from memory.llm.slm import SLMProvider, get_slm_provider
from memory.query.interface import MemoryQuery, QueryContext
from memory.schema.memory_entry import MemoryEntry, MemoryType
from memory.storage.manager import StorageManager


@dataclass
class AugmentationConfig:
    """Configuration for memory augmentation."""

    # Memory retrieval
    max_memories: int = 15
    max_tokens: int = 3000
    min_confidence: float = 0.3
    min_relevance_score: float = 0.2

    # Format
    export_format: ExportFormat = ExportFormat.CLAUDE_FORMAT
    include_metadata: bool = True

    # Behavior
    always_include_preferences: bool = True
    always_include_identity: bool = True
    detect_domains: bool = True


@dataclass
class AugmentedPrompt:
    """Result of prompt augmentation."""

    original_prompt: str
    augmented_prompt: str
    memory_context: str
    memories_used: list[MemoryEntry]
    memory_count: int
    token_estimate: int
    domains_detected: list[str]
    processing_time_ms: float


@dataclass
class InteractionResult:
    """Result of a memory-tracked interaction."""

    response: str
    memories_referenced: list[MemoryEntry]
    new_memories_created: list[MemoryEntry]
    contradictions_detected: list[tuple[MemoryEntry, MemoryEntry]]


class MemoryAugmenter:
    """
    Core intelligence layer for memory-augmented LLM interactions.

    This class:
    1. Analyzes incoming queries to understand context and intent
    2. Retrieves relevant memories from all tiers
    3. Formats memories for injection into LLM prompts
    4. Tracks interactions for future memory creation
    """

    def __init__(
        self,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider | None = None,
        slm_provider: SLMProvider | None = None,
        config: AugmentationConfig | None = None,
    ):
        self._storage = storage
        self._embeddings = embedding_provider
        self._slm = slm_provider
        self._config = config or AugmentationConfig()
        self._query: MemoryQuery | None = None
        self._exporter = MemoryExporter()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        if not self._embeddings:
            try:
                self._embeddings = get_embedding_provider()
            except ImportError:
                pass  # Embeddings not available, will use keyword search

        # Initialize SLM - prefer Ollama for intelligent synthesis
        if not self._slm:
            try:
                ollama_slm = get_slm_provider(provider="ollama", model="tinyllama")
                if hasattr(ollama_slm, 'is_available'):
                    is_avail = await ollama_slm.is_available()
                    if is_avail:
                        self._slm = ollama_slm
            except Exception:
                pass

            # Fall back to mock if Ollama unavailable
            if not self._slm:
                self._slm = get_slm_provider(provider="mock")

        self._query = MemoryQuery(self._storage, self._embeddings)
        await self._query.initialize()

        self._initialized = True

    async def augment_prompt(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> AugmentedPrompt:
        """
        Augment a user prompt with relevant memories.

        This is the main entry point for memory injection.

        Args:
            user_prompt: The user's message/query
            system_prompt: Optional existing system prompt to extend
            additional_context: Extra context (e.g., current file, project)

        Returns:
            AugmentedPrompt with memory context injected
        """
        if not self._query:
            raise RuntimeError("Augmenter not initialized")

        start = datetime.now()
        detected_domains: list[str] = []

        # Detect domains from the query
        if self._config.detect_domains and self._slm:
            detected_domains = await self._slm.extract_domains(user_prompt)

        # Build query context
        QueryContext(
            domains=detected_domains,
            min_confidence=self._config.min_confidence,
        )

        # Retrieve relevant memories
        result = await self._query.get_context_for_query(
            user_prompt,
            max_memories=self._config.max_memories,
            max_tokens=self._config.max_tokens,
        )

        memories = result.memories
        scores = result.scores

        # Filter by relevance score
        filtered = [
            (m, s) for m, s in zip(memories, scores)
            if s >= self._config.min_relevance_score
        ]
        memories = [m for m, s in filtered]
        scores = [s for m, s in filtered]

        # Always include identity/preferences if configured
        if self._config.always_include_preferences:
            prefs = await self._query.get_preferences(limit=3)
            for mem in prefs.memories:
                if mem.id not in {m.id for m in memories}:
                    memories.append(mem)
                    scores.append(0.8)

        # Format for export
        export_config = ExportConfig(
            max_memories=self._config.max_memories,
            include_metadata=self._config.include_metadata,
        )

        memory_context = self._exporter.export(
            memories,
            format=self._config.export_format,
            scores=scores,
            config=export_config,
        )

        # Build augmented prompt
        augmented = self._build_augmented_prompt(
            user_prompt,
            system_prompt,
            memory_context,
            additional_context,
        )

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return AugmentedPrompt(
            original_prompt=user_prompt,
            augmented_prompt=augmented,
            memory_context=memory_context,
            memories_used=memories,
            memory_count=len(memories),
            token_estimate=len(augmented) // 4,
            domains_detected=detected_domains,
            processing_time_ms=elapsed,
        )

    def _build_augmented_prompt(
        self,
        user_prompt: str,
        system_prompt: str | None,
        memory_context: str,
        additional_context: dict[str, Any] | None,
    ) -> str:
        """Build the final augmented prompt."""
        parts = []

        # Start with existing system prompt if provided
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")

        # Add memory context
        parts.append(memory_context)
        parts.append("")

        # Add additional context if provided
        if additional_context:
            parts.append("<additional_context>")
            for key, value in additional_context.items():
                parts.append(f"  {key}: {value}")
            parts.append("</additional_context>")
            parts.append("")

        return "\n".join(parts)

    async def get_context_injection(
        self,
        query: str,
        format: ExportFormat = ExportFormat.CLAUDE_FORMAT,
        max_memories: int = 10,
        synthesize: bool = True,
    ) -> str:
        """
        Get memory context for injection into any LLM.

        When synthesize=True (default), uses the local SLM to extract
        meaningful insights from memories rather than returning raw snippets.

        Args:
            query: The query to match memories against
            format: Output format
            max_memories: Maximum memories to include
            synthesize: Whether to synthesize insights using SLM

        Returns:
            Formatted memory context string
        """
        if not self._query:
            raise RuntimeError("Augmenter not initialized")

        result = await self._query.get_context_for_query(query, max_memories=max_memories * 2)

        if not result.memories:
            return "<user_profile>\nNo relevant memories found for this query.\n</user_profile>"

        # Try to synthesize using Ollama if available
        if synthesize and self._slm:
            try:
                # Check if SLM is available (for Ollama)
                if hasattr(self._slm, 'is_available'):
                    is_avail = await self._slm.is_available()
                    if not is_avail:
                        # Fall back to raw export
                        return self._exporter.export(
                            result.memories[:max_memories],
                            format=format,
                            scores=result.scores[:max_memories],
                        )

                # Extract memory contents for synthesis
                memory_contents = [
                    m.summary if m.summary else m.content[:500]
                    for m in result.memories[:max_memories]
                ]

                # Synthesize insights
                synthesized = await self._slm.synthesize_context(
                    query=query,
                    memories=memory_contents,
                    user_name="Rod",  # Could be loaded from identity
                )

                if synthesized and len(synthesized) > 50:
                    # Format synthesized context
                    return f"""<user_profile>
The following represents synthesized knowledge about the user based on their personal memory system.
Use this context to provide personalized, relevant responses.

<synthesized_insights>
{synthesized}
</synthesized_insights>

<memory_sources>
Based on {len(result.memories)} relevant memories from emails, documents, and interactions.
</memory_sources>
</user_profile>"""

            except Exception as e:
                import logging
                logging.warning(f"Synthesis failed: {e}")
                pass  # Fall back to raw export on any error

        # Fall back to formatted export if synthesis fails or is disabled
        return self._exporter.export(
            result.memories[:max_memories],
            format=format,
            scores=result.scores[:max_memories],
        )

    async def create_memory_from_interaction(
        self,
        user_message: str,
        assistant_response: str,
        extract_facts: bool = True,
    ) -> list[MemoryEntry]:
        """
        Create memories from an interaction.

        Analyzes the exchange and extracts learnable information
        for storage in short-term memory.

        Args:
            user_message: What the user said
            assistant_response: How the assistant responded
            extract_facts: Whether to extract facts from response

        Returns:
            List of created memories
        """
        if not self._slm:
            return []

        created: list[MemoryEntry] = []

        # Extract from user message
        user_domains = await self._slm.extract_domains(user_message)
        user_entities = await self._slm.extract_entities(user_message)

        # Check for preferences
        if any(word in user_message.lower() for word in ["prefer", "like", "want", "always", "never"]):
            memory = MemoryEntry(
                content=user_message,
                memory_type=MemoryType.PREFERENCE,
                domains=user_domains,
                entities=user_entities,
            )
            stored = await self._storage.store(memory)
            created.append(stored)

        # Check for beliefs
        elif any(word in user_message.lower() for word in ["believe", "think", "feel that", "opinion"]):
            memory = MemoryEntry(
                content=user_message,
                memory_type=MemoryType.BELIEF,
                domains=user_domains,
                entities=user_entities,
            )
            stored = await self._storage.store(memory)
            created.append(stored)

        return created

    async def learn_from_feedback(
        self,
        memory_id: str,
        feedback: str,  # "correct", "incorrect", "outdated"
    ) -> MemoryEntry | None:
        """
        Update a memory based on user feedback.

        This enables the memory system to learn and improve over time.
        """
        from uuid import UUID

        memory = await self._storage.get(UUID(memory_id))
        if not memory:
            return None

        if feedback == "correct":
            memory.confidence.corroboration_count += 1
            memory.confidence.recalculate()
        elif feedback == "incorrect":
            memory.confidence.contradiction_count += 1
            memory.confidence.recalculate()
            if memory.confidence.overall < 0.3:
                memory.is_active = False
        elif feedback == "outdated":
            memory.is_active = False

        return await self._storage.update(memory)


class MemoryMiddleware:
    """
    Middleware for integrating memory augmentation into LLM pipelines.

    Can be used as a decorator or context manager to automatically
    augment prompts and track interactions.
    """

    def __init__(
        self,
        augmenter: MemoryAugmenter,
        auto_learn: bool = True,
    ):
        self._augmenter = augmenter
        self._auto_learn = auto_learn

    async def wrap_llm_call(
        self,
        llm_function: Callable,
        user_prompt: str,
        **kwargs: Any,
    ) -> tuple[str, AugmentedPrompt]:
        """
        Wrap an LLM call with memory augmentation.

        Args:
            llm_function: The LLM function to call
            user_prompt: User's message
            **kwargs: Additional kwargs for the LLM function

        Returns:
            Tuple of (response, augmentation_info)
        """
        # Augment the prompt
        augmented = await self._augmenter.augment_prompt(user_prompt)

        # Call the LLM with augmented system prompt
        response = await llm_function(
            user_prompt,
            system=augmented.memory_context,
            **kwargs,
        )

        # Learn from interaction if enabled
        if self._auto_learn:
            await self._augmenter.create_memory_from_interaction(
                user_prompt,
                response,
            )

        return response, augmented


async def create_augmenter(
    base_path: str = "~/memory/data",
) -> MemoryAugmenter:
    """
    Factory function to create a configured MemoryAugmenter.

    Usage:
        augmenter = await create_augmenter()
        result = await augmenter.augment_prompt("How do I configure Drupal?")
        print(result.memory_context)
    """
    storage = StorageManager(base_path=base_path)
    await storage.initialize()

    augmenter = MemoryAugmenter(storage)
    await augmenter.initialize()

    return augmenter
