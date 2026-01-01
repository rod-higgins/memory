"""
Integration between PLM Memory Store and Memory-Augmented Attention.

Connects the memory storage system to the attention mechanism:
- Converts memories to K,V embeddings
- Manages cache synchronization
- Provides query-time memory retrieval
- Supports incremental updates
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .cache import CacheConfig, CacheEntry, MemoryKeyValueCache
from .core import MemoryAttention, MemoryAttentionConfig
from .fusion import FusionStrategy


@dataclass
class PLMMemoryAdapterConfig:
    """Configuration for PLM memory adapter."""

    # Model dimensions
    d_model: int = 768
    num_heads: int = 12
    d_key: int = 64
    d_value: int = 64

    # Memory settings
    max_memories: int = 10000
    embedding_field: str = "embedding"  # Field name in memory entries

    # Caching
    cache_dir: str = ""
    hot_cache_size: int = 1000
    warm_cache_size: int = 5000

    # Retrieval
    topk_retrieval: int = 100
    relevance_threshold: float = 0.5

    # Update policy
    sync_interval_seconds: int = 60
    batch_size: int = 100


class PLMMemoryAdapter:
    """
    Adapter connecting PLM memory store to memory-augmented attention.

    Responsibilities:
    1. Convert MemoryEntry objects to attention K,V pairs
    2. Maintain synchronized cache with memory store
    3. Provide efficient retrieval for attention queries
    4. Handle memory updates and invalidation
    """

    def __init__(
        self,
        config: PLMMemoryAdapterConfig,
        memory_store: Any = None,  # MemoryStore from PLM
    ):
        self.config = config
        self.memory_store = memory_store

        # K,V projection layers (if using torch)
        self._key_proj = None
        self._value_proj = None
        self._init_projections()

        # Memory cache
        cache_config = CacheConfig(
            d_key=config.d_key,
            d_value=config.d_value,
            num_heads=config.num_heads,
            max_entries=config.max_memories,
            hot_entries=config.hot_cache_size,
            warm_entries=config.warm_cache_size,
            cache_dir=config.cache_dir,
        )
        self.cache = MemoryKeyValueCache(
            cache_config,
            key_projection=self._key_proj,
            value_projection=self._value_proj,
        )

        # Sync state
        self._last_sync: datetime | None = None
        self._sync_task: asyncio.Task | None = None

        # Statistics
        self.stats = {
            "memories_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "retrievals": 0,
        }

    def _init_projections(self) -> None:
        """Initialize K,V projection layers."""
        if HAS_TORCH:
            d_total = self.config.num_heads * self.config.d_key
            self._key_proj = nn.Linear(self.config.d_model, d_total)
            self._value_proj = nn.Linear(self.config.d_model, d_total)

            # Initialize with small weights for stability
            nn.init.xavier_uniform_(self._key_proj.weight, gain=0.1)
            nn.init.xavier_uniform_(self._value_proj.weight, gain=0.1)

    async def load_memories(
        self,
        limit: int | None = None,
        memory_type: str | None = None,
        domains: list[str] | None = None,
    ) -> int:
        """
        Load memories from store into cache.

        Args:
            limit: Maximum memories to load
            memory_type: Filter by memory type
            domains: Filter by domains

        Returns:
            Number of memories loaded
        """
        if self.memory_store is None:
            return 0

        limit = limit or self.config.max_memories
        loaded = 0

        try:
            # Query memories from store
            # This integrates with the PLM MemoryStore
            memories = await self._query_memories(limit, memory_type, domains)

            # Batch load into cache
            batch_ids = []
            batch_embeddings = []

            for memory in memories:
                embedding = getattr(memory, self.config.embedding_field, None)
                if embedding is None:
                    continue

                memory_id = getattr(memory, "id", str(loaded))
                batch_ids.append(memory_id)

                if HAS_TORCH:
                    if not isinstance(embedding, torch.Tensor):
                        embedding = torch.tensor(embedding, dtype=torch.float32)
                    batch_embeddings.append(embedding)
                elif HAS_NUMPY:
                    batch_embeddings.append(np.array(embedding))
                else:
                    batch_embeddings.append(embedding)

                # Process in batches
                if len(batch_ids) >= self.config.batch_size:
                    if HAS_TORCH:
                        embeddings_tensor = torch.stack(batch_embeddings)
                    elif HAS_NUMPY:
                        embeddings_tensor = np.stack(batch_embeddings)
                    else:
                        embeddings_tensor = batch_embeddings

                    self.cache.add_batch(batch_ids, embeddings_tensor)
                    loaded += len(batch_ids)
                    batch_ids = []
                    batch_embeddings = []

            # Process remaining
            if batch_ids:
                if HAS_TORCH:
                    embeddings_tensor = torch.stack(batch_embeddings)
                elif HAS_NUMPY:
                    embeddings_tensor = np.stack(batch_embeddings)
                else:
                    embeddings_tensor = batch_embeddings

                self.cache.add_batch(batch_ids, embeddings_tensor)
                loaded += len(batch_ids)

        except Exception as e:
            print(f"Error loading memories: {e}")

        self.stats["memories_loaded"] = loaded
        self._last_sync = datetime.now()

        return loaded

    async def _query_memories(
        self,
        limit: int,
        memory_type: str | None = None,
        domains: list[str] | None = None,
    ) -> list[Any]:
        """Query memories from the store."""
        try:
            # Try different store interfaces
            if hasattr(self.memory_store, "get_all"):
                return await self.memory_store.get_all(limit=limit)
            elif hasattr(self.memory_store, "search"):
                return await self.memory_store.search(
                    query="*",
                    limit=limit,
                    filters={
                        "memory_type": memory_type,
                        "domains": domains,
                    }
                    if memory_type or domains
                    else None,
                )
            elif hasattr(self.memory_store, "list"):
                return await self.memory_store.list(limit=limit)
        except Exception:
            pass

        return []

    def get_memory_kv(
        self,
        query_embedding: Any = None,
        topk: int | None = None,
    ) -> tuple[Any, Any, list[str]]:
        """
        Get memory K,V pairs for attention.

        Args:
            query_embedding: Optional query for relevance filtering
            topk: Number of memories to retrieve

        Returns:
            (keys, values, memory_ids) tuple
        """
        topk = topk or self.config.topk_retrieval
        self.stats["retrievals"] += 1

        if query_embedding is not None:
            # Relevance-based retrieval
            keys, values, memory_ids = self.cache.get_topk(query_embedding, topk)
            if keys is not None:
                self.stats["cache_hits"] += 1
            else:
                self.stats["cache_misses"] += 1
            return keys, values, memory_ids
        else:
            # Return all cached K,V
            keys, values = self.cache.get_all()
            return keys, values, []

    async def add_memory(
        self,
        memory_id: str,
        embedding: Any,
    ) -> CacheEntry:
        """
        Add a new memory to the cache.

        Called when new memories are added to the store.
        """
        return self.cache.add(memory_id, embedding)

    async def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory from the cache."""
        if memory_id in self.cache:
            # Clear from all tiers
            if memory_id in self.cache.hot:
                del self.cache.hot[memory_id]
            if memory_id in self.cache.warm:
                del self.cache.warm[memory_id]
            if memory_id in self.cache.cold_index:
                del self.cache.cold_index[memory_id]
            return True
        return False

    async def start_sync(self) -> None:
        """Start background synchronization with memory store."""
        if self._sync_task is not None:
            return

        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop_sync(self) -> None:
        """Stop background synchronization."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)
                await self._sync_updates()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Sync error: {e}")

    async def _sync_updates(self) -> None:
        """Sync recent updates from memory store."""
        if self.memory_store is None or self._last_sync is None:
            return

        # Get memories updated since last sync
        try:
            if hasattr(self.memory_store, "get_updated_since"):
                updates = await self.memory_store.get_updated_since(self._last_sync)

                for memory in updates:
                    memory_id = getattr(memory, "id", None)
                    embedding = getattr(memory, self.config.embedding_field, None)

                    if memory_id and embedding is not None:
                        # Update or add to cache
                        if memory_id in self.cache:
                            await self.remove_memory(memory_id)
                        self.cache.add(memory_id, embedding)

            self._last_sync = datetime.now()

        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        cache_stats = self.cache.stats()
        return {
            **self.stats,
            "cache": cache_stats,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
        }


def create_memory_attention(
    memory_store: Any = None,
    d_model: int = 768,
    num_heads: int = 12,
    fusion_strategy: FusionStrategy = FusionStrategy.GATED,
    max_memories: int = 10000,
    cache_dir: str = "",
) -> tuple[MemoryAttention, PLMMemoryAdapter]:
    """
    Create a complete memory attention setup.

    Returns configured MemoryAttention module and PLMMemoryAdapter.

    Usage:
        attention, adapter = create_memory_attention(memory_store)
        await adapter.load_memories()

        # In forward pass
        keys, values, _ = adapter.get_memory_kv(query_embedding)
        output = attention(query, context, memory_keys=keys, memory_values=values)
    """
    # Configure attention
    attention_config = MemoryAttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        d_key=d_model // num_heads,
        d_value=d_model // num_heads,
        max_memory_tokens=max_memories,
        memory_dim=d_model,
    )

    # Configure adapter
    adapter_config = PLMMemoryAdapterConfig(
        d_model=d_model,
        num_heads=num_heads,
        d_key=d_model // num_heads,
        d_value=d_model // num_heads,
        max_memories=max_memories,
        cache_dir=cache_dir,
    )

    if HAS_TORCH:
        from .core import MemoryAttention as TorchMemoryAttention

        attention = TorchMemoryAttention(attention_config)
    else:
        from .core import MemoryAttention

        attention = MemoryAttention(attention_config)

    adapter = PLMMemoryAdapter(adapter_config, memory_store)

    # Share projections
    if HAS_TORCH and adapter._key_proj is not None:
        adapter.cache.key_projection = attention.W_k_mem
        adapter.cache.value_projection = attention.W_v_mem

    return attention, adapter


class MemoryAugmentedInference:
    """
    High-level interface for memory-augmented inference.

    Combines:
    - Model (transformer with memory attention)
    - Memory adapter (connected to PLM store)
    - Query-time memory retrieval

    Usage:
        inference = MemoryAugmentedInference(model, memory_store)
        await inference.initialize()

        output = inference.generate("What did I say about...")
    """

    def __init__(
        self,
        model: Any,  # HuggingFace model or custom
        memory_store: Any,
        d_model: int = 768,
        num_heads: int = 12,
        topk_memories: int = 100,
    ):
        self.model = model
        self.d_model = d_model
        self.topk_memories = topk_memories

        # Create adapter
        adapter_config = PLMMemoryAdapterConfig(
            d_model=d_model,
            num_heads=num_heads,
            max_memories=10000,
            topk_retrieval=topk_memories,
        )
        self.adapter = PLMMemoryAdapter(adapter_config, memory_store)

        # Injector for adding memory to model
        from .layer import MemoryInjector

        self.injector = MemoryInjector(self.adapter.cache)

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize by loading memories and injecting into model."""
        # Load memories into cache
        await self.adapter.load_memories()

        # Inject memory attention into model
        self.injector.inject(self.model)

        # Start background sync
        await self.adapter.start_sync()

        self._initialized = True

    async def close(self) -> None:
        """Clean up resources."""
        await self.adapter.stop_sync()
        self.injector.remove(self.model)
        self.adapter.cache.close()

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any = None,
        query_for_memory: Any = None,
        **kwargs,
    ) -> Any:
        """
        Forward pass with memory retrieval.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            query_for_memory: Embedding for memory retrieval
            **kwargs: Additional model arguments
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")

        # Get relevant memories
        if query_for_memory is not None:
            keys, values, memory_ids = self.adapter.get_memory_kv(query_for_memory, self.topk_memories)
            kwargs["memory_keys"] = keys
            kwargs["memory_values"] = values

        # Forward through model
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    async def generate(
        self,
        prompt: str,
        tokenizer: Any = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> str:
        """
        Generate text with memory augmentation.

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer for encoding/decoding
            max_new_tokens: Maximum tokens to generate
            **kwargs: Generation arguments
        """
        if tokenizer is None:
            raise ValueError("Tokenizer required for generation")

        # Tokenize
        if HAS_TORCH:
            inputs = tokenizer(prompt, return_tensors="pt")
        else:
            inputs = tokenizer(prompt)

        # Get query embedding for memory retrieval
        query_embedding = None
        if hasattr(self.model, "get_input_embeddings"):
            with torch.no_grad():
                input_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
                query_embedding = input_embeds.mean(dim=1)  # Average pool

        # Get memories
        keys, values, _ = self.adapter.get_memory_kv(query_embedding)

        # Generate
        if hasattr(self.model, "generate"):
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                memory_keys=keys,
                memory_values=values,
                **kwargs,
            )
        else:
            outputs = self.forward(
                inputs["input_ids"],
                memory_keys=keys,
                memory_values=values,
            )

        # Decode
        if HAS_TORCH and hasattr(outputs, "shape"):
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            return str(outputs)

    def get_stats(self) -> dict[str, Any]:
        """Get inference statistics."""
        return {
            "adapter": self.adapter.get_stats(),
            "initialized": self._initialized,
        }
