"""
Memory Key-Value Cache for efficient attention over memory store.

Pre-computes and caches K,V projections for memory entries, enabling
efficient attention computation without re-projecting on every query.

Features:
- Hierarchical caching (hot/warm/cold)
- Incremental updates
- Efficient retrieval with pre-filtering
- Memory-mapped storage for large caches
"""

from __future__ import annotations

import importlib.util
import mmap
import struct
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_NUMPY:
    import numpy as np


class CacheTier(Enum):
    """Tier for cached memory entries."""

    HOT = "hot"  # Recently accessed, in memory
    WARM = "warm"  # Less recent, compressed in memory
    COLD = "cold"  # Rarely accessed, on disk


@dataclass
class CacheConfig:
    """Configuration for memory K,V cache."""

    # Dimensions
    d_key: int = 64
    d_value: int = 64
    num_heads: int = 12

    # Capacity
    max_entries: int = 100_000
    hot_entries: int = 10_000  # Keep in memory
    warm_entries: int = 50_000  # Compressed in memory

    # Storage
    cache_dir: str = ""  # For cold storage
    use_mmap: bool = True  # Memory-mapped files for cold tier

    # Update policy
    update_batch_size: int = 100
    eviction_policy: str = "lru"  # lru, lfu, fifo

    # Compression
    quantize_warm: bool = True  # 8-bit quantization for warm tier
    quantize_bits: int = 8


@dataclass
class CacheEntry:
    """A cached memory K,V pair."""

    id: str = field(default_factory=lambda: str(uuid4()))
    memory_id: str = ""  # Reference to source memory

    # Cached projections
    key: Any = None  # [num_heads * d_key] tensor/array
    value: Any = None  # [num_heads * d_value] tensor/array

    # Original embedding (for re-projection)
    embedding: Any = None

    # Metadata
    tier: CacheTier = CacheTier.HOT
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # For warm tier (quantized)
    key_quantized: bytes | None = None
    value_quantized: bytes | None = None
    key_scale: float = 1.0
    value_scale: float = 1.0


class MemoryKeyValueCache:
    """
    Efficient cache for memory K,V projections.

    Enables O(1) lookup of pre-computed key-value pairs for memory
    attention, with tiered storage for scalability.

    Usage:
        cache = MemoryKeyValueCache(config)

        # Add memories
        cache.add(memory_id, embedding)

        # Get K,V for attention
        keys, values = cache.get_all()

        # Or filter by relevance
        keys, values, indices = cache.get_topk(query_embedding, k=100)
    """

    def __init__(
        self,
        config: CacheConfig,
        key_projection: Any = None,  # nn.Linear or weight matrix
        value_projection: Any = None,
    ):
        self.config = config
        self.key_projection = key_projection
        self.value_projection = value_projection

        # Tiered storage
        self.hot: dict[str, CacheEntry] = {}
        self.warm: dict[str, CacheEntry] = {}
        self.cold_index: dict[str, int] = {}  # memory_id -> file offset

        # Access tracking for eviction
        self.access_order: list[str] = []  # For LRU
        self.access_counts: dict[str, int] = {}  # For LFU

        # Cold storage
        self._cold_file: Any = None
        self._cold_mmap: mmap.mmap | None = None

        # Thread safety
        self._lock = threading.RLock()

        # Initialize cold storage
        if config.cache_dir:
            self._init_cold_storage()

    def _init_cold_storage(self) -> None:
        """Initialize memory-mapped cold storage."""
        if not self.config.cache_dir:
            return

        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        cold_file = cache_path / "kv_cache.bin"

        if self.config.use_mmap:
            # Create or open cold storage file
            if not cold_file.exists():
                # Create with initial size
                initial_size = self.config.max_entries * (
                    self.config.num_heads * (self.config.d_key + self.config.d_value) * 4
                )
                with open(cold_file, "wb") as f:
                    f.write(b'\0' * min(initial_size, 1024 * 1024 * 100))  # Max 100MB initial

            self._cold_file = open(cold_file, "r+b")
            self._cold_mmap = mmap.mmap(
                self._cold_file.fileno(), 0,
                access=mmap.ACCESS_WRITE
            )

    def add(
        self,
        memory_id: str,
        embedding: Any,
        project: bool = True,
    ) -> CacheEntry:
        """
        Add a memory to the cache.

        Args:
            memory_id: Unique memory identifier
            embedding: Memory embedding vector
            project: Whether to compute K,V projections

        Returns:
            CacheEntry with cached K,V
        """
        with self._lock:
            # Check if already cached
            if memory_id in self.hot:
                return self.hot[memory_id]
            if memory_id in self.warm:
                return self._promote_warm(memory_id)
            if memory_id in self.cold_index:
                return self._promote_cold(memory_id)

            # Compute K,V projections
            key, value = self._project(embedding) if project else (None, None)

            entry = CacheEntry(
                memory_id=memory_id,
                key=key,
                value=value,
                embedding=embedding,
                tier=CacheTier.HOT,
            )

            # Add to hot tier
            self.hot[memory_id] = entry
            self._update_access(memory_id)

            # Evict if needed
            self._evict_if_needed()

            return entry

    def add_batch(
        self,
        memory_ids: list[str],
        embeddings: Any,  # [batch, d_model]
        project: bool = True,
    ) -> list[CacheEntry]:
        """Add a batch of memories efficiently."""
        entries = []

        if project and HAS_TORCH and self.key_projection is not None:
            import torch

            # Batch projection
            with torch.no_grad():
                if isinstance(embeddings, torch.Tensor):
                    keys = self.key_projection(embeddings)
                    values = self.value_projection(embeddings)
                else:
                    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
                    keys = self.key_projection(emb_tensor)
                    values = self.value_projection(emb_tensor)

            for i, memory_id in enumerate(memory_ids):
                entry = CacheEntry(
                    memory_id=memory_id,
                    key=keys[i],
                    value=values[i],
                    embedding=embeddings[i] if hasattr(embeddings, '__getitem__') else None,
                    tier=CacheTier.HOT,
                )
                entries.append(entry)

                with self._lock:
                    self.hot[memory_id] = entry
                    self._update_access(memory_id)

        else:
            for i, memory_id in enumerate(memory_ids):
                emb = embeddings[i] if hasattr(embeddings, '__getitem__') else embeddings
                entries.append(self.add(memory_id, emb, project))

        self._evict_if_needed()
        return entries

    def get(self, memory_id: str) -> CacheEntry | None:
        """Get a cached entry by memory ID."""
        with self._lock:
            if memory_id in self.hot:
                self._update_access(memory_id)
                return self.hot[memory_id]
            if memory_id in self.warm:
                return self._promote_warm(memory_id)
            if memory_id in self.cold_index:
                return self._promote_cold(memory_id)
        return None

    def get_all(self) -> tuple[Any, Any]:
        """
        Get all cached K,V pairs.

        Returns:
            (keys, values) tensors of shape [num_entries, num_heads * d]
        """
        with self._lock:
            all_keys = []
            all_values = []

            # Hot tier
            for entry in self.hot.values():
                if entry.key is not None:
                    all_keys.append(entry.key)
                    all_values.append(entry.value)

            # Warm tier (dequantize)
            for entry in self.warm.values():
                key, value = self._dequantize_entry(entry)
                if key is not None:
                    all_keys.append(key)
                    all_values.append(value)

            if not all_keys:
                return None, None

            if HAS_TORCH:
                import torch
                if isinstance(all_keys[0], torch.Tensor):
                    return torch.stack(all_keys), torch.stack(all_values)

            if HAS_NUMPY:
                return np.stack(all_keys), np.stack(all_values)

            return all_keys, all_values

    def get_topk(
        self,
        query_embedding: Any,
        k: int = 100,
    ) -> tuple[Any, Any, list[str]]:
        """
        Get top-k most relevant K,V pairs.

        Uses approximate similarity for efficiency.

        Returns:
            (keys, values, memory_ids) for top-k entries
        """
        with self._lock:
            # Collect all embeddings
            embeddings = []
            memory_ids = []

            for memory_id, entry in self.hot.items():
                if entry.embedding is not None:
                    embeddings.append(entry.embedding)
                    memory_ids.append(memory_id)

            for memory_id, entry in self.warm.items():
                if entry.embedding is not None:
                    embeddings.append(entry.embedding)
                    memory_ids.append(memory_id)

            if not embeddings:
                return None, None, []

            # Compute similarities
            if HAS_TORCH:
                import torch
                if not isinstance(query_embedding, torch.Tensor):
                    query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
                if not isinstance(embeddings[0], torch.Tensor):
                    embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]

                emb_stack = torch.stack(embeddings)
                similarities = torch.matmul(emb_stack, query_embedding)
                topk_values, topk_indices = torch.topk(similarities, min(k, len(embeddings)))

                topk_ids = [memory_ids[i] for i in topk_indices.tolist()]

            elif HAS_NUMPY:
                emb_stack = np.stack(embeddings)
                query = np.array(query_embedding)
                similarities = emb_stack @ query
                topk_indices = np.argsort(similarities)[-k:][::-1]

                topk_ids = [memory_ids[i] for i in topk_indices]

            else:
                # Simple fallback
                topk_ids = memory_ids[:k]

            # Collect K,V for top-k
            keys = []
            values = []
            for memory_id in topk_ids:
                entry = self.get(memory_id)
                if entry and entry.key is not None:
                    keys.append(entry.key)
                    values.append(entry.value)

            if not keys:
                return None, None, []

            if HAS_TORCH:
                return torch.stack(keys), torch.stack(values), topk_ids
            elif HAS_NUMPY:
                return np.stack(keys), np.stack(values), topk_ids

            return keys, values, topk_ids

    def _project(self, embedding: Any) -> tuple[Any, Any]:
        """Compute K,V projections for an embedding."""
        if self.key_projection is None or self.value_projection is None:
            return embedding, embedding

        if HAS_TORCH:
            import torch

            with torch.no_grad():
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                key = self.key_projection(embedding)
                value = self.value_projection(embedding)
                return key, value

        # NumPy fallback
        if HAS_NUMPY:
            if hasattr(self.key_projection, 'weight'):
                key = embedding @ self.key_projection.weight.T.numpy()
                value = embedding @ self.value_projection.weight.T.numpy()
            else:
                key = embedding @ self.key_projection
                value = embedding @ self.value_projection
            return key, value

        return embedding, embedding

    def _update_access(self, memory_id: str) -> None:
        """Update access tracking for eviction policy."""
        # Update access count
        self.access_counts[memory_id] = self.access_counts.get(memory_id, 0) + 1

        # Update LRU order
        if memory_id in self.access_order:
            self.access_order.remove(memory_id)
        self.access_order.append(memory_id)

    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        # Check hot tier
        while len(self.hot) > self.config.hot_entries:
            self._demote_to_warm()

        # Check warm tier
        while len(self.warm) > self.config.warm_entries:
            self._demote_to_cold()

    def _demote_to_warm(self) -> None:
        """Demote least recently used entry from hot to warm."""
        if not self.hot:
            return

        # Find victim based on eviction policy
        if self.config.eviction_policy == "lru":
            # Find oldest access in hot tier
            victim_id = None
            for memory_id in self.access_order:
                if memory_id in self.hot:
                    victim_id = memory_id
                    break
        elif self.config.eviction_policy == "lfu":
            # Find least frequently used
            victim_id = min(
                self.hot.keys(),
                key=lambda x: self.access_counts.get(x, 0)
            )
        else:
            # FIFO
            victim_id = next(iter(self.hot))

        if victim_id is None:
            return

        entry = self.hot.pop(victim_id)
        entry.tier = CacheTier.WARM

        # Quantize for warm tier
        if self.config.quantize_warm:
            self._quantize_entry(entry)

        self.warm[victim_id] = entry

    def _demote_to_cold(self) -> None:
        """Demote entry from warm to cold (disk)."""
        if not self.warm or self._cold_mmap is None:
            return

        # Find victim
        victim_id = next(iter(self.warm))
        entry = self.warm.pop(victim_id)
        entry.tier = CacheTier.COLD

        # Write to cold storage
        offset = self._write_cold(entry)
        self.cold_index[victim_id] = offset

    def _promote_warm(self, memory_id: str) -> CacheEntry:
        """Promote entry from warm to hot tier."""
        entry = self.warm.pop(memory_id)
        entry.tier = CacheTier.HOT
        entry.last_accessed = datetime.now()
        entry.access_count += 1

        # Dequantize
        if entry.key_quantized is not None:
            entry.key, entry.value = self._dequantize_entry(entry)
            entry.key_quantized = None
            entry.value_quantized = None

        self.hot[memory_id] = entry
        self._update_access(memory_id)
        self._evict_if_needed()

        return entry

    def _promote_cold(self, memory_id: str) -> CacheEntry | None:
        """Promote entry from cold (disk) to hot tier."""
        if memory_id not in self.cold_index:
            return None

        offset = self.cold_index.pop(memory_id)
        entry = self._read_cold(offset, memory_id)

        if entry:
            entry.tier = CacheTier.HOT
            entry.last_accessed = datetime.now()
            self.hot[memory_id] = entry
            self._update_access(memory_id)
            self._evict_if_needed()

        return entry

    def _quantize_entry(self, entry: CacheEntry) -> None:
        """Quantize K,V to 8-bit for warm tier."""
        if entry.key is None:
            return

        if HAS_TORCH:
            import torch

            if isinstance(entry.key, torch.Tensor):
                key = entry.key.float()
            else:
                key = torch.tensor(entry.key)
            if isinstance(entry.value, torch.Tensor):
                value = entry.value.float()
            else:
                value = torch.tensor(entry.value)

            # Compute scale
            entry.key_scale = key.abs().max().item() / 127.0
            entry.value_scale = value.abs().max().item() / 127.0

            # Quantize to int8
            key_q = (key / entry.key_scale).round().clamp(-128, 127).to(torch.int8)
            value_q = (value / entry.value_scale).round().clamp(-128, 127).to(torch.int8)

            entry.key_quantized = key_q.numpy().tobytes()
            entry.value_quantized = value_q.numpy().tobytes()

        elif HAS_NUMPY:
            key = np.array(entry.key)
            value = np.array(entry.value)

            entry.key_scale = np.abs(key).max() / 127.0
            entry.value_scale = np.abs(value).max() / 127.0

            key_q = np.round(key / entry.key_scale).clip(-128, 127).astype(np.int8)
            value_q = np.round(value / entry.value_scale).clip(-128, 127).astype(np.int8)

            entry.key_quantized = key_q.tobytes()
            entry.value_quantized = value_q.tobytes()

    def _dequantize_entry(self, entry: CacheEntry) -> tuple[Any, Any]:
        """Dequantize K,V from 8-bit."""
        if entry.key_quantized is None:
            return entry.key, entry.value

        if HAS_TORCH:
            import torch

            key_q = torch.frombuffer(bytearray(entry.key_quantized), dtype=torch.int8).float()
            value_q = torch.frombuffer(bytearray(entry.value_quantized), dtype=torch.int8).float()

            key = key_q * entry.key_scale
            value = value_q * entry.value_scale

            return key, value

        elif HAS_NUMPY:
            key_q = np.frombuffer(entry.key_quantized, dtype=np.int8).astype(np.float32)
            value_q = np.frombuffer(entry.value_quantized, dtype=np.int8).astype(np.float32)

            key = key_q * entry.key_scale
            value = value_q * entry.value_scale

            return key, value

        return None, None

    def _write_cold(self, entry: CacheEntry) -> int:
        """Write entry to cold storage."""
        if self._cold_mmap is None:
            return -1

        # Serialize entry
        data = self._serialize_entry(entry)

        # Find write position (simple append)
        # In production, would use proper allocation
        offset = len(self.cold_index) * (len(data) + 4)

        if offset + len(data) + 4 > len(self._cold_mmap):
            return -1  # Would need to grow file

        # Write length + data
        self._cold_mmap[offset:offset + 4] = struct.pack('I', len(data))
        self._cold_mmap[offset + 4:offset + 4 + len(data)] = data

        return offset

    def _read_cold(self, offset: int, memory_id: str) -> CacheEntry | None:
        """Read entry from cold storage."""
        if self._cold_mmap is None or offset < 0:
            return None

        try:
            # Read length
            length = struct.unpack('I', self._cold_mmap[offset:offset + 4])[0]
            # Read data
            data = bytes(self._cold_mmap[offset + 4:offset + 4 + length])

            return self._deserialize_entry(data, memory_id)
        except Exception:
            return None

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize entry for cold storage."""
        parts = []

        # Key
        if entry.key_quantized:
            parts.append(entry.key_quantized)
        elif entry.key is not None:
            if HAS_TORCH:
                import torch
                if isinstance(entry.key, torch.Tensor):
                    parts.append(entry.key.numpy().tobytes())
                else:
                    parts.append(np.array(entry.key).tobytes())
            elif HAS_NUMPY:
                parts.append(np.array(entry.key).tobytes())

        # Value
        if entry.value_quantized:
            parts.append(entry.value_quantized)
        elif entry.value is not None:
            if HAS_TORCH:
                import torch
                if isinstance(entry.value, torch.Tensor):
                    parts.append(entry.value.numpy().tobytes())
                else:
                    parts.append(np.array(entry.value).tobytes())
            elif HAS_NUMPY:
                parts.append(np.array(entry.value).tobytes())

        return b''.join(parts)

    def _deserialize_entry(self, data: bytes, memory_id: str) -> CacheEntry:
        """Deserialize entry from cold storage."""
        dim = self.config.num_heads * self.config.d_key
        bytes_per_entry = dim * 4  # float32

        entry = CacheEntry(memory_id=memory_id, tier=CacheTier.HOT)

        if len(data) >= bytes_per_entry * 2:
            if HAS_NUMPY:
                entry.key = np.frombuffer(data[:bytes_per_entry], dtype=np.float32)
                value_start = bytes_per_entry
                value_end = bytes_per_entry * 2
                entry.value = np.frombuffer(data[value_start:value_end], dtype=np.float32)

        return entry

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self.hot.clear()
            self.warm.clear()
            self.cold_index.clear()
            self.access_order.clear()
            self.access_counts.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "hot_entries": len(self.hot),
            "warm_entries": len(self.warm),
            "cold_entries": len(self.cold_index),
            "total_entries": len(self.hot) + len(self.warm) + len(self.cold_index),
            "hot_capacity": self.config.hot_entries,
            "warm_capacity": self.config.warm_entries,
        }

    def __len__(self) -> int:
        """Total number of cached entries."""
        return len(self.hot) + len(self.warm) + len(self.cold_index)

    def __contains__(self, memory_id: str) -> bool:
        """Check if memory is cached."""
        return (
            memory_id in self.hot or
            memory_id in self.warm or
            memory_id in self.cold_index
        )

    def close(self) -> None:
        """Close cache and release resources."""
        if self._cold_mmap:
            self._cold_mmap.close()
        if self._cold_file:
            self._cold_file.close()
