"""
Memory-Augmented Attention for PLM.

Extends transformer attention to directly attend over the memory store,
enabling native memory integration rather than simple context prepending.

Mathematical formulation:
    Attention([Q_input, Q_memory], [K_context, K_memory], [V_context, V_memory])

This allows the model to seamlessly query and integrate memories during
inference, with learned attention weights determining relevance.

Key components:
- MemoryAttention: Core attention mechanism with memory integration
- MemoryKeyValueCache: Efficient K,V storage for memory embeddings
- AttentionFusion: Strategies for combining context and memory attention
- MemoryAugmentedLayer: Drop-in transformer layer replacement
"""

from .cache import (
    CacheConfig,
    CacheEntry,
    MemoryKeyValueCache,
)
from .core import (
    AttentionOutput,
    MemoryAttention,
    MemoryAttentionConfig,
)
from .fusion import (
    AttentionFusion,
    ConcatFusion,
    CrossAttentionFusion,
    FusionStrategy,
    GatedFusion,
)
from .integration import (
    PLMMemoryAdapter,
    create_memory_attention,
)
from .layer import (
    MemoryAugmentedLayer,
    MemoryAugmentedTransformer,
)

__all__ = [
    # Core attention
    "MemoryAttention",
    "MemoryAttentionConfig",
    "AttentionOutput",
    # Cache
    "MemoryKeyValueCache",
    "CacheConfig",
    "CacheEntry",
    # Fusion strategies
    "AttentionFusion",
    "FusionStrategy",
    "GatedFusion",
    "ConcatFusion",
    "CrossAttentionFusion",
    # Layer
    "MemoryAugmentedLayer",
    "MemoryAugmentedTransformer",
    # Integration
    "PLMMemoryAdapter",
    "create_memory_attention",
]
