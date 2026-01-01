# Memory-Augmented Attention

## Overview

The memory-augmented attention module extends standard transformer attention to directly attend over the memory store, enabling native memory integration rather than simple context prepending.

## Mathematical Formulation

### Standard Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

### Memory-Augmented Attention

```
Attention([Q_input, Q_memory], [K_context, K_memory], [V_context, V_memory])
```

This allows the model to seamlessly query and integrate memories during inference, with learned attention weights determining relevance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEMORY-AUGMENTED ATTENTION                            │
│                                                                         │
│   Input Sequence                    Memory Store                         │
│   ┌──────────────┐                 ┌──────────────┐                     │
│   │  Embeddings  │                 │   K,V Cache  │                     │
│   └──────┬───────┘                 └──────┬───────┘                     │
│          │                                │                              │
│          ▼                                ▼                              │
│   ┌──────────────┐                 ┌──────────────┐                     │
│   │   Q_input    │                 │   K_memory   │                     │
│   │   K_context  │                 │   V_memory   │                     │
│   │   V_context  │                 └──────────────┘                     │
│   └──────┬───────┘                        │                              │
│          │                                │                              │
│          └────────────┬───────────────────┘                              │
│                       │                                                  │
│                       ▼                                                  │
│              ┌────────────────┐                                          │
│              │  Concatenate   │                                          │
│              │    K, V        │                                          │
│              └────────┬───────┘                                          │
│                       │                                                  │
│                       ▼                                                  │
│              ┌────────────────┐                                          │
│              │   Attention    │                                          │
│              │   Computation  │                                          │
│              └────────┬───────┘                                          │
│                       │                                                  │
│                       ▼                                                  │
│              ┌────────────────┐                                          │
│              │    Fusion      │                                          │
│              │   Strategy     │                                          │
│              └────────┬───────┘                                          │
│                       │                                                  │
│                       ▼                                                  │
│                   Output                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. MemoryAttention

The core attention mechanism with memory integration.

```python
from memory.attention import MemoryAttention, MemoryAttentionConfig

config = MemoryAttentionConfig(
    d_model=768,
    num_heads=12,
    d_key=64,
    d_value=64,
    dropout=0.1,
    max_memory_size=1000,
    memory_gate=True,  # Learnable gate for memory contribution
)

attention = MemoryAttention(config)

# Forward pass
output = attention(
    query=input_tensor,           # [batch, seq_len, d_model]
    context=context_tensor,       # [batch, ctx_len, d_model]
    memory_keys=memory_k,         # [batch, mem_len, d_key * num_heads]
    memory_values=memory_v,       # [batch, mem_len, d_value * num_heads]
    attention_mask=mask,          # Optional attention mask
)
```

### 2. MemoryKeyValueCache

Efficient K,V storage for memory embeddings with hierarchical caching.

```python
from memory.attention import MemoryKeyValueCache, CacheConfig

cache_config = CacheConfig(
    num_heads=12,
    d_key=64,
    d_value=64,
    hot_capacity=1000,      # In-memory, full precision
    warm_capacity=10000,    # In-memory, quantized (8-bit)
    cold_capacity=100000,   # Memory-mapped on disk
    eviction_policy="lru",  # or "fifo", "importance"
)

cache = MemoryKeyValueCache(cache_config)

# Add entries
cache.add(memory_id="mem-123", key=key_tensor, value=value_tensor)

# Retrieve
keys, values, mask = cache.get_batch(memory_ids)

# Automatic tier management
cache.manage_tiers()  # Promotes/demotes based on access patterns
```

### 3. Fusion Strategies

Multiple strategies for combining context and memory attention outputs.

#### Concatenation Fusion

```python
from memory.attention import ConcatFusion

fusion = ConcatFusion(d_model=768)
output = fusion(context_attention, memory_attention)
```

#### Gated Fusion

Learnable gate determines memory contribution:

```python
from memory.attention import GatedFusion

fusion = GatedFusion(d_model=768)
# gate = sigmoid(W * concat(context, memory))
# output = gate * memory + (1 - gate) * context
output = fusion(context_attention, memory_attention)
```

#### Cross-Attention Fusion

Query attends to both context and memory outputs:

```python
from memory.attention import CrossAttentionFusion

fusion = CrossAttentionFusion(
    d_model=768,
    num_heads=8,
)
output = fusion(context_attention, memory_attention)
```

### 4. MemoryAugmentedLayer

Drop-in replacement for standard transformer layers.

```python
from memory.attention import MemoryAugmentedLayer

layer = MemoryAugmentedLayer(
    d_model=768,
    num_heads=12,
    d_ff=3072,
    memory_attention_config=config,
    fusion_strategy="gated",
)

output = layer(
    hidden_states=x,
    memory_keys=mem_k,
    memory_values=mem_v,
    attention_mask=mask,
)
```

### 5. MemoryAugmentedTransformer

Full transformer with memory augmentation.

```python
from memory.attention import MemoryAugmentedTransformer

transformer = MemoryAugmentedTransformer(
    num_layers=12,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    memory_config=config,
    fusion_strategy="gated",
)

output = transformer(
    input_ids=tokens,
    memory_keys=mem_k,
    memory_values=mem_v,
)
```

## Cache Tier Management

The cache uses three tiers for efficiency:

| Tier | Storage | Precision | Latency | Capacity |
|------|---------|-----------|---------|----------|
| **Hot** | In-memory | float32 | <1ms | 1K entries |
| **Warm** | In-memory | int8 | ~2ms | 10K entries |
| **Cold** | Disk (mmap) | float32 | ~10ms | 100K entries |

### Eviction Policies

- **LRU** (Least Recently Used): Evict entries not accessed recently
- **FIFO** (First In First Out): Evict oldest entries
- **Importance**: Evict based on attention scores and access patterns

### Automatic Tier Promotion

```python
# Entries are automatically promoted based on access patterns:
# Cold → Warm: Accessed 2+ times in 24 hours
# Warm → Hot: Accessed 5+ times in 1 hour or high attention score
# Hot → Warm: Not accessed in 1 hour
# Warm → Cold: Not accessed in 24 hours
```

## Integration with PLM

### PLMMemoryAdapter

Bridges the memory system with the attention layer:

```python
from memory.attention import PLMMemoryAdapter
from memory import MemoryAPI

# Initialize
api = MemoryAPI()
await api.initialize()

adapter = PLMMemoryAdapter(
    memory_api=api,
    cache_config=cache_config,
    attention_config=attention_config,
)

# Get memory K,V for a query
memory_k, memory_v = await adapter.get_memory_kv(
    query="What are my programming preferences?",
    max_memories=100,
)

# Use in attention
output = attention(
    query=input_tensor,
    memory_keys=memory_k,
    memory_values=memory_v,
)
```

## Performance Considerations

### Memory Efficiency

- **Quantization**: Warm tier uses 8-bit quantization (4x memory reduction)
- **Memory mapping**: Cold tier uses memory-mapped files (no RAM usage)
- **Lazy loading**: K,V pairs loaded on demand

### Computation Efficiency

- **Pre-computed K,V**: Memory projections computed once at storage
- **Chunked attention**: Large memory sets processed in chunks
- **Sparse attention**: Optional top-k filtering for very large memory

### Batching

```python
# Efficient batch processing
batch_outputs = attention.forward_batch(
    queries=[q1, q2, q3],
    memory_ids=[ids1, ids2, ids3],
    chunk_size=100,
)
```

## Example: Complete Integration

```python
import asyncio
from memory import MemoryAPI
from memory.attention import (
    MemoryAttention,
    MemoryAttentionConfig,
    MemoryKeyValueCache,
    CacheConfig,
    PLMMemoryAdapter,
)

async def main():
    # Initialize memory system
    api = MemoryAPI()
    await api.initialize()

    # Configure attention
    attention_config = MemoryAttentionConfig(
        d_model=768,
        num_heads=12,
        memory_gate=True,
    )

    cache_config = CacheConfig(
        num_heads=12,
        d_key=64,
        d_value=64,
        hot_capacity=1000,
    )

    # Create adapter
    adapter = PLMMemoryAdapter(
        memory_api=api,
        cache_config=cache_config,
        attention_config=attention_config,
    )

    # Create attention layer
    attention = MemoryAttention(attention_config)

    # Process query with memory
    query = "How should I structure my Drupal module?"
    memory_k, memory_v = await adapter.get_memory_kv(query)

    # In practice, input_tensor comes from your model
    input_tensor = get_embeddings(query)

    output = attention(
        query=input_tensor,
        memory_keys=memory_k,
        memory_values=memory_v,
    )

    print(f"Output shape: {output.output.shape}")
    print(f"Memory attention weight: {output.memory_gate_value:.2%}")

asyncio.run(main())
```

## Configuration Reference

### MemoryAttentionConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 768 | Model dimension |
| `num_heads` | int | 12 | Number of attention heads |
| `d_key` | int | 64 | Key dimension per head |
| `d_value` | int | 64 | Value dimension per head |
| `dropout` | float | 0.1 | Dropout rate |
| `max_memory_size` | int | 1000 | Maximum memories to attend |
| `memory_gate` | bool | True | Use learnable memory gate |
| `scale_memory` | bool | True | Scale memory attention scores |

### CacheConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_heads` | int | 12 | Number of attention heads |
| `d_key` | int | 64 | Key dimension per head |
| `d_value` | int | 64 | Value dimension per head |
| `hot_capacity` | int | 1000 | Hot tier capacity |
| `warm_capacity` | int | 10000 | Warm tier capacity |
| `cold_capacity` | int | 100000 | Cold tier capacity |
| `eviction_policy` | str | "lru" | Eviction policy |
| `cold_storage_path` | Path | None | Path for cold storage |

---

*See [ARCHITECTURE.md](./ARCHITECTURE.md) for overall system architecture.*
*See [THEORY.md](./THEORY.md) for theoretical foundations.*
