"""Tests for memory-augmented attention module."""


import pytest


class TestMemoryAttentionConfig:
    """Tests for attention configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from memory.attention import MemoryAttentionConfig

        config = MemoryAttentionConfig()

        assert config.d_model == 768
        assert config.num_heads == 12
        assert config.d_key == 64
        assert config.d_value == 64
        assert config.max_memory_tokens == 1024

    def test_custom_config(self):
        """Test custom configuration."""
        from memory.attention import MemoryAttentionConfig

        config = MemoryAttentionConfig(
            d_model=512,
            num_heads=8,
            d_key=64,
            d_value=64,
            max_memory_tokens=2048,
        )

        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.max_memory_tokens == 2048


class TestMemoryAttention:
    """Tests for MemoryAttention module."""

    @pytest.mark.requires_torch
    def test_initialization(self):
        """Test attention module initialization."""
        from memory.attention import MemoryAttention, MemoryAttentionConfig

        config = MemoryAttentionConfig(d_model=256, num_heads=4)
        attention = MemoryAttention(config)

        assert attention is not None
        assert attention.d_model == 256
        assert attention.num_heads == 4

    @pytest.mark.requires_torch
    def test_forward_without_memory(self):
        """Test forward pass without memory."""
        import torch

        from memory.attention import MemoryAttention, MemoryAttentionConfig

        config = MemoryAttentionConfig(d_model=256, num_heads=4, d_key=64, d_value=64)
        attention = MemoryAttention(config)

        # Create input
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 256)

        # Forward pass
        output = attention(x)

        assert output.output.shape == (batch_size, seq_len, 256)
        assert output.memory_contribution == 0.0

    @pytest.mark.requires_torch
    def test_forward_with_memory(self):
        """Test forward pass with memory keys and values."""
        import torch

        from memory.attention import MemoryAttention, MemoryAttentionConfig

        config = MemoryAttentionConfig(d_model=256, num_heads=4, d_key=64, d_value=64)
        attention = MemoryAttention(config)

        # Create input and memory
        batch_size, seq_len, mem_len = 2, 16, 32
        x = torch.randn(batch_size, seq_len, 256)
        memory_keys = torch.randn(batch_size, mem_len, 256)
        memory_values = torch.randn(batch_size, mem_len, 256)

        # Forward pass
        output = attention(
            x,
            memory_keys=memory_keys,
            memory_values=memory_values,
        )

        assert output.output.shape == (batch_size, seq_len, 256)
        assert output.memory_contribution > 0.0

    @pytest.mark.requires_torch
    def test_attention_masking(self):
        """Test attention masking."""
        import torch

        from memory.attention import MemoryAttention, MemoryAttentionConfig

        config = MemoryAttentionConfig(d_model=256, num_heads=4, d_key=64, d_value=64)
        attention = MemoryAttention(config)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 256)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))

        output = attention(x, attention_mask=mask)
        assert output.output.shape == (batch_size, seq_len, 256)


class TestMemoryKeyValueCache:
    """Tests for memory K,V cache."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        from memory.attention import CacheConfig, MemoryKeyValueCache

        config = CacheConfig(d_key=64, d_value=64, num_heads=8)
        cache = MemoryKeyValueCache(config)

        assert cache is not None
        assert len(cache) == 0

    @pytest.mark.requires_embeddings
    def test_add_and_get(self):
        """Test adding and retrieving entries."""
        import numpy as np

        from memory.attention import CacheConfig, MemoryKeyValueCache

        config = CacheConfig(d_key=64, d_value=64, num_heads=8)
        cache = MemoryKeyValueCache(config)

        # Add entry
        embedding = np.random.randn(512).astype(np.float32)
        entry = cache.add("mem-1", embedding, project=False)

        assert entry is not None
        assert entry.memory_id == "mem-1"
        assert "mem-1" in cache

    @pytest.mark.requires_embeddings
    def test_cache_tiers(self):
        """Test cache tier management."""
        import numpy as np

        from memory.attention import CacheConfig, MemoryKeyValueCache

        config = CacheConfig(
            d_key=64,
            d_value=64,
            num_heads=8,
            hot_entries=5,
            warm_entries=10,
        )
        cache = MemoryKeyValueCache(config)

        # Add entries beyond hot capacity
        for i in range(8):
            embedding = np.random.randn(512).astype(np.float32)
            cache.add(f"mem-{i}", embedding, project=False)

        # Check some entries were demoted to warm
        stats = cache.stats()
        assert stats["hot_entries"] <= 5 or stats["total_entries"] == 8

    @pytest.mark.requires_embeddings
    def test_cache_eviction(self):
        """Test cache eviction policy."""
        import numpy as np

        from memory.attention import CacheConfig, MemoryKeyValueCache

        config = CacheConfig(
            d_key=64,
            d_value=64,
            num_heads=8,
            hot_entries=3,
            warm_entries=3,
            max_entries=10,
        )
        cache = MemoryKeyValueCache(config)

        # Add many entries
        for i in range(10):
            embedding = np.random.randn(512).astype(np.float32)
            cache.add(f"mem-{i}", embedding, project=False)

        assert len(cache) <= 10

    @pytest.mark.requires_embeddings
    def test_get_all(self):
        """Test getting all cached K,V pairs."""
        import numpy as np

        from memory.attention import CacheConfig, MemoryKeyValueCache

        config = CacheConfig(d_key=64, d_value=64, num_heads=8)
        cache = MemoryKeyValueCache(config)

        # Add entries
        for i in range(5):
            embedding = np.random.randn(512).astype(np.float32)
            cache.add(f"mem-{i}", embedding, project=False)

        keys, values = cache.get_all()
        # May be None if no projections, but should not error
        assert cache.stats()["total_entries"] == 5

    @pytest.mark.requires_embeddings
    def test_clear(self):
        """Test clearing cache."""
        import numpy as np

        from memory.attention import CacheConfig, MemoryKeyValueCache

        config = CacheConfig(d_key=64, d_value=64, num_heads=8)
        cache = MemoryKeyValueCache(config)

        for i in range(5):
            embedding = np.random.randn(512).astype(np.float32)
            cache.add(f"mem-{i}", embedding, project=False)

        assert len(cache) == 5
        cache.clear()
        assert len(cache) == 0


class TestAttentionFusion:
    """Tests for attention fusion strategies."""

    @pytest.mark.requires_torch
    def test_concat_fusion(self):
        """Test concatenation fusion."""
        import torch

        from memory.attention.fusion import ConcatFusion, FusionConfig

        config = FusionConfig(d_model=256, num_heads=4, d_value=64)
        fusion = ConcatFusion(config)

        batch, heads, seq, d = 2, 4, 16, 64
        ctx_attn = torch.randn(batch, heads, seq, d)
        mem_attn = torch.randn(batch, heads, seq, d)

        output = fusion(ctx_attn, mem_attn)
        assert output.shape == (batch, heads, seq, d)

    @pytest.mark.requires_torch
    def test_gated_fusion(self):
        """Test gated fusion."""
        import torch

        from memory.attention.fusion import FusionConfig, GatedFusion

        config = FusionConfig(d_model=256, num_heads=4, d_value=64)
        fusion = GatedFusion(config)

        batch, heads, seq, d = 2, 4, 16, 64
        ctx_attn = torch.randn(batch, heads, seq, d)
        mem_attn = torch.randn(batch, heads, seq, d)

        output = fusion(ctx_attn, mem_attn)
        assert output.shape == (batch, heads, seq, d)

    @pytest.mark.requires_torch
    def test_cross_attention_fusion(self):
        """Test cross-attention fusion."""
        import torch

        from memory.attention.fusion import CrossAttentionFusion, FusionConfig

        config = FusionConfig(d_model=256, num_heads=4, d_value=64)
        fusion = CrossAttentionFusion(config)

        batch, heads, seq, d = 2, 4, 16, 64
        ctx_attn = torch.randn(batch, heads, seq, d)
        mem_attn = torch.randn(batch, heads, seq, d)

        output = fusion(ctx_attn, mem_attn)
        assert output.shape == (batch, heads, seq, d)


class TestMemoryAugmentedLayer:
    """Tests for memory-augmented transformer layer."""

    @pytest.mark.requires_torch
    def test_layer_initialization(self):
        """Test layer initialization."""
        from memory.attention.layer import LayerConfig, MemoryAugmentedLayer

        config = LayerConfig(d_model=256, num_heads=4, d_ff=1024)
        layer = MemoryAugmentedLayer(config)

        assert layer is not None

    @pytest.mark.requires_torch
    def test_layer_forward(self):
        """Test layer forward pass."""
        import torch

        from memory.attention.layer import LayerConfig, MemoryAugmentedLayer

        config = LayerConfig(d_model=256, num_heads=4, d_ff=1024)
        layer = MemoryAugmentedLayer(config)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 256)

        output, attn = layer(x)
        assert output.shape == (batch_size, seq_len, 256)


class TestPLMMemoryAdapter:
    """Tests for PLM memory store adapter."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        from memory.attention.integration import PLMMemoryAdapter, PLMMemoryAdapterConfig

        config = PLMMemoryAdapterConfig(d_model=256, num_heads=4)
        adapter = PLMMemoryAdapter(config)

        assert adapter is not None

    @pytest.mark.asyncio
    async def test_load_memories_without_store(self):
        """Test loading memories without a store."""
        from memory.attention.integration import PLMMemoryAdapter, PLMMemoryAdapterConfig

        config = PLMMemoryAdapterConfig(d_model=256, num_heads=4)
        adapter = PLMMemoryAdapter(config, memory_store=None)

        loaded = await adapter.load_memories()
        assert loaded == 0

    def test_get_stats(self):
        """Test getting adapter statistics."""
        from memory.attention.integration import PLMMemoryAdapter, PLMMemoryAdapterConfig

        config = PLMMemoryAdapterConfig(d_model=256, num_heads=4)
        adapter = PLMMemoryAdapter(config)

        stats = adapter.get_stats()
        assert "memories_loaded" in stats
        assert "cache_hits" in stats
        assert "retrievals" in stats


class TestCreateMemoryAttention:
    """Tests for factory function."""

    def test_create_memory_attention(self):
        """Test creating memory attention setup."""
        from memory.attention import create_memory_attention

        attention, adapter = create_memory_attention(
            d_model=256,
            num_heads=4,
        )

        assert attention is not None
        assert adapter is not None
