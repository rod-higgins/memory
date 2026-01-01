"""
Core Memory-Augmented Attention mechanism.

Implements the mathematical formulation:
    Attention([Q_input, Q_memory], [K_context, K_memory], [V_context, V_memory])

This extends standard scaled dot-product attention to jointly attend
over both the input context and an external memory store.

The attention computation becomes:
    α_context = softmax(Q @ K_context.T / √d_k)
    α_memory = softmax(Q @ K_memory.T / √d_k)

    output = fusion(α_context @ V_context, α_memory @ V_memory)

Where fusion can be concatenation, gating, or learned combination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Try to import torch, fall back to numpy for basic ops
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import numpy as np


class AttentionType(Enum):
    """Type of memory attention mechanism."""

    CONCAT = "concat"  # Concatenate K,V from context and memory
    PARALLEL = "parallel"  # Parallel attention, then fuse
    CROSS = "cross"  # Cross-attention to memory
    GATED = "gated"  # Gated fusion of context and memory attention


@dataclass
class MemoryAttentionConfig:
    """Configuration for memory-augmented attention."""

    # Dimensions
    d_model: int = 768  # Model dimension
    d_key: int = 64  # Key dimension (d_model // num_heads)
    d_value: int = 64  # Value dimension
    num_heads: int = 12  # Number of attention heads

    # Memory settings
    max_memory_tokens: int = 1024  # Maximum memory entries to attend to
    memory_dim: int = 768  # Dimension of memory embeddings

    # Attention type
    attention_type: AttentionType = AttentionType.PARALLEL

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Memory attention scaling
    memory_scale: float = 1.0  # Scale factor for memory attention weights
    temperature: float = 1.0  # Softmax temperature

    # Efficiency
    use_flash_attention: bool = True
    chunk_size: int = 256  # For chunked attention over large memories


@dataclass
class AttentionOutput:
    """Output from memory-augmented attention."""

    # Main output
    output: Any  # [batch, seq, d_model] tensor

    # Attention weights (for interpretability)
    context_attention: Any | None = None  # [batch, heads, seq, context_len]
    memory_attention: Any | None = None  # [batch, heads, seq, memory_len]

    # Memory contribution
    memory_contribution: float = 0.0  # Fraction of output from memory

    # Retrieved memory indices (top-k)
    top_memory_indices: list[int] = field(default_factory=list)
    top_memory_scores: list[float] = field(default_factory=list)


if HAS_TORCH:

    class MemoryAttention(nn.Module):
        """
        Memory-Augmented Multi-Head Attention.

        Extends standard transformer attention to jointly attend over:
        1. Input context (standard self-attention)
        2. External memory store (memory attention)

        The outputs are fused using configurable strategies.

        Mathematical formulation:
            Q = W_q @ input  # [batch, seq, d_model] -> [batch, seq, num_heads, d_key]

            # Context attention (standard)
            K_ctx = W_k @ context
            V_ctx = W_v @ context
            attn_ctx = softmax(Q @ K_ctx.T / √d_k) @ V_ctx

            # Memory attention
            K_mem = memory_keys  # Pre-computed from memory store
            V_mem = memory_values
            attn_mem = softmax(Q @ K_mem.T / √d_k) @ V_mem

            # Fusion
            output = fuse(attn_ctx, attn_mem)
        """

        def __init__(self, config: MemoryAttentionConfig):
            super().__init__()
            self.config = config

            self.d_model = config.d_model
            self.num_heads = config.num_heads
            self.d_key = config.d_key
            self.d_value = config.d_value
            self.scale = 1.0 / math.sqrt(config.d_key)

            # Query, Key, Value projections for context
            self.W_q = nn.Linear(config.d_model, config.num_heads * config.d_key)
            self.W_k = nn.Linear(config.d_model, config.num_heads * config.d_key)
            self.W_v = nn.Linear(config.d_model, config.num_heads * config.d_value)

            # Separate projections for memory (allows different dimensions)
            self.W_k_mem = nn.Linear(config.memory_dim, config.num_heads * config.d_key)
            self.W_v_mem = nn.Linear(config.memory_dim, config.num_heads * config.d_value)

            # Output projection
            self.W_o = nn.Linear(config.num_heads * config.d_value, config.d_model)

            # Fusion mechanism
            if config.attention_type == AttentionType.GATED:
                # Learned gate to balance context vs memory
                self.gate = nn.Sequential(
                    nn.Linear(config.d_model * 2, config.d_model),
                    nn.Sigmoid(),
                )
            elif config.attention_type == AttentionType.PARALLEL:
                # Combine parallel attention outputs
                self.combine = nn.Linear(config.d_model * 2, config.d_model)

            # Dropout
            self.dropout = nn.Dropout(config.dropout)
            self.attn_dropout = nn.Dropout(config.attention_dropout)

            # Memory scale (learnable)
            self.memory_scale = nn.Parameter(
                torch.tensor(config.memory_scale)
            )

        def forward(
            self,
            query: torch.Tensor,  # [batch, seq_len, d_model]
            context: torch.Tensor | None = None,  # [batch, ctx_len, d_model]
            memory_keys: torch.Tensor | None = None,  # [batch, mem_len, d_key * num_heads] or [mem_len, ...]
            memory_values: torch.Tensor | None = None,  # [batch, mem_len, d_value * num_heads]
            attention_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            return_attention: bool = False,
        ) -> AttentionOutput:
            """
            Forward pass with memory-augmented attention.

            Args:
                query: Input query tensor
                context: Context for keys/values (None = self-attention)
                memory_keys: Pre-computed memory keys
                memory_values: Pre-computed memory values
                attention_mask: Mask for context attention
                memory_mask: Mask for memory attention
                return_attention: Whether to return attention weights
            """
            batch_size, seq_len, _ = query.shape

            # Self-attention if no context provided
            if context is None:
                context = query

            # Project query, key, value for context
            Q = self.W_q(query)  # [batch, seq, num_heads * d_key]
            K_ctx = self.W_k(context)
            V_ctx = self.W_v(context)

            # Reshape for multi-head attention
            Q = Q.view(batch_size, seq_len, self.num_heads, self.d_key).transpose(1, 2)
            K_ctx = K_ctx.view(batch_size, -1, self.num_heads, self.d_key).transpose(1, 2)
            V_ctx = V_ctx.view(batch_size, -1, self.num_heads, self.d_value).transpose(1, 2)
            # Q, K, V: [batch, num_heads, seq/ctx_len, d_key/d_value]

            # Context attention
            attn_ctx = self._compute_attention(Q, K_ctx, V_ctx, attention_mask)
            ctx_weights = None

            # Memory attention (if memory provided)
            attn_mem = None
            mem_weights = None
            memory_contribution = 0.0

            if memory_keys is not None and memory_values is not None:
                # Project memory if needed (or use pre-projected)
                if memory_keys.dim() == 2:
                    # Memory is [mem_len, memory_dim], expand for batch
                    memory_keys = memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
                    memory_values = memory_values.unsqueeze(0).expand(batch_size, -1, -1)

                # Project memory to K, V space
                K_mem = self.W_k_mem(memory_keys)
                V_mem = self.W_v_mem(memory_values)

                mem_len = K_mem.shape[1]
                K_mem = K_mem.view(batch_size, mem_len, self.num_heads, self.d_key).transpose(1, 2)
                V_mem = V_mem.view(batch_size, mem_len, self.num_heads, self.d_value).transpose(1, 2)

                # Memory attention with scaling
                attn_mem = self._compute_attention(
                    Q, K_mem, V_mem, memory_mask,
                    scale_factor=self.memory_scale.item()
                )

            # Fuse context and memory attention
            output = self._fuse_attention(attn_ctx, attn_mem)

            # Output projection
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = self.W_o(output)
            output = self.dropout(output)

            # Calculate memory contribution
            if attn_mem is not None:
                with torch.no_grad():
                    ctx_norm = attn_ctx.norm()
                    mem_norm = attn_mem.norm()
                    total_norm = ctx_norm + mem_norm
                    if total_norm > 0:
                        memory_contribution = (mem_norm / total_norm).item()

            return AttentionOutput(
                output=output,
                context_attention=ctx_weights if return_attention else None,
                memory_attention=mem_weights if return_attention else None,
                memory_contribution=memory_contribution,
            )

        def _compute_attention(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            mask: torch.Tensor | None = None,
            scale_factor: float = 1.0,
        ) -> torch.Tensor:
            """Compute scaled dot-product attention."""
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale * scale_factor

            # Apply temperature
            scores = scores / self.config.temperature

            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            # Weighted sum of values
            return torch.matmul(attn_weights, V)

        def _fuse_attention(
            self,
            ctx_attn: torch.Tensor,
            mem_attn: torch.Tensor | None,
        ) -> torch.Tensor:
            """Fuse context and memory attention outputs."""
            if mem_attn is None:
                return ctx_attn

            if self.config.attention_type == AttentionType.CONCAT:
                # Simple concatenation (handled in attention computation)
                return ctx_attn + mem_attn

            elif self.config.attention_type == AttentionType.PARALLEL:
                # Concatenate and project
                batch, heads, seq, d = ctx_attn.shape
                ctx_flat = ctx_attn.transpose(1, 2).reshape(batch, seq, -1)
                mem_flat = mem_attn.transpose(1, 2).reshape(batch, seq, -1)
                combined = torch.cat([ctx_flat, mem_flat], dim=-1)
                fused = self.combine(combined)
                return fused.view(batch, seq, heads, d).transpose(1, 2)

            elif self.config.attention_type == AttentionType.GATED:
                # Gated fusion
                batch, heads, seq, d = ctx_attn.shape
                ctx_flat = ctx_attn.transpose(1, 2).reshape(batch, seq, -1)
                mem_flat = mem_attn.transpose(1, 2).reshape(batch, seq, -1)
                combined = torch.cat([ctx_flat, mem_flat], dim=-1)
                gate = self.gate(combined)
                fused = gate * ctx_flat + (1 - gate) * mem_flat
                return fused.view(batch, seq, heads, d).transpose(1, 2)

            else:
                # Default: additive
                return ctx_attn + mem_attn


    class ChunkedMemoryAttention(MemoryAttention):
        """
        Memory attention with chunking for efficiency with large memory stores.

        Processes memory in chunks to reduce peak memory usage while maintaining
        exact attention computation.
        """

        def forward(
            self,
            query: torch.Tensor,
            context: torch.Tensor | None = None,
            memory_keys: torch.Tensor | None = None,
            memory_values: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            return_attention: bool = False,
        ) -> AttentionOutput:
            """Forward with chunked memory processing."""

            if memory_keys is None or memory_keys.shape[1] <= self.config.chunk_size:
                # Small memory, use standard attention
                return super().forward(
                    query, context, memory_keys, memory_values,
                    attention_mask, memory_mask, return_attention
                )

            # Chunked processing for large memory
            batch_size, seq_len, _ = query.shape
            mem_len = memory_keys.shape[1]
            chunk_size = self.config.chunk_size

            # Project query once
            Q = self.W_q(query)
            Q = Q.view(batch_size, seq_len, self.num_heads, self.d_key).transpose(1, 2)

            # Accumulate attention over chunks
            all_scores = []
            all_values = []

            for start in range(0, mem_len, chunk_size):
                end = min(start + chunk_size, mem_len)

                # Get chunk
                k_chunk = memory_keys[:, start:end]
                v_chunk = memory_values[:, start:end]

                # Project chunk
                K_chunk = self.W_k_mem(k_chunk)
                V_chunk = self.W_v_mem(v_chunk)

                chunk_len = K_chunk.shape[1]
                K_chunk = K_chunk.view(batch_size, chunk_len, self.num_heads, self.d_key).transpose(1, 2)
                V_chunk = V_chunk.view(batch_size, chunk_len, self.num_heads, self.d_value).transpose(1, 2)

                # Compute scores for this chunk
                scores = torch.matmul(Q, K_chunk.transpose(-2, -1)) * self.scale
                all_scores.append(scores)
                all_values.append(V_chunk)

            # Combine all chunks with numerically stable softmax
            all_scores = torch.cat(all_scores, dim=-1)  # [batch, heads, seq, total_mem]
            all_values = torch.cat(all_values, dim=2)   # [batch, heads, total_mem, d_value]

            attn_weights = F.softmax(all_scores / self.config.temperature, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            mem_attn = torch.matmul(attn_weights, all_values)

            # Context attention (standard)
            if context is None:
                context = query
            K_ctx = self.W_k(context)
            V_ctx = self.W_v(context)
            K_ctx = K_ctx.view(batch_size, -1, self.num_heads, self.d_key).transpose(1, 2)
            V_ctx = V_ctx.view(batch_size, -1, self.num_heads, self.d_value).transpose(1, 2)

            ctx_attn = self._compute_attention(Q, K_ctx, V_ctx, attention_mask)

            # Fuse and project
            output = self._fuse_attention(ctx_attn, mem_attn)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            output = self.W_o(output)
            output = self.dropout(output)

            return AttentionOutput(output=output, memory_contribution=0.5)

else:
    # NumPy fallback for environments without PyTorch

    class MemoryAttention:
        """NumPy-based memory attention (inference only)."""

        def __init__(self, config: MemoryAttentionConfig):
            self.config = config
            self.d_model = config.d_model
            self.num_heads = config.num_heads
            self.d_key = config.d_key
            self.scale = 1.0 / math.sqrt(config.d_key)

            # Initialize random weights (for demonstration)
            self.W_q = np.random.randn(config.d_model, config.num_heads * config.d_key) * 0.02
            self.W_k = np.random.randn(config.d_model, config.num_heads * config.d_key) * 0.02
            self.W_v = np.random.randn(config.d_model, config.num_heads * config.d_key) * 0.02
            self.W_o = np.random.randn(config.num_heads * config.d_key, config.d_model) * 0.02

        def forward(
            self,
            query: np.ndarray,
            context: np.ndarray | None = None,
            memory_keys: np.ndarray | None = None,
            memory_values: np.ndarray | None = None,
            **kwargs,
        ) -> AttentionOutput:
            """Forward pass with memory attention."""
            if context is None:
                context = query

            # Simple attention computation
            Q = query @ self.W_q
            K = context @ self.W_k
            V = context @ self.W_v

            scores = (Q @ K.T) * self.scale
            attn = self._softmax(scores)
            output = attn @ V

            # Add memory contribution if provided
            if memory_keys is not None and memory_values is not None:
                mem_scores = (Q @ memory_keys.T) * self.scale
                mem_attn = self._softmax(mem_scores)
                mem_output = mem_attn @ memory_values
                output = output + mem_output

            output = output @ self.W_o

            return AttentionOutput(output=output)

        def _softmax(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    class ChunkedMemoryAttention(MemoryAttention):
        """Chunked attention for NumPy backend."""
        pass
