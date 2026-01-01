"""
Attention Fusion Strategies for Memory-Augmented Transformers.

Provides multiple strategies for combining context and memory attention:
- Concatenation: Simple concat of K,V spaces
- Gated: Learned gating between context and memory
- Cross-Attention: Separate cross-attention to memory
- Hierarchical: Attend to memory summaries first, then details
- Mixture-of-Experts: Route queries to context or memory

Mathematical formulation:
    Given:
        A_ctx = Attention(Q, K_ctx, V_ctx)  # Context attention output
        A_mem = Attention(Q, K_mem, V_mem)  # Memory attention output

    Fusion strategies:
        Concat:       output = W_o @ [A_ctx; A_mem]
        Gated:        g = σ(W_g @ [A_ctx; A_mem])
                      output = g * A_ctx + (1-g) * A_mem
        Cross:        output = A_ctx + CrossAttn(A_ctx, K_mem, V_mem)
        Hierarchical: summary = Attn(Q, K_summary, V_summary)
                      detail = Attn(Q, K_mem, V_mem, mask=TopK(summary))
                      output = A_ctx + detail
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FusionStrategy(Enum):
    """Available fusion strategies."""

    CONCAT = "concat"  # Concatenate and project
    ADDITIVE = "additive"  # Simple addition
    GATED = "gated"  # Learned gating
    CROSS = "cross"  # Cross-attention fusion
    HIERARCHICAL = "hierarchical"  # Hierarchical attention
    MOE = "moe"  # Mixture of experts routing


@dataclass
class FusionConfig:
    """Configuration for attention fusion."""

    d_model: int = 768
    num_heads: int = 12
    d_key: int = 64
    d_value: int = 64

    # Gated fusion
    gate_activation: str = "sigmoid"  # sigmoid, softmax, tanh

    # Hierarchical fusion
    num_summaries: int = 32  # Number of memory summary vectors
    topk_details: int = 64  # Top-k detailed memories to attend

    # MoE fusion
    num_experts: int = 4
    expert_capacity: float = 1.25

    # Regularization
    dropout: float = 0.1


if HAS_TORCH:

    class AttentionFusion(nn.Module, ABC):
        """Base class for attention fusion strategies."""

        def __init__(self, config: FusionConfig):
            super().__init__()
            self.config = config

        @abstractmethod
        def forward(
            self,
            context_attn: torch.Tensor,  # [batch, heads, seq, d_value]
            memory_attn: torch.Tensor,  # [batch, heads, seq, d_value]
            context_weights: torch.Tensor | None = None,
            memory_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Fuse context and memory attention outputs."""
            pass

    class ConcatFusion(AttentionFusion):
        """
        Concatenation fusion: concat attention outputs and project.

        output = W_o @ [A_ctx; A_mem]

        Simple and effective, doubles the intermediate dimension.
        """

        def __init__(self, config: FusionConfig):
            super().__init__(config)

            d_head = config.d_value
            self.projection = nn.Linear(
                config.num_heads * d_head * 2,
                config.num_heads * d_head,
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            context_attn: torch.Tensor,
            memory_attn: torch.Tensor,
            context_weights: torch.Tensor | None = None,
            memory_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, heads, seq, d = context_attn.shape

            # Flatten heads
            ctx_flat = context_attn.transpose(1, 2).reshape(batch, seq, -1)
            mem_flat = memory_attn.transpose(1, 2).reshape(batch, seq, -1)

            # Concatenate and project
            combined = torch.cat([ctx_flat, mem_flat], dim=-1)
            output = self.projection(combined)
            output = self.dropout(output)

            # Reshape back
            return output.view(batch, seq, heads, d).transpose(1, 2)

    class GatedFusion(AttentionFusion):
        """
        Gated fusion: learned gate balances context and memory.

        g = σ(W_g @ [A_ctx; A_mem] + b_g)
        output = g * A_ctx + (1 - g) * A_mem

        Learns when to rely on context vs memory.
        """

        def __init__(self, config: FusionConfig):
            super().__init__(config)

            d_head = config.d_value
            d_total = config.num_heads * d_head

            # Gate network
            self.gate_net = nn.Sequential(
                nn.Linear(d_total * 2, d_total),
                nn.LayerNorm(d_total),
                nn.ReLU(),
                nn.Linear(d_total, d_total),
            )

            # Activation
            if config.gate_activation == "sigmoid":
                self.gate_activation = nn.Sigmoid()
            elif config.gate_activation == "softmax":
                self.gate_activation = None  # Handle separately
            else:
                self.gate_activation = nn.Tanh()

            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            context_attn: torch.Tensor,
            memory_attn: torch.Tensor,
            context_weights: torch.Tensor | None = None,
            memory_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, heads, seq, d = context_attn.shape

            # Flatten
            ctx_flat = context_attn.transpose(1, 2).reshape(batch, seq, -1)
            mem_flat = memory_attn.transpose(1, 2).reshape(batch, seq, -1)

            # Compute gate
            combined = torch.cat([ctx_flat, mem_flat], dim=-1)
            gate_logits = self.gate_net(combined)

            if self.gate_activation is not None:
                gate = self.gate_activation(gate_logits)
            else:
                # Softmax over 2 options (context vs memory)
                gate_stack = torch.stack([gate_logits, -gate_logits], dim=-1)
                gate = F.softmax(gate_stack, dim=-1)[..., 0]

            # Apply gate
            output = gate * ctx_flat + (1 - gate) * mem_flat
            output = self.dropout(output)

            return output.view(batch, seq, heads, d).transpose(1, 2)

    class CrossAttentionFusion(AttentionFusion):
        """
        Cross-attention fusion: context attends to memory output.

        intermediate = CrossAttn(A_ctx, A_mem, A_mem)
        output = A_ctx + intermediate

        Allows context to selectively incorporate memory information.
        """

        def __init__(self, config: FusionConfig):
            super().__init__(config)

            d_head = config.d_value

            # Cross-attention projections
            self.W_q = nn.Linear(config.num_heads * d_head, config.num_heads * d_head)
            self.W_k = nn.Linear(config.num_heads * d_head, config.num_heads * d_head)
            self.W_v = nn.Linear(config.num_heads * d_head, config.num_heads * d_head)
            self.W_o = nn.Linear(config.num_heads * d_head, config.num_heads * d_head)

            self.scale = 1.0 / math.sqrt(d_head)
            self.dropout = nn.Dropout(config.dropout)
            self.attn_dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            context_attn: torch.Tensor,
            memory_attn: torch.Tensor,
            context_weights: torch.Tensor | None = None,
            memory_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, heads, seq, d = context_attn.shape

            # Flatten
            ctx_flat = context_attn.transpose(1, 2).reshape(batch, seq, -1)
            mem_flat = memory_attn.transpose(1, 2).reshape(batch, seq, -1)

            # Cross-attention: context queries memory
            Q = self.W_q(ctx_flat).view(batch, seq, heads, d).transpose(1, 2)
            K = self.W_k(mem_flat).view(batch, seq, heads, d).transpose(1, 2)
            V = self.W_v(mem_flat).view(batch, seq, heads, d).transpose(1, 2)

            # Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)

            cross_output = torch.matmul(attn, V)

            # Project and residual
            cross_flat = cross_output.transpose(1, 2).reshape(batch, seq, -1)
            output = ctx_flat + self.W_o(cross_flat)
            output = self.dropout(output)

            return output.view(batch, seq, heads, d).transpose(1, 2)

    class HierarchicalFusion(AttentionFusion):
        """
        Hierarchical fusion: attend to summaries, then details.

        1. Compute summary attention: α_summary = softmax(Q @ K_summary.T)
        2. Select top-k memories based on summary relevance
        3. Attend to selected detailed memories

        Efficient for large memory stores.
        """

        def __init__(self, config: FusionConfig):
            super().__init__(config)

            d_head = config.d_value
            d_total = config.num_heads * d_head

            # Summary projection
            self.summary_proj = nn.Linear(d_total, config.num_summaries)

            # Detail selection
            self.detail_gate = nn.Linear(config.num_summaries, d_total)

            self.dropout = nn.Dropout(config.dropout)
            self.topk = config.topk_details

        def forward(
            self,
            context_attn: torch.Tensor,
            memory_attn: torch.Tensor,
            context_weights: torch.Tensor | None = None,
            memory_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, heads, seq, d = context_attn.shape

            # Flatten
            ctx_flat = context_attn.transpose(1, 2).reshape(batch, seq, -1)
            mem_flat = memory_attn.transpose(1, 2).reshape(batch, seq, -1)

            # Compute summary relevance
            summary_scores = self.summary_proj(mem_flat)  # [batch, seq, num_summaries]
            summary_weights = F.softmax(summary_scores, dim=-1)

            # Gate memory based on summary
            gate = torch.sigmoid(self.detail_gate(summary_weights))
            gated_memory = gate * mem_flat

            # Combine with context
            output = ctx_flat + gated_memory
            output = self.dropout(output)

            return output.view(batch, seq, heads, d).transpose(1, 2)

    class MixtureOfExpertsFusion(AttentionFusion):
        """
        Mixture of Experts fusion: route queries to specialized experts.

        Each expert specializes in different types of context-memory fusion.
        Router learns to dispatch queries to appropriate experts.

        Experts:
        - Context-focused: Prioritizes context attention
        - Memory-focused: Prioritizes memory attention
        - Balanced: Equal weighting
        - Cross-modal: Cross-attention based fusion
        """

        def __init__(self, config: FusionConfig):
            super().__init__(config)

            d_head = config.d_value
            d_total = config.num_heads * d_head
            num_experts = config.num_experts

            # Router
            self.router = nn.Linear(d_total * 2, num_experts)

            # Experts (each is a simple fusion function)
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_total * 2, d_total),
                        nn.ReLU(),
                        nn.Linear(d_total, d_total),
                    )
                    for _ in range(num_experts)
                ]
            )

            # Expert biases for specialization
            self.expert_biases = nn.Parameter(torch.randn(num_experts, 2))
            # bias[:, 0] = context weight, bias[:, 1] = memory weight

            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            context_attn: torch.Tensor,
            memory_attn: torch.Tensor,
            context_weights: torch.Tensor | None = None,
            memory_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, heads, seq, d = context_attn.shape

            # Flatten
            ctx_flat = context_attn.transpose(1, 2).reshape(batch, seq, -1)
            mem_flat = memory_attn.transpose(1, 2).reshape(batch, seq, -1)
            combined = torch.cat([ctx_flat, mem_flat], dim=-1)

            # Compute routing probabilities
            router_logits = self.router(combined)
            router_probs = F.softmax(router_logits, dim=-1)  # [batch, seq, num_experts]

            # Apply each expert
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                # Apply expert bias to inputs
                bias = F.softmax(self.expert_biases[i], dim=0)
                biased_input = bias[0] * ctx_flat + bias[1] * mem_flat

                # Expert forward
                exp_combined = torch.cat([biased_input, combined[:, :, combined.shape[-1] // 2 :]], dim=-1)
                expert_out = expert(exp_combined)
                expert_outputs.append(expert_out)

            # Stack and weight by router probabilities
            expert_stack = torch.stack(expert_outputs, dim=-1)  # [batch, seq, d_total, num_experts]
            output = torch.sum(expert_stack * router_probs.unsqueeze(-2), dim=-1)
            output = self.dropout(output)

            return output.view(batch, seq, heads, d).transpose(1, 2)

    class AdaptiveFusion(AttentionFusion):
        """
        Adaptive fusion: dynamically selects fusion strategy per token.

        Learns to choose between:
        - Pure context (for tokens not needing memory)
        - Pure memory (for tokens requiring memory lookup)
        - Gated blend (for nuanced combination)
        """

        def __init__(self, config: FusionConfig):
            super().__init__(config)

            d_head = config.d_value
            d_total = config.num_heads * d_head

            # Strategy selector
            self.strategy_net = nn.Sequential(
                nn.Linear(d_total * 2, d_total),
                nn.LayerNorm(d_total),
                nn.ReLU(),
                nn.Linear(d_total, 3),  # 3 strategies
            )

            # Strategy-specific parameters
            self.blend_gate = nn.Linear(d_total * 2, d_total)

            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            context_attn: torch.Tensor,
            memory_attn: torch.Tensor,
            context_weights: torch.Tensor | None = None,
            memory_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch, heads, seq, d = context_attn.shape

            # Flatten
            ctx_flat = context_attn.transpose(1, 2).reshape(batch, seq, -1)
            mem_flat = memory_attn.transpose(1, 2).reshape(batch, seq, -1)
            combined = torch.cat([ctx_flat, mem_flat], dim=-1)

            # Select strategy per token
            strategy_logits = self.strategy_net(combined)
            strategy_probs = F.softmax(strategy_logits, dim=-1)  # [batch, seq, 3]

            # Strategy 0: Pure context
            # Strategy 1: Pure memory
            # Strategy 2: Gated blend
            gate = torch.sigmoid(self.blend_gate(combined))

            pure_ctx = ctx_flat
            pure_mem = mem_flat
            blended = gate * ctx_flat + (1 - gate) * mem_flat

            # Weighted combination of strategies
            output = (
                strategy_probs[..., 0:1] * pure_ctx
                + strategy_probs[..., 1:2] * pure_mem
                + strategy_probs[..., 2:3] * blended
            )
            output = self.dropout(output)

            return output.view(batch, seq, heads, d).transpose(1, 2)

    def create_fusion(strategy: FusionStrategy, config: FusionConfig) -> AttentionFusion:
        """Create a fusion module for the given strategy."""
        if strategy == FusionStrategy.CONCAT:
            return ConcatFusion(config)
        elif strategy == FusionStrategy.GATED:
            return GatedFusion(config)
        elif strategy == FusionStrategy.CROSS:
            return CrossAttentionFusion(config)
        elif strategy == FusionStrategy.HIERARCHICAL:
            return HierarchicalFusion(config)
        elif strategy == FusionStrategy.MOE:
            return MixtureOfExpertsFusion(config)
        else:
            return GatedFusion(config)  # Default

else:
    # Stub classes for non-PyTorch environments

    class AttentionFusion:
        """Base fusion class (stub)."""

        def __init__(self, config: FusionConfig):
            self.config = config

        def forward(self, context_attn, memory_attn, **kwargs):
            return context_attn + memory_attn

    class ConcatFusion(AttentionFusion):
        pass

    class GatedFusion(AttentionFusion):
        pass

    class CrossAttentionFusion(AttentionFusion):
        pass

    class HierarchicalFusion(AttentionFusion):
        pass

    class MixtureOfExpertsFusion(AttentionFusion):
        pass

    class AdaptiveFusion(AttentionFusion):
        pass

    def create_fusion(strategy: FusionStrategy, config: FusionConfig) -> AttentionFusion:
        return AttentionFusion(config)
