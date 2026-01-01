"""
Memory-Augmented Transformer Layers.

Provides drop-in replacements for standard transformer layers
that include memory attention capabilities.

Can be used to:
1. Replace layers in existing models (e.g., HuggingFace transformers)
2. Build new memory-augmented architectures from scratch
3. Add memory to specific layers (e.g., only middle layers)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .cache import MemoryKeyValueCache
from .core import AttentionOutput, MemoryAttention, MemoryAttentionConfig
from .fusion import FusionStrategy


@dataclass
class LayerConfig:
    """Configuration for memory-augmented layer."""

    # Model dimensions
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072  # Feed-forward dimension

    # Memory attention
    memory_attention: MemoryAttentionConfig | None = None

    # Fusion
    fusion_strategy: FusionStrategy = FusionStrategy.GATED

    # Layer options
    pre_norm: bool = True  # Pre-LayerNorm (like GPT-2) vs Post-LayerNorm
    use_memory_in_ff: bool = False  # Also attend to memory in FF block

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1


if HAS_TORCH:

    class MemoryAugmentedLayer(nn.Module):
        """
        A single transformer layer with memory attention.

        Structure (pre-norm):
            x = x + MemoryAttention(LayerNorm(x), memory)
            x = x + FeedForward(LayerNorm(x))

        Drop-in replacement for standard transformer layers.
        """

        def __init__(self, config: LayerConfig):
            super().__init__()
            self.config = config

            # Memory attention configuration
            mem_config = config.memory_attention or MemoryAttentionConfig(
                d_model=config.d_model,
                num_heads=config.num_heads,
            )

            # Self-attention with memory
            self.attention = MemoryAttention(mem_config)

            # Feed-forward network
            self.ff = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout),
            )

            # Layer norms
            self.norm1 = nn.LayerNorm(config.d_model)
            self.norm2 = nn.LayerNorm(config.d_model)

            # Optional memory attention in FF
            if config.use_memory_in_ff:
                self.ff_memory_attn = MemoryAttention(mem_config)
                self.norm_ff_mem = nn.LayerNorm(config.d_model)

            self.pre_norm = config.pre_norm

        def forward(
            self,
            x: torch.Tensor,  # [batch, seq, d_model]
            memory_keys: torch.Tensor | None = None,
            memory_values: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            return_attention: bool = False,
        ) -> tuple[torch.Tensor, AttentionOutput | None]:
            """
            Forward pass through memory-augmented layer.

            Args:
                x: Input tensor
                memory_keys: Pre-computed memory keys
                memory_values: Pre-computed memory values
                attention_mask: Mask for self-attention
                memory_mask: Mask for memory attention
                return_attention: Whether to return attention weights

            Returns:
                (output, attention_output) tuple
            """
            # Self-attention with memory
            if self.pre_norm:
                normed = self.norm1(x)
                attn_out = self.attention(
                    normed, normed,
                    memory_keys=memory_keys,
                    memory_values=memory_values,
                    attention_mask=attention_mask,
                    memory_mask=memory_mask,
                    return_attention=return_attention,
                )
                x = x + attn_out.output
            else:
                attn_out = self.attention(
                    x, x,
                    memory_keys=memory_keys,
                    memory_values=memory_values,
                    attention_mask=attention_mask,
                    memory_mask=memory_mask,
                    return_attention=return_attention,
                )
                x = self.norm1(x + attn_out.output)

            # Feed-forward
            if self.pre_norm:
                ff_out = self.ff(self.norm2(x))
                x = x + ff_out
            else:
                ff_out = self.ff(x)
                x = self.norm2(x + ff_out)

            # Optional: memory attention in FF pathway
            if hasattr(self, 'ff_memory_attn') and memory_keys is not None:
                if self.pre_norm:
                    mem_ff_out = self.ff_memory_attn(
                        self.norm_ff_mem(x), None,
                        memory_keys=memory_keys,
                        memory_values=memory_values,
                    )
                    x = x + mem_ff_out.output * 0.1  # Small contribution

            return x, attn_out if return_attention else None


    class MemoryAugmentedTransformer(nn.Module):
        """
        Full memory-augmented transformer model.

        Stacks multiple MemoryAugmentedLayers and provides:
        - Embedding layer
        - Positional encoding
        - Memory cache management
        - Output projection
        """

        def __init__(
            self,
            num_layers: int = 12,
            d_model: int = 768,
            num_heads: int = 12,
            d_ff: int = 3072,
            vocab_size: int = 50257,
            max_seq_len: int = 2048,
            memory_layers: list[int] | None = None,  # Which layers get memory
            fusion_strategy: FusionStrategy = FusionStrategy.GATED,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.d_model = d_model
            self.num_layers = num_layers

            # Embeddings
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)

            # Layers
            self.layers = nn.ModuleList()
            memory_layers = memory_layers or list(range(num_layers // 3, 2 * num_layers // 3))

            for i in range(num_layers):
                layer_config = LayerConfig(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    fusion_strategy=fusion_strategy,
                    dropout=dropout,
                )
                if i in memory_layers:
                    layer_config.memory_attention = MemoryAttentionConfig(
                        d_model=d_model,
                        num_heads=num_heads,
                    )
                self.layers.append(MemoryAugmentedLayer(layer_config))

            # Final layer norm
            self.final_norm = nn.LayerNorm(d_model)

            # Output projection (optional, for language modeling)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            # Memory indices (which layers have memory)
            self.memory_layers = memory_layers

            # Memory cache
            self._memory_cache: MemoryKeyValueCache | None = None

        def set_memory_cache(self, cache: MemoryKeyValueCache) -> None:
            """Set the memory cache for attention."""
            self._memory_cache = cache

        def forward(
            self,
            input_ids: torch.Tensor,  # [batch, seq_len]
            attention_mask: torch.Tensor | None = None,
            memory_keys: torch.Tensor | None = None,
            memory_values: torch.Tensor | None = None,
            return_hidden_states: bool = False,
            return_attentions: bool = False,
        ) -> dict[str, Any]:
            """
            Forward pass through the transformer.

            Args:
                input_ids: Input token IDs
                attention_mask: Attention mask for padding
                memory_keys: Optional pre-computed memory keys
                memory_values: Optional pre-computed memory values
                return_hidden_states: Whether to return all hidden states
                return_attentions: Whether to return attention weights

            Returns:
                Dictionary with 'logits', 'hidden_states', 'attentions'
            """
            batch_size, seq_len = input_ids.shape

            # Embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + self.position_embedding(positions)

            # Get memory from cache if not provided
            if memory_keys is None and self._memory_cache is not None:
                memory_keys, memory_values = self._memory_cache.get_all()
                if memory_keys is not None:
                    # Ensure proper shape
                    if memory_keys.dim() == 2:
                        memory_keys = memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
                        memory_values = memory_values.unsqueeze(0).expand(batch_size, -1, -1)

            # Process through layers
            hidden_states = [x] if return_hidden_states else []
            attentions = [] if return_attentions else []

            for i, layer in enumerate(self.layers):
                # Only pass memory to designated layers
                if i in self.memory_layers:
                    x, attn = layer(
                        x,
                        memory_keys=memory_keys,
                        memory_values=memory_values,
                        attention_mask=attention_mask,
                        return_attention=return_attentions,
                    )
                else:
                    x, attn = layer(
                        x,
                        attention_mask=attention_mask,
                        return_attention=return_attentions,
                    )

                if return_hidden_states:
                    hidden_states.append(x)
                if return_attentions and attn:
                    attentions.append(attn)

            # Final norm and output
            x = self.final_norm(x)
            logits = self.lm_head(x)

            return {
                "logits": logits,
                "hidden_states": hidden_states if return_hidden_states else None,
                "attentions": attentions if return_attentions else None,
                "last_hidden_state": x,
            }

        def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
        ) -> torch.Tensor:
            """Generate tokens autoregressively."""
            for _ in range(max_new_tokens):
                # Truncate to max sequence length
                idx_cond = input_ids[:, -2048:]

                # Forward pass
                outputs = self.forward(idx_cond)
                logits = outputs["logits"][:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

            return input_ids


    class MemoryInjector:
        """
        Utility to inject memory attention into existing models.

        Allows adding memory capabilities to pre-trained transformers
        without full retraining.

        Usage:
            model = AutoModel.from_pretrained("gpt2")
            injector = MemoryInjector(memory_cache)
            injector.inject(model, layers=[6, 7, 8, 9, 10, 11])
        """

        def __init__(
            self,
            memory_cache: MemoryKeyValueCache,
            fusion_strategy: FusionStrategy = FusionStrategy.GATED,
        ):
            self.memory_cache = memory_cache
            self.fusion_strategy = fusion_strategy
            self._original_forwards = {}

        def inject(
            self,
            model: nn.Module,
            layers: list[int] | None = None,
            attention_module_name: str = "attn",
        ) -> None:
            """
            Inject memory attention into specified layers.

            Args:
                model: The transformer model to modify
                layers: Which layer indices to modify (None = all)
                attention_module_name: Name of attention module in each layer
            """
            # Find transformer layers
            transformer_layers = self._find_layers(model)

            if layers is None:
                # Default: middle third of layers
                n = len(transformer_layers)
                layers = list(range(n // 3, 2 * n // 3))

            for idx in layers:
                if idx < len(transformer_layers):
                    layer = transformer_layers[idx]
                    self._inject_layer(layer, idx, attention_module_name)

        def _find_layers(self, model: nn.Module) -> list[nn.Module]:
            """Find transformer layers in model."""
            # Try common architectures
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                return list(model.transformer.h)
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                return list(model.encoder.layer)
            if hasattr(model, 'layers'):
                return list(model.layers)
            if hasattr(model, 'h'):
                return list(model.h)

            # Search recursively
            for name, child in model.named_children():
                if 'layer' in name.lower() or 'block' in name.lower():
                    if isinstance(child, nn.ModuleList):
                        return list(child)

            return []

        def _inject_layer(
            self,
            layer: nn.Module,
            layer_idx: int,
            attention_module_name: str,
        ) -> None:
            """Inject memory attention into a single layer."""
            # Find attention module
            if hasattr(layer, attention_module_name):
                attn = getattr(layer, attention_module_name)
            elif hasattr(layer, 'self_attn'):
                attn = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn = layer.attention
            else:
                return

            # Store original forward
            original_forward = attn.forward
            self._original_forwards[layer_idx] = original_forward

            # Get memory
            cache = self.memory_cache

            # Create wrapper
            def memory_forward(*args, **kwargs):
                # Call original attention
                output = original_forward(*args, **kwargs)

                # Add memory attention
                keys, values = cache.get_all()
                if keys is not None:
                    # Simple additive memory (could be more sophisticated)
                    hidden = output[0] if isinstance(output, tuple) else output

                    # Query memory
                    batch_size = hidden.shape[0]
                    if keys.dim() == 2:
                        keys = keys.unsqueeze(0).expand(batch_size, -1, -1)
                        values = values.unsqueeze(0).expand(batch_size, -1, -1)

                    # Simple attention to memory
                    mem_attn = torch.matmul(hidden, keys.transpose(-2, -1))
                    mem_attn = F.softmax(mem_attn / (hidden.shape[-1] ** 0.5), dim=-1)
                    mem_out = torch.matmul(mem_attn, values)

                    # Add to output (gated)
                    gate = 0.1  # Small initial contribution
                    if isinstance(output, tuple):
                        return (output[0] + gate * mem_out,) + output[1:]
                    else:
                        return output + gate * mem_out

                return output

            # Replace forward
            attn.forward = memory_forward

        def remove(self, model: nn.Module) -> None:
            """Remove injected memory attention."""
            transformer_layers = self._find_layers(model)

            for idx, original_forward in self._original_forwards.items():
                if idx < len(transformer_layers):
                    layer = transformer_layers[idx]
                    for name in ['attn', 'self_attn', 'attention']:
                        if hasattr(layer, name):
                            getattr(layer, name).forward = original_forward
                            break

            self._original_forwards.clear()

else:
    # Stubs for non-PyTorch environments

    class MemoryAugmentedLayer:
        def __init__(self, config: LayerConfig):
            self.config = config


    class MemoryAugmentedTransformer:
        def __init__(self, **kwargs):
            pass


    class MemoryInjector:
        def __init__(self, memory_cache, **kwargs):
            self.memory_cache = memory_cache

        def inject(self, model, **kwargs):
            pass

        def remove(self, model):
            pass
