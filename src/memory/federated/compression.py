"""
Model Compression for PLM federated learning.

Implements efficient compression for model updates:
- Top-k sparsification
- Quantization
- Random sparsification
- Error feedback for compression
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompressionMethod(Enum):
    """Method for compressing model updates."""

    NONE = "none"
    TOP_K = "top_k"  # Keep only top-k largest values
    RANDOM_K = "random_k"  # Random sparsification
    THRESHOLD = "threshold"  # Keep values above threshold
    QUANTIZE = "quantize"  # Reduce precision
    COMBINED = "combined"  # Sparsify + quantize


@dataclass
class CompressedUpdate:
    """A compressed model update."""

    layer_name: str
    method: CompressionMethod

    # For sparse methods
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    original_length: int = 0

    # For quantization
    scale: float = 1.0
    zero_point: float = 0.0
    quantized_values: list[int] = field(default_factory=list)
    bits: int = 8

    # Compression stats
    compression_ratio: float = 1.0
    original_bytes: int = 0
    compressed_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for transmission."""
        return {
            "layer_name": self.layer_name,
            "method": self.method.value,
            "indices": self.indices,
            "values": self.values,
            "original_length": self.original_length,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "quantized_values": self.quantized_values,
            "bits": self.bits,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressedUpdate:
        """Deserialize from transmission."""
        return cls(
            layer_name=data.get("layer_name", ""),
            method=CompressionMethod(data.get("method", "none")),
            indices=data.get("indices", []),
            values=data.get("values", []),
            original_length=data.get("original_length", 0),
            scale=data.get("scale", 1.0),
            zero_point=data.get("zero_point", 0.0),
            quantized_values=data.get("quantized_values", []),
            bits=data.get("bits", 8),
            compression_ratio=data.get("compression_ratio", 1.0),
        )


class ModelCompressor:
    """
    Compresses model updates for efficient federated communication.

    Supports:
    - Sparsification (top-k, random-k, threshold)
    - Quantization (8-bit, 4-bit, 2-bit)
    - Error feedback to reduce compression error
    """

    def __init__(
        self,
        method: CompressionMethod = CompressionMethod.TOP_K,
        compression_ratio: float = 0.1,
        quantize_bits: int = 8,
        use_error_feedback: bool = True,
    ):
        """
        Initialize compressor.

        Args:
            method: Compression method to use
            compression_ratio: Fraction of values to keep (for sparse methods)
            quantize_bits: Bits for quantization (8, 4, or 2)
            use_error_feedback: Accumulate compression error for next round
        """
        self.method = method
        self.compression_ratio = compression_ratio
        self.quantize_bits = quantize_bits
        self.use_error_feedback = use_error_feedback

        # Error feedback buffer per layer
        self.error_buffer: dict[str, list[float]] = {}

    def compress(
        self,
        layer_name: str,
        values: list[float],
    ) -> CompressedUpdate:
        """Compress a layer's update values."""
        if self.method == CompressionMethod.NONE:
            return self._no_compression(layer_name, values)
        elif self.method == CompressionMethod.TOP_K:
            return self._top_k_compress(layer_name, values)
        elif self.method == CompressionMethod.RANDOM_K:
            return self._random_k_compress(layer_name, values)
        elif self.method == CompressionMethod.THRESHOLD:
            return self._threshold_compress(layer_name, values)
        elif self.method == CompressionMethod.QUANTIZE:
            return self._quantize_compress(layer_name, values)
        elif self.method == CompressionMethod.COMBINED:
            return self._combined_compress(layer_name, values)
        else:
            return self._no_compression(layer_name, values)

    def decompress(
        self,
        compressed: CompressedUpdate,
    ) -> list[float]:
        """Decompress an update back to full values."""
        if compressed.method == CompressionMethod.NONE:
            return compressed.values

        if compressed.method in [
            CompressionMethod.TOP_K,
            CompressionMethod.RANDOM_K,
            CompressionMethod.THRESHOLD,
        ]:
            # Sparse decompression
            result = [0.0] * compressed.original_length
            for idx, val in zip(compressed.indices, compressed.values):
                if idx < compressed.original_length:
                    result[idx] = val
            return result

        if compressed.method == CompressionMethod.QUANTIZE:
            # Dequantize
            return [
                (q - compressed.zero_point) * compressed.scale
                for q in compressed.quantized_values
            ]

        if compressed.method == CompressionMethod.COMBINED:
            # First dequantize
            dequantized = [
                (q - compressed.zero_point) * compressed.scale
                for q in compressed.quantized_values
            ]
            # Then expand sparse
            result = [0.0] * compressed.original_length
            for idx, val in zip(compressed.indices, dequantized):
                if idx < compressed.original_length:
                    result[idx] = val
            return result

        return compressed.values

    def _no_compression(
        self,
        layer_name: str,
        values: list[float],
    ) -> CompressedUpdate:
        """No compression - pass through."""
        return CompressedUpdate(
            layer_name=layer_name,
            method=CompressionMethod.NONE,
            values=values,
            original_length=len(values),
            compression_ratio=1.0,
        )

    def _top_k_compress(
        self,
        layer_name: str,
        values: list[float],
    ) -> CompressedUpdate:
        """Keep only top-k largest magnitude values."""
        # Apply error feedback
        if self.use_error_feedback and layer_name in self.error_buffer:
            error = self.error_buffer[layer_name]
            if len(error) == len(values):
                values = [v + e for v, e in zip(values, error)]

        k = max(1, int(len(values) * self.compression_ratio))

        # Find top-k indices
        indexed = [(i, abs(v), v) for i, v in enumerate(values)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_k = indexed[:k]

        indices = [item[0] for item in top_k]
        kept_values = [item[2] for item in top_k]

        # Compute and store error for feedback
        if self.use_error_feedback:
            compressed_full = [0.0] * len(values)
            for idx, val in zip(indices, kept_values):
                compressed_full[idx] = val
            self.error_buffer[layer_name] = [
                v - c for v, c in zip(values, compressed_full)
            ]

        return CompressedUpdate(
            layer_name=layer_name,
            method=CompressionMethod.TOP_K,
            indices=indices,
            values=kept_values,
            original_length=len(values),
            compression_ratio=k / len(values) if values else 1.0,
        )

    def _random_k_compress(
        self,
        layer_name: str,
        values: list[float],
    ) -> CompressedUpdate:
        """Randomly select k values (unbiased estimator)."""
        k = max(1, int(len(values) * self.compression_ratio))

        # Random selection
        indices = random.sample(range(len(values)), k)

        # Scale values to be unbiased
        scale = len(values) / k
        kept_values = [values[i] * scale for i in indices]

        return CompressedUpdate(
            layer_name=layer_name,
            method=CompressionMethod.RANDOM_K,
            indices=indices,
            values=kept_values,
            original_length=len(values),
            compression_ratio=k / len(values) if values else 1.0,
        )

    def _threshold_compress(
        self,
        layer_name: str,
        values: list[float],
    ) -> CompressedUpdate:
        """Keep values above a magnitude threshold."""
        if not values:
            return self._no_compression(layer_name, values)

        # Compute threshold to achieve target compression ratio
        magnitudes = sorted([abs(v) for v in values], reverse=True)
        k = max(1, int(len(values) * self.compression_ratio))
        threshold = magnitudes[min(k, len(magnitudes) - 1)]

        indices = []
        kept_values = []
        for i, v in enumerate(values):
            if abs(v) >= threshold:
                indices.append(i)
                kept_values.append(v)

        return CompressedUpdate(
            layer_name=layer_name,
            method=CompressionMethod.THRESHOLD,
            indices=indices,
            values=kept_values,
            original_length=len(values),
            compression_ratio=len(indices) / len(values) if values else 1.0,
        )

    def _quantize_compress(
        self,
        layer_name: str,
        values: list[float],
    ) -> CompressedUpdate:
        """Quantize values to reduced precision."""
        if not values:
            return self._no_compression(layer_name, values)

        min_val = min(values)
        max_val = max(values)

        # Compute scale and zero point
        num_levels = (1 << self.quantize_bits) - 1
        if max_val - min_val > 0:
            scale = (max_val - min_val) / num_levels
            zero_point = -min_val / scale
        else:
            scale = 1.0
            zero_point = 0.0

        # Quantize
        quantized = []
        for v in values:
            q = round((v - min_val) / scale) if scale > 0 else 0
            q = max(0, min(num_levels, q))
            quantized.append(int(q))

        # Compute compression ratio
        # Original: 32 bits per float, Quantized: self.quantize_bits per value
        ratio = self.quantize_bits / 32

        return CompressedUpdate(
            layer_name=layer_name,
            method=CompressionMethod.QUANTIZE,
            original_length=len(values),
            scale=scale,
            zero_point=zero_point,
            quantized_values=quantized,
            bits=self.quantize_bits,
            compression_ratio=ratio,
        )

    def _combined_compress(
        self,
        layer_name: str,
        values: list[float],
    ) -> CompressedUpdate:
        """Apply both sparsification and quantization."""
        # First sparsify
        sparse = self._top_k_compress(layer_name, values)

        # Then quantize the sparse values
        if not sparse.values:
            return sparse

        min_val = min(sparse.values)
        max_val = max(sparse.values)

        num_levels = (1 << self.quantize_bits) - 1
        if max_val - min_val > 0:
            scale = (max_val - min_val) / num_levels
            zero_point = -min_val / scale
        else:
            scale = 1.0
            zero_point = 0.0

        quantized = []
        for v in sparse.values:
            q = round((v - min_val) / scale) if scale > 0 else 0
            q = max(0, min(num_levels, q))
            quantized.append(int(q))

        # Combined compression ratio
        sparse_ratio = len(sparse.indices) / len(values) if values else 1.0
        quant_ratio = self.quantize_bits / 32
        combined_ratio = sparse_ratio * quant_ratio

        return CompressedUpdate(
            layer_name=layer_name,
            method=CompressionMethod.COMBINED,
            indices=sparse.indices,
            original_length=len(values),
            scale=scale,
            zero_point=zero_point,
            quantized_values=quantized,
            bits=self.quantize_bits,
            compression_ratio=combined_ratio,
        )

    def compress_model(
        self,
        model: dict[str, list[float]],
    ) -> dict[str, CompressedUpdate]:
        """Compress all layers of a model."""
        return {
            layer: self.compress(layer, values)
            for layer, values in model.items()
        }

    def decompress_model(
        self,
        compressed_model: dict[str, CompressedUpdate],
    ) -> dict[str, list[float]]:
        """Decompress all layers of a model."""
        return {
            layer: self.decompress(update)
            for layer, update in compressed_model.items()
        }

    def get_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        return {
            "method": self.method.value,
            "compression_ratio": self.compression_ratio,
            "quantize_bits": self.quantize_bits,
            "use_error_feedback": self.use_error_feedback,
            "error_buffer_layers": list(self.error_buffer.keys()),
        }

    def reset_error_buffer(self) -> None:
        """Reset error feedback buffer."""
        self.error_buffer = {}


class AdaptiveCompressor(ModelCompressor):
    """
    Adaptive compression that adjusts based on network conditions.

    Monitors bandwidth and latency to choose optimal compression.
    """

    def __init__(
        self,
        target_bytes: int = 1_000_000,  # 1MB target per update
        min_ratio: float = 0.01,
        max_ratio: float = 0.5,
    ):
        super().__init__()
        self.target_bytes = target_bytes
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        # Bandwidth estimation
        self.bandwidth_history: list[float] = []
        self.latency_history: list[float] = []

    def adapt_compression(
        self,
        model_size_bytes: int,
        available_bandwidth: float | None = None,
    ) -> None:
        """Adapt compression ratio based on conditions."""
        if available_bandwidth:
            self.bandwidth_history.append(available_bandwidth)

        # Estimate required compression
        if model_size_bytes > 0:
            target_ratio = self.target_bytes / model_size_bytes
            target_ratio = max(self.min_ratio, min(self.max_ratio, target_ratio))
            self.compression_ratio = target_ratio

        # Choose method based on ratio
        if self.compression_ratio < 0.05:
            self.method = CompressionMethod.COMBINED
            self.quantize_bits = 4
        elif self.compression_ratio < 0.2:
            self.method = CompressionMethod.TOP_K
        else:
            self.method = CompressionMethod.QUANTIZE
            self.quantize_bits = 8
