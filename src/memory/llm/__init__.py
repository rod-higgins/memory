"""LLM providers, embeddings, and memory augmentation."""

from memory.llm.augmentation import (
    AugmentationConfig,
    AugmentedPrompt,
    MemoryAugmenter,
    MemoryMiddleware,
    create_augmenter,
)
from memory.llm.embeddings import EmbeddingProvider, SentenceTransformerProvider
from memory.llm.slm import OllamaSLM, SLMProvider

__all__ = [
    # Embeddings
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    # SLM
    "SLMProvider",
    "OllamaSLM",
    # Augmentation
    "MemoryAugmenter",
    "MemoryMiddleware",
    "AugmentationConfig",
    "AugmentedPrompt",
    "create_augmenter",
]
