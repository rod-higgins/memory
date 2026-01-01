"""
Embedding providers for the memory system.

All embedding operations are local and private - no external API calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Runs entirely on-device, no external API calls.
    Model is downloaded and cached on first use.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path | None = None,
        device: str | None = None,
    ):
        self._model_name = model_name
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._device = device
        self._model: Any = None
        self._dimensions: int = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        if not self._dimensions:
            self._load_model()
        return self._dimensions

    def _load_model(self) -> None:
        """Lazy load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        # Load model with optional cache directory
        kwargs: dict[str, Any] = {}
        if self._cache_dir:
            kwargs["cache_folder"] = str(self._cache_dir)
        if self._device:
            kwargs["device"] = self._device

        self._model = SentenceTransformer(self._model_name, **kwargs)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using Ollama's local models.

    Requires Ollama to be running locally.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self._model_name = model_name
        self._base_url = base_url
        self._dimensions = 768  # nomic-embed-text default

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model_name, "prompt": text},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't have native batch embedding, so we call one by one
        return [await self.embed(text) for text in texts]


# Model configurations for common embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "description": "Fast, lightweight model. Good for most use cases.",
        "size_mb": 80,
    },
    "all-mpnet-base-v2": {
        "dimensions": 768,
        "description": "Higher quality, slightly slower.",
        "size_mb": 420,
    },
    "BAAI/bge-small-en-v1.5": {
        "dimensions": 384,
        "description": "Excellent quality, competitive with larger models.",
        "size_mb": 130,
    },
    "BAAI/bge-base-en-v1.5": {
        "dimensions": 768,
        "description": "High quality, good balance of speed and accuracy.",
        "size_mb": 440,
    },
    "nomic-embed-text": {
        "dimensions": 768,
        "description": "Via Ollama. Good quality, requires Ollama.",
        "size_mb": 274,
    },
}


def get_embedding_provider(
    provider: str = "sentence_transformers",
    model: str = "all-MiniLM-L6-v2",
    **kwargs: Any,
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider.

    Args:
        provider: Provider type ("sentence_transformers" or "ollama")
        model: Model name
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedding provider

    Raises:
        ImportError: If required dependencies are not installed
    """
    if provider == "sentence_transformers":
        # Check if sentence_transformers is available before creating
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        return SentenceTransformerProvider(model_name=model, **kwargs)
    elif provider == "ollama":
        return OllamaEmbeddingProvider(model_name=model, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
