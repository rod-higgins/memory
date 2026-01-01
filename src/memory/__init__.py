"""
Personal Memory System (PMS)

A three-tier personal memory model for AI interactions.

The system provides:
- Three-tier storage (short-term, long-term, persistent)
- Semantic search across all memories
- LLM context injection for personalized AI interactions
- Learning from interactions to build memory over time

Quick Start:
    from memory import MemoryAPI

    api = MemoryAPI()
    await api.initialize()

    # Store a memory
    await api.remember("I prefer Python for data processing")

    # Get context for an LLM
    context = await api.get_context("Help me with a Python project")

    # Augment a prompt
    augmented = await api.augment_prompt("How should I structure this?")
"""

__version__ = "0.1.0"

# Schema
# High-level API
from memory.api.memory_api import MemoryAPI, get_memory_context, remember

# Export
from memory.export.formats import ExportFormat, MemoryExporter

# LLM Integration
from memory.llm.augmentation import (
    AugmentedPrompt,
    MemoryAugmenter,
    create_augmenter,
)

# Query
from memory.query.interface import MemoryQuery, QueryContext, QueryResult
from memory.schema.domain import DomainContext
from memory.schema.identity import IdentityProfile
from memory.schema.memory_entry import (
    ConfidenceScore,
    MemoryEntry,
    MemorySource,
    MemoryTier,
    MemoryType,
    SourceType,
    TruthCategory,
)

# Storage
from memory.storage.manager import StorageManager

__all__ = [
    # Version
    "__version__",
    # Schema
    "MemoryEntry",
    "MemoryTier",
    "TruthCategory",
    "MemoryType",
    "SourceType",
    "MemorySource",
    "ConfidenceScore",
    "IdentityProfile",
    "DomainContext",
    # API
    "MemoryAPI",
    "get_memory_context",
    "remember",
    # Storage
    "StorageManager",
    # Query
    "MemoryQuery",
    "QueryResult",
    "QueryContext",
    # Export
    "MemoryExporter",
    "ExportFormat",
    # Augmentation
    "MemoryAugmenter",
    "AugmentedPrompt",
    "create_augmenter",
]
