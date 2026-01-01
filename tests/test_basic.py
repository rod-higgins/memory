"""Basic tests for PLM framework."""

import pytest


def test_import_memory():
    """Test that the memory package can be imported."""
    from memory import schema

    assert schema is not None


def test_import_storage():
    """Test that storage backends can be imported."""
    from memory.storage import DictStore, LanceDBStore, SQLiteStore

    assert DictStore is not None
    assert LanceDBStore is not None
    assert SQLiteStore is not None


def test_import_schema():
    """Test that schema models can be imported."""
    from memory.schema import MemoryEntry, MemoryTier, MemoryType, TruthCategory

    assert MemoryEntry is not None
    assert MemoryType is not None
    assert TruthCategory is not None
    assert MemoryTier is not None


def test_memory_entry_creation():
    """Test creating a basic memory entry."""
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory

    entry = MemoryEntry(
        content="Test memory content",
        memory_type=MemoryType.FACT,
        truth_category=TruthCategory.CONTEXTUAL,
        tier=MemoryTier.SHORT_TERM,
        confidence=ConfidenceScore(overall=0.8),
    )

    assert entry.content == "Test memory content"
    assert entry.memory_type == MemoryType.FACT
    assert entry.confidence.overall == 0.8
    assert entry.id is not None


@pytest.mark.asyncio
async def test_dict_store_basic():
    """Test basic DictStore operations."""
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory
    from memory.storage import DictStore

    store = DictStore()

    entry = MemoryEntry(
        content="Test memory",
        memory_type=MemoryType.FACT,
        truth_category=TruthCategory.CONTEXTUAL,
        tier=MemoryTier.SHORT_TERM,
        confidence=ConfidenceScore(overall=0.7),
    )

    # Store and retrieve (async methods)
    stored = await store.store(entry)
    assert stored.id == entry.id

    retrieved = await store.get(entry.id)
    assert retrieved is not None
    assert retrieved.content == "Test memory"


@pytest.mark.asyncio
async def test_storage_manager_initialization():
    """Test that StorageManager can be created."""
    from memory.storage import StorageManager

    manager = StorageManager()
    assert manager is not None
