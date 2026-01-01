"""
Pytest configuration and shared fixtures for PLM tests.
"""

import asyncio
import importlib.util
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest


# Configure asyncio for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_memory_entry():
    """Create a sample memory entry for testing."""
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory

    return MemoryEntry(
        content="I prefer Python for data processing tasks",
        memory_type=MemoryType.PREFERENCE,
        truth_category=TruthCategory.CONTEXTUAL,
        tier=MemoryTier.SHORT_TERM,
        confidence=ConfidenceScore(overall=0.85),
        domains=["programming", "python"],
        tags=["preference", "language"],
    )


@pytest.fixture
def sample_memories():
    """Create a list of sample memories for testing."""
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory

    return [
        MemoryEntry(
            content="Python is my preferred language for data science",
            memory_type=MemoryType.PREFERENCE,
            truth_category=TruthCategory.CONTEXTUAL,
            tier=MemoryTier.SHORT_TERM,
            confidence=ConfidenceScore(overall=0.9),
            domains=["programming"],
        ),
        MemoryEntry(
            content="I work at a technology company",
            memory_type=MemoryType.FACT,
            truth_category=TruthCategory.ABSOLUTE,
            tier=MemoryTier.PERSISTENT,
            confidence=ConfidenceScore(overall=1.0),
            domains=["work"],
        ),
        MemoryEntry(
            content="React is better than Angular for my projects",
            memory_type=MemoryType.BELIEF,
            truth_category=TruthCategory.OPINION,
            tier=MemoryTier.LONG_TERM,
            confidence=ConfidenceScore(overall=0.7),
            domains=["programming", "frontend"],
        ),
        MemoryEntry(
            content="Meeting with team every Monday at 10am",
            memory_type=MemoryType.EVENT,
            truth_category=TruthCategory.CONTEXTUAL,
            tier=MemoryTier.LONG_TERM,
            confidence=ConfidenceScore(overall=0.95),
            domains=["work", "schedule"],
        ),
    ]


@pytest.fixture
def dict_store():
    """Create a DictStore instance for testing."""
    from memory.storage import DictStore

    return DictStore()


@pytest.fixture
async def sqlite_store(temp_dir: Path) -> AsyncGenerator:
    """Create a SQLiteStore instance for testing."""
    from memory.storage import SQLiteStore

    db_path = temp_dir / "test.db"
    store = SQLiteStore(str(db_path))
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def lancedb_store(temp_dir: Path) -> AsyncGenerator:
    """Create a LanceDBStore instance for testing."""
    from memory.storage import LanceDBStore

    store = LanceDBStore(str(temp_dir / "lancedb"))
    await store.initialize()
    yield store


@pytest.fixture
async def storage_manager(temp_dir: Path) -> AsyncGenerator:
    """Create a StorageManager instance for testing."""
    from memory.storage import StorageManager

    manager = StorageManager(
        base_path=str(temp_dir),
    )
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    import random

    random.seed(42)
    return [random.random() for _ in range(384)]


@pytest.fixture
def sample_embeddings():
    """Create multiple sample embeddings."""
    import random

    random.seed(42)
    return [[random.random() for _ in range(384)] for _ in range(10)]


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_torch: marks tests that require PyTorch")
    config.addinivalue_line("markers", "requires_embeddings: marks tests that require sentence-transformers")


# Skip conditions
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_EMBEDDINGS = importlib.util.find_spec("sentence_transformers") is not None


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    skip_torch = pytest.mark.skip(reason="PyTorch not installed")
    skip_embeddings = pytest.mark.skip(reason="sentence-transformers not installed")

    for item in items:
        if "requires_torch" in item.keywords and not HAS_TORCH:
            item.add_marker(skip_torch)
        if "requires_embeddings" in item.keywords and not HAS_EMBEDDINGS:
            item.add_marker(skip_embeddings)
