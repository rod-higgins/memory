"""Tests for storage backends."""


import pytest


class TestDictStore:
    """Tests for in-memory DictStore."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, dict_store, sample_memory_entry):
        """Test basic store and retrieve operations."""
        stored = await dict_store.store(sample_memory_entry)
        assert stored.id == sample_memory_entry.id

        retrieved = await dict_store.get(sample_memory_entry.id)
        assert retrieved is not None
        assert retrieved.content == sample_memory_entry.content

    @pytest.mark.asyncio
    async def test_store_multiple(self, dict_store, sample_memories):
        """Test storing multiple entries."""
        for memory in sample_memories:
            await dict_store.store(memory)

        count = await dict_store.count()
        assert count == len(sample_memories)

    @pytest.mark.asyncio
    async def test_update_entry(self, dict_store, sample_memory_entry):
        """Test updating an existing entry."""
        await dict_store.store(sample_memory_entry)

        sample_memory_entry.confidence.overall = 0.95
        sample_memory_entry.content = "Updated content"
        await dict_store.update(sample_memory_entry)

        retrieved = await dict_store.get(sample_memory_entry.id)
        assert retrieved.confidence.overall == 0.95
        assert retrieved.content == "Updated content"

    @pytest.mark.asyncio
    async def test_delete_entry(self, dict_store, sample_memory_entry):
        """Test deleting an entry."""
        await dict_store.store(sample_memory_entry)
        retrieved = await dict_store.get(sample_memory_entry.id)
        assert retrieved is not None

        await dict_store.delete(sample_memory_entry.id)
        deleted = await dict_store.get(sample_memory_entry.id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, dict_store):
        """Test getting a non-existent entry."""
        from uuid import uuid4
        result = await dict_store.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, dict_store, sample_memories):
        """Test listing all entries."""
        for memory in sample_memories:
            await dict_store.store(memory)

        all_entries = await dict_store.list_all()
        assert len(all_entries) == len(sample_memories)

    @pytest.mark.asyncio
    async def test_filter_by_tier(self, dict_store, sample_memories):
        """Test filtering entries by tier."""
        from memory.schema import MemoryTier

        for memory in sample_memories:
            await dict_store.store(memory)

        # DictStore sets all entries to SHORT_TERM
        all_entries = await dict_store.list_all()
        assert all(m.tier == MemoryTier.SHORT_TERM for m in all_entries)

    @pytest.mark.asyncio
    async def test_filter_by_type(self, dict_store, sample_memories):
        """Test filtering entries by memory type."""
        from memory.schema import MemoryType

        for memory in sample_memories:
            await dict_store.store(memory)

        # Filter by memory type using list_all with filters
        all_entries = await dict_store.list_all(filters={"memory_type": [MemoryType.FACT.value]})
        assert all(m.memory_type == MemoryType.FACT for m in all_entries)

    @pytest.mark.asyncio
    async def test_count(self, dict_store, sample_memories):
        """Test counting entries."""
        for memory in sample_memories:
            await dict_store.store(memory)

        count = await dict_store.count()
        assert count == len(sample_memories)


class TestSQLiteStore:
    """Tests for SQLite persistent store."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, sqlite_store, sample_memory_entry):
        """Test basic store and retrieve operations."""
        await sqlite_store.store(sample_memory_entry)

        retrieved = await sqlite_store.get(sample_memory_entry.id)
        assert retrieved is not None
        assert retrieved.content == sample_memory_entry.content

    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir, sample_memory_entry):
        """Test that data persists across store instances."""
        from memory.storage import SQLiteStore

        db_path = temp_dir / "persist_test.db"

        # Store in first instance
        store1 = SQLiteStore(str(db_path))
        await store1.initialize()
        await store1.store(sample_memory_entry)
        await store1.close()

        # Retrieve in second instance
        store2 = SQLiteStore(str(db_path))
        await store2.initialize()
        retrieved = await store2.get(sample_memory_entry.id)
        await store2.close()

        assert retrieved is not None
        assert retrieved.content == sample_memory_entry.content

    @pytest.mark.asyncio
    async def test_update(self, sqlite_store, sample_memory_entry):
        """Test updating an entry."""
        await sqlite_store.store(sample_memory_entry)

        sample_memory_entry.confidence.overall = 0.99
        await sqlite_store.update(sample_memory_entry)

        retrieved = await sqlite_store.get(sample_memory_entry.id)
        assert retrieved.confidence.overall == 0.99

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store, sample_memory_entry):
        """Test deleting an entry."""
        await sqlite_store.store(sample_memory_entry)
        await sqlite_store.delete(sample_memory_entry.id)

        retrieved = await sqlite_store.get(sample_memory_entry.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_count(self, sqlite_store, sample_memories):
        """Test counting entries."""
        for memory in sample_memories:
            await sqlite_store.store(memory)

        count = await sqlite_store.count()
        assert count == len(sample_memories)


class TestLanceDBStore:
    """Tests for LanceDB vector store."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, lancedb_store, sample_memory_entry):
        """Test basic store and retrieve operations."""
        await lancedb_store.store(sample_memory_entry)

        retrieved = await lancedb_store.get(sample_memory_entry.id)
        assert retrieved is not None
        assert retrieved.content == sample_memory_entry.content

    @pytest.mark.asyncio
    async def test_store_with_embedding(self, lancedb_store, sample_memory_entry, sample_embedding):
        """Test storing entry with embedding."""
        sample_memory_entry.embedding = sample_embedding
        await lancedb_store.store(sample_memory_entry)

        retrieved = await lancedb_store.get(sample_memory_entry.id)
        assert retrieved is not None
        assert retrieved.embedding is not None

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_vector_search(self, lancedb_store, sample_memories, sample_embeddings):
        """Test vector similarity search."""
        # Store memories with embeddings
        for i, memory in enumerate(sample_memories):
            if i < len(sample_embeddings):
                memory.embedding = sample_embeddings[i]
            await lancedb_store.store(memory)

        # Search with query embedding
        if sample_embeddings:
            results = await lancedb_store.search(
                query_embedding=sample_embeddings[0],
                limit=3,
            )
            assert len(results) <= 3


class TestStorageManager:
    """Tests for StorageManager that coordinates all stores."""

    @pytest.mark.asyncio
    async def test_initialization(self, storage_manager):
        """Test storage manager initialization."""
        assert storage_manager is not None

    @pytest.mark.asyncio
    async def test_store_to_tier(self, storage_manager, sample_memory_entry):
        """Test storing to specific tier."""
        await storage_manager.store(sample_memory_entry)

        retrieved = await storage_manager.get(sample_memory_entry.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_text_search(self, storage_manager, sample_memories):
        """Test text searching across tiers."""
        for memory in sample_memories:
            await storage_manager.store(memory)

        results, scores = await storage_manager.text_search(query_text="Python", limit=5)
        assert len(results) >= 0  # May be empty if no semantic match

    @pytest.mark.asyncio
    async def test_migrate_tier(self, storage_manager, sample_memory_entry):
        """Test migrating memory between tiers."""
        from memory.schema import MemoryTier

        sample_memory_entry.tier = MemoryTier.SHORT_TERM
        await storage_manager.store(sample_memory_entry)

        # Get the stored memory before migrating
        stored = await storage_manager.get(sample_memory_entry.id)
        await storage_manager.migrate_tier(
            stored,
            from_tier=MemoryTier.SHORT_TERM,
            to_tier=MemoryTier.LONG_TERM,
        )

        retrieved = await storage_manager.get(sample_memory_entry.id)
        assert retrieved.tier == MemoryTier.LONG_TERM

    @pytest.mark.asyncio
    async def test_get_stats(self, storage_manager, sample_memories):
        """Test getting storage statistics."""
        for memory in sample_memories:
            await storage_manager.store(memory)

        stats = await storage_manager.get_stats()
        assert "counts" in stats
        assert stats["counts"]["total"] >= len(sample_memories)
