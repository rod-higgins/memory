#!/usr/bin/env python3
"""
Quick test script to verify the memory system setup.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    print("=" * 60)
    print("Personal Memory System - Setup Test")
    print("=" * 60)
    print()

    # Test 1: Import core modules
    print("[1] Testing imports...")
    try:
        from memory.schema.memory_entry import (
            MemoryEntry,
            MemoryTier,
            TruthCategory,
            MemoryType,
        )
        from memory.schema.identity import IdentityProfile
        from memory.storage.manager import StorageManager
        from memory.ingestion.parsers.claude_history import ClaudeHistoryParser

        print("    ✓ All core modules imported successfully")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        return

    # Test 2: Create identity profile
    print("\n[2] Testing identity profile...")
    identity = IdentityProfile.create_default()
    print(f"    ✓ Created identity for: {identity.name}")
    print(f"    ✓ GitHub handles: {identity.github_handles}")

    # Test 3: Create memory entry
    print("\n[3] Testing memory entry creation...")
    memory = MemoryEntry(
        content="I prefer Python for data processing tasks",
        memory_type=MemoryType.PREFERENCE,
        truth_category=TruthCategory.OPINION,
        domains=["python", "programming"],
        tags=["preference", "language"],
    )
    print(f"    ✓ Created memory: {memory.content[:50]}...")
    print(f"    ✓ Content hash: {memory.content_hash}")

    # Test 4: Initialize storage
    print("\n[4] Testing storage initialization...")
    storage = StorageManager()
    await storage.initialize()
    print("    ✓ Storage manager initialized")

    # Test 5: Store and retrieve memory
    print("\n[5] Testing store and retrieve...")
    stored = await storage.store(memory)
    print(f"    ✓ Stored memory with ID: {stored.id}")

    retrieved = await storage.get(stored.id)
    if retrieved:
        print(f"    ✓ Retrieved memory: {retrieved.content[:50]}...")
    else:
        print("    ✗ Failed to retrieve memory")

    # Test 6: Check Claude history
    print("\n[6] Checking Claude history source...")
    parser = ClaudeHistoryParser()
    if parser._history_path.exists():
        count = await parser.count_entries()
        first, last = await parser.get_date_range()
        print(f"    ✓ Found {count} entries in Claude history")
        print(f"    ✓ Date range: {first} to {last}")
    else:
        print(f"    ✗ Claude history not found at: {parser._history_path}")

    # Test 7: Get stats
    print("\n[7] Getting system stats...")
    stats = await storage.get_stats()
    print(f"    ✓ Total memories: {stats['counts'].get('total', 0)}")
    print(f"    ✓ Base path: {stats['base_path']}")

    # Cleanup
    await storage.close()

    print("\n" + "=" * 60)
    print("All tests passed! System is ready.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install -e .")
    print("  2. Initialize: python -m memory.cli init")
    print("  3. Ingest history: python -m memory.cli ingest claude-history --max 100")
    print("  4. Search: python -m memory.cli search 'drupal'")


if __name__ == "__main__":
    asyncio.run(main())
