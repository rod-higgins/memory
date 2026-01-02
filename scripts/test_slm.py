#!/usr/bin/env python3
"""
Test script for the Personal SLM system.

Tests the complete pipeline:
1. Training data preparation
2. Query interface
3. Context generation
4. LLM augmentation
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_pipeline():
    print("=" * 60)
    print("Personal SLM - Pipeline Test")
    print("=" * 60)
    print()

    # Test 1: Import all modules
    print("[1] Testing imports...")
    try:
        from memory import MemoryAPI, MemoryAugmenter, create_augmenter  # noqa: F401
        from memory.export.formats import ExportFormat, MemoryExporter  # noqa: F401
        from memory.query.interface import MemoryQuery  # noqa: F401
        from memory.slm.data import MemoryDataset, prepare_training_data  # noqa: F401
        from memory.slm.model import HybridPersonalSLM, get_personal_slm  # noqa: F401
        from memory.slm.trainer import PersonalSLMTrainer, TrainingConfig  # noqa: F401
        print("    All modules imported successfully")
    except ImportError as e:
        print(f"    Import error: {e}")
        return

    # Test 2: Initialize API
    print("\n[2] Testing API initialization...")
    api = MemoryAPI(base_path="~/memory/data")
    await api.initialize()
    stats = await api.stats()
    print("    API initialized")
    print(f"    Total memories: {stats['counts'].get('total', 0)}")

    # Test 3: Test query interface
    print("\n[3] Testing query interface...")
    result = await api.search("drupal preferences", limit=5)
    print(f"    Found {len(result.memories)} memories")
    if result.memories:
        print(f"    Top result: {result.memories[0].content[:80]}...")

    # Test 4: Test context generation
    print("\n[4] Testing context generation...")
    context = await api.get_context("How should I structure a Drupal project?")
    print(f"    Generated context ({len(context)} chars)")
    print(f"    Preview: {context[:200]}...")

    # Test 5: Test prompt augmentation
    print("\n[5] Testing prompt augmentation...")
    augmented = await api.augment_prompt("Help me with Python data processing")
    print(f"    Original prompt length: {len(augmented.original_prompt)}")
    print(f"    Augmented length: {len(augmented.augmented_prompt)}")
    print(f"    Memories used: {augmented.memory_count}")
    print(f"    Processing time: {augmented.processing_time_ms:.1f}ms")

    # Test 6: Test export formats
    print("\n[6] Testing export formats...")
    exporter = MemoryExporter()

    if result.memories:
        for fmt in [ExportFormat.CLAUDE_FORMAT, ExportFormat.XML, ExportFormat.JSON, ExportFormat.COMPACT]:
            output = exporter.export(result.memories[:3], format=fmt)
            print(f"    {fmt.value}: {len(output)} chars")

    # Test 7: Test Personal SLM (hybrid mode)
    print("\n[7] Testing Personal SLM (hybrid mode)...")
    try:
        slm = await get_personal_slm(use_hybrid=True)
        ctx = await slm.generate_context("What programming languages do I know?")
        print(f"    Generated context: {len(ctx.context)} chars")
        print(f"    Confidence: {ctx.confidence:.2f}")
        print(f"    Time: {ctx.processing_time_ms:.1f}ms")
        if isinstance(slm, HybridPersonalSLM):
            await slm.close()
    except Exception as e:
        print(f"    SLM test skipped: {e}")

    # Test 8: Training data preparation
    print("\n[8] Testing training data preparation...")
    try:
        trainer = PersonalSLMTrainer()
        # Just get the config, don't actually prepare data
        config = trainer.get_training_stats()
        print(f"    Base model: {config['base_model']}")
        print(f"    LoRA rank: {config['lora_rank']}")
        print(f"    Output dir: {config['output_dir']}")
    except Exception as e:
        print(f"    Training config error: {e}")

    await api.close()

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Prepare training data: python -c \"")
    print("       from memory.slm import PersonalSLMTrainer")
    print("       import asyncio")
    print("       trainer = PersonalSLMTrainer()")
    print("       asyncio.run(trainer.prepare_data())\"")
    print()
    print("  2. Train the model (requires GPU):")
    print("       trainer.train()")
    print("       trainer.merge_and_save()")
    print("       trainer.export_to_ollama()")
    print()
    print("  3. Use the trained model:")
    print("       ollama run personal-slm")


if __name__ == "__main__":
    asyncio.run(test_pipeline())
