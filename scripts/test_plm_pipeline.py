#!/usr/bin/env python3
"""
Test the PLM ingestion and training data preparation pipeline.

This script:
1. Tests data ingestion from configured sources
2. Prepares training data from memories
3. Shows what would be needed for full PLM training

Usage:
    python scripts/test_plm_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_ingestion():
    """Test data ingestion from configured sources."""
    from memory.storage import StorageManager
    from memory.schema import MemoryEntry, MemoryType, MemoryTier, TruthCategory, ConfidenceScore

    print("\n" + "=" * 70)
    print("PHASE 1: DATA INGESTION TEST")
    print("=" * 70)

    data_path = Path.home() / "memory" / "data"
    config_file = data_path / "persistent" / "data_sources.json"

    if not config_file.exists():
        print(f"  Config file not found: {config_file}")
        return 0

    # Load config
    with open(config_file) as f:
        config = json.load(f)

    print(f"\n  Config loaded: {config.get('owner', 'Unknown')}")

    # Initialize storage
    manager = StorageManager(base_path=str(data_path))
    await manager.initialize()

    ingested_count = 0

    # Test GitHub ingestion
    print("\n  Testing GitHub ingestion...")
    github_accounts = [a.get("username") for a in config.get("code", {}).get("github", {}).get("accounts", [])]

    if github_accounts:
        from memory.ingestion.sources import GitHubSource
        import subprocess

        # Get token from gh CLI config
        import yaml
        try:
            gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
            if gh_config.exists():
                with open(gh_config) as f:
                    hosts = yaml.safe_load(f)
                token = hosts.get("github.com", {}).get("oauth_token")
            else:
                token = None

            for username in github_accounts[:1]:  # Test with first account only
                print(f"    Ingesting from @{username}...")
                source = GitHubSource(
                    token=token,
                    username=username,
                    include_repos=True,
                    include_issues=False,
                    include_prs=False,
                    include_stars=False,
                )

                count = 0
                async for data_point in source.iterate():
                    if count >= 10:  # Limit for testing
                        break

                    memory = MemoryEntry(
                        content=data_point.content,
                        memory_type=MemoryType.FACT,
                        truth_category=TruthCategory.ABSOLUTE,
                        confidence=ConfidenceScore(overall=0.9),
                        tags=list(data_point.topics) + ["github", username],
                        domains=["code"],
                        tier=MemoryTier.PERSISTENT,  # Store to persistent for durability
                    )
                    await manager.store(memory)
                    count += 1

                print(f"      Ingested {count} items from @{username}")
                ingested_count += count

        except Exception as e:
            print(f"    GitHub error: {e}")

    # Test Chrome History ingestion
    print("\n  Testing Chrome History ingestion...")
    from memory.ingestion.sources import ChromeHistorySource

    chrome_source = ChromeHistorySource(max_entries=100)  # Limit for testing

    if await chrome_source.is_available():
        count = 0
        try:
            async for data_point in chrome_source.iterate():
                if count >= 50:  # Limit for testing
                    break

                memory = MemoryEntry(
                    content=data_point.content,
                    memory_type=MemoryType.FACT,
                    truth_category=TruthCategory.CONTEXTUAL,
                    confidence=ConfidenceScore(overall=0.7),
                    tags=list(data_point.topics) + ["browser"],
                    domains=["browsing", "research"],
                    tier=MemoryTier.PERSISTENT,
                )
                await manager.store(memory)
                count += 1

            print(f"      Ingested {count} browser history items")
            ingested_count += count
        except Exception as e:
            print(f"      Chrome history error: {e}")
    else:
        print("      Chrome history not available")

    # Test AWS ingestion
    print("\n  Testing AWS ingestion...")
    aws_config = config.get("cloud_services", {}).get("aws", {})

    if aws_config.get("cli_accessible"):
        from memory.ingestion.sources import AWSSource

        source = AWSSource(
            profile=aws_config.get("profile", "default"),
            include_s3=True,
            include_lambda=True,
            include_ec2=False,
            include_dynamodb=False,
            include_cloudwatch=False,
            include_iam=False,
        )

        if await source.is_available():
            count = 0
            async for data_point in source.iterate():
                if count >= 10:  # Limit for testing
                    break

                memory = MemoryEntry(
                    content=data_point.content,
                    memory_type=MemoryType.FACT,
                    truth_category=TruthCategory.ABSOLUTE,
                    confidence=ConfidenceScore(overall=0.95),
                    tags=list(data_point.topics) + ["aws"],
                    domains=["cloud", "infrastructure"],
                    tier=MemoryTier.PERSISTENT,  # Store to persistent for durability
                )
                await manager.store(memory)
                count += 1

            print(f"      Ingested {count} AWS resources")
            ingested_count += count
        else:
            print("      AWS CLI not configured")

    # Get stats
    stats = await manager.get_stats()
    counts = stats.get("counts", {})
    print(f"\n  Storage stats:")
    print(f"    Short-term: {counts.get('short_term', 0)}")
    print(f"    Long-term:  {counts.get('long_term', 0)}")
    print(f"    Persistent: {counts.get('persistent', 0)}")
    print(f"    Total:      {counts.get('total', 0)}")

    await manager.close()
    return ingested_count


async def test_training_data():
    """Test training data preparation."""
    from memory.slm.data import prepare_training_data

    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING DATA PREPARATION")
    print("=" * 70)

    data_path = Path.home() / "memory" / "data"
    output_path = data_path / "training"

    print(f"\n  Preparing training data from: {data_path}")
    print(f"  Output path: {output_path}")

    try:
        stats = await prepare_training_data(
            storage_path=str(data_path),
            output_path=str(output_path),
            min_confidence=0.3,  # Lower for testing
        )

        print(f"\n  Results:")
        print(f"    Total memories:     {stats.get('total_memories', 0)}")
        print(f"    Filtered memories:  {stats.get('filtered_memories', 0)}")
        print(f"    Training examples:  {stats.get('training_examples', 0)}")
        print(f"    JSONL path:         {stats.get('jsonl_path', 'N/A')}")
        print(f"    Chat format path:   {stats.get('chat_path', 'N/A')}")

        # Show sample training data
        chat_path = Path(stats.get("chat_path", ""))
        if chat_path.exists():
            print(f"\n  Sample training example:")
            with open(chat_path) as f:
                first_line = f.readline()
                if first_line:
                    sample = json.loads(first_line)
                    print(f"    System: {sample['messages'][0]['content'][:60]}...")
                    print(f"    User:   {sample['messages'][1]['content'][:60]}...")
                    print(f"    Assist: {sample['messages'][2]['content'][:60]}...")

        return stats

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def check_training_requirements():
    """Check if training requirements are met."""
    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING REQUIREMENTS CHECK")
    print("=" * 70)

    requirements = {
        "torch": False,
        "transformers": False,
        "peft": False,
        "bitsandbytes": False,
        "accelerate": False,
        "datasets": False,
        "ollama": False,
        "gpu": False,
    }

    # Check Python packages
    try:
        import torch
        requirements["torch"] = True
        requirements["gpu"] = torch.cuda.is_available() or torch.backends.mps.is_available()
    except ImportError:
        pass

    try:
        import transformers
        requirements["transformers"] = True
    except ImportError:
        pass

    try:
        import peft
        requirements["peft"] = True
    except ImportError:
        pass

    try:
        import bitsandbytes
        requirements["bitsandbytes"] = True
    except ImportError:
        pass

    try:
        import accelerate
        requirements["accelerate"] = True
    except ImportError:
        pass

    try:
        import datasets
        requirements["datasets"] = True
    except ImportError:
        pass

    # Check Ollama
    import subprocess
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        requirements["ollama"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print("\n  Dependency status:")
    for dep, available in requirements.items():
        status = "OK" if available else "MISSING"
        print(f"    {dep:15} [{status}]")

    all_ready = all(requirements.values())

    if all_ready:
        print("\n  All dependencies satisfied. Ready for PLM training!")
    else:
        missing = [k for k, v in requirements.items() if not v]
        print(f"\n  Missing: {', '.join(missing)}")
        print("\n  To install training dependencies:")
        print("    pip install -e '.[tuning]'")
        print("\n  To install Ollama:")
        print("    brew install ollama  # macOS")
        print("    # or visit https://ollama.ai")

    return requirements


async def main():
    """Run the PLM pipeline test."""
    print("=" * 70)
    print("PLM PIPELINE TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: Ingestion
    ingested = await test_ingestion()

    # Phase 2: Training data preparation
    training_stats = await test_training_data()

    # Phase 3: Check requirements
    requirements = check_training_requirements()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Ingested items:     {ingested}")
    print(f"  Training examples:  {training_stats.get('training_examples', 0)}")

    if all(requirements.values()):
        print("\n  Next step: Run PLM training")
        print("    python -c 'from memory.slm import PersonalSLMTrainer; ...'")
    else:
        print("\n  Install missing dependencies before training")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
