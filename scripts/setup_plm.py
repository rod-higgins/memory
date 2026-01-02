#!/usr/bin/env python3
"""
PLM Setup Script

This script sets up and ingests digital life data sources from a config file.
Personal data is stored in data/persistent/data_sources.json (gitignored).

Supported data sources:
- GitHub repositories (via gh CLI)
- Local documents
- Gmail (requires Google Takeout export)
- Twitter/X (requires data export)
- Facebook (requires data export)
- LinkedIn (requires data export)
- AWS resources (via AWS CLI)

Usage:
    python scripts/setup_plm.py              # Full setup
    python scripts/setup_plm.py --ingest-exports  # Only ingest exports
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

# Data directories
HOME = Path.home()
PLM_DATA = HOME / "memory" / "data"
CONFIG_FILE = PLM_DATA / "persistent" / "data_sources.json"
EXPORTS_DIR = PLM_DATA / "exports"


def load_config() -> dict[str, Any]:
    """Load configuration from data_sources.json."""
    if not CONFIG_FILE.exists():
        print(f"⚠ Config file not found: {CONFIG_FILE}")
        print("  Please create data/persistent/data_sources.json with your data sources.")
        print("  See docs/SETUP.md for the config format.")
        return {}

    with open(CONFIG_FILE) as f:
        return json.load(f)


def setup_directories(config: dict[str, Any]):
    """Create all necessary directories."""
    print("\n📁 Setting up directories...")

    # Create main data directory
    PLM_DATA.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (PLM_DATA / "short_term").mkdir(exist_ok=True)
    (PLM_DATA / "long_term").mkdir(exist_ok=True)
    (PLM_DATA / "persistent").mkdir(exist_ok=True)

    # Create export directories from config
    for email_config in config.get("email", {}).get("gmail", []):
        export_path = Path(email_config.get("export_path", "")).expanduser()
        if export_path:
            export_path.mkdir(parents=True, exist_ok=True)

    for platform, social_config in config.get("social_media", {}).items():
        export_path = Path(social_config.get("export_path", "")).expanduser()
        if export_path:
            export_path.mkdir(parents=True, exist_ok=True)

    print("  ✓ All directories created")


def get_github_token():
    """Get GitHub token from gh CLI or config file."""
    # Try gh auth token first (newer gh versions)
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        pass

    # Fall back to reading config file
    try:
        import yaml
        gh_config = HOME / ".config" / "gh" / "hosts.yml"
        if gh_config.exists():
            with open(gh_config) as f:
                hosts = yaml.safe_load(f)
            return hosts.get("github.com", {}).get("oauth_token")
    except Exception:
        pass

    return None


async def ingest_github(config: dict[str, Any]):
    """Ingest GitHub repositories and activity."""
    print("\n🐙 Ingesting GitHub data...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from memory.ingestion.sources import GitHubSource
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory
    from memory.storage import StorageManager

    github_config = config.get("code", {}).get("github", {})
    accounts = [a.get("username") for a in github_config.get("accounts", []) if a.get("username")]

    if not accounts:
        print("  ⚠ No GitHub accounts configured.")
        return 0

    token = get_github_token()
    if not token:
        print("  ⚠ No GitHub token found. Run 'gh auth login' first.")
        return 0

    manager = StorageManager(base_path=str(PLM_DATA))
    await manager.initialize()

    total_count = 0

    for username in accounts:
        print(f"\n  Processing @{username}...")
        source = GitHubSource(
            token=token,
            username=username,
            include_repos=True,
            include_issues=True,
            include_prs=True,
            include_stars=True,
        )

        count = 0
        async for data_point in source.iterate():
            memory = MemoryEntry(tier=MemoryTier.PERSISTENT,
                content=data_point.content,
                memory_type=MemoryType.FACT,
                truth_category=TruthCategory.ABSOLUTE,
                confidence=ConfidenceScore(overall=0.9, source_reliability=0.95),
                tags=list(data_point.topics) + ["github", username],
                domains=["code", "software_development"],
                metadata={
                    "source": f"github_{username}",
                    "source_type": data_point.source_type,
                    **data_point.raw_data,
                },
            )
            await manager.store(memory)
            count += 1

            if count % 50 == 0:
                print(f"    Processed {count} items...")

        print(f"  ✓ @{username}: {count} items ingested")
        total_count += count

    await manager.close()
    return total_count


async def ingest_local_files(config: dict[str, Any]):
    """Ingest local documents."""
    print("\n📄 Ingesting local files...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from memory.ingestion.sources import LocalDocumentsSource
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory
    from memory.storage import StorageManager

    manager = StorageManager(base_path=str(PLM_DATA))
    await manager.initialize()

    total_count = 0
    local_docs = config.get("local_documents", {}).get("paths", [])

    for path_config in local_docs:
        name = path_config.get("name", "unknown")
        path = Path(path_config.get("path", "")).expanduser()
        extensions = path_config.get("extensions", [".txt", ".md", ".pdf", ".docx"])

        if not path.exists():
            print(f"  ⚠ {name}: {path} does not exist, skipping")
            continue

        print(f"\n  Processing {name} ({path})...")
        source = LocalDocumentsSource(
            path=str(path),
            extensions=extensions,
            recursive=True,
        )

        count = 0
        try:
            async for data_point in source.iterate():
                memory = MemoryEntry(tier=MemoryTier.PERSISTENT,
                    content=data_point.content[:50000],
                    memory_type=MemoryType.FACT,
                    truth_category=TruthCategory.CONTEXTUAL,
                    confidence=ConfidenceScore(overall=0.8, source_reliability=0.9),
                    tags=list(data_point.topics) + ["local_file", name],
                    domains=["personal", "documents"],
                    metadata={
                        "source": f"local_{name}",
                        "file_path": str(data_point.raw_data.get("file_path", "")),
                        **data_point.raw_data,
                    },
                )
                await manager.store(memory)
                count += 1

                if count % 100 == 0:
                    print(f"    Processed {count} files...")
        except Exception as e:
            print(f"    Error: {e}")

        print(f"  ✓ {name}: {count} files ingested")
        total_count += count

    await manager.close()
    return total_count


async def ingest_git_repos(config: dict[str, Any]):
    """Ingest local git repositories."""
    print("\n📂 Ingesting local git repositories...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from memory.ingestion.sources import LocalGitSource
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory
    from memory.storage import StorageManager

    code_path = config.get("code", {}).get("local_repos", {}).get("path", "")
    if not code_path:
        print("  ⚠ No local repos path configured.")
        return 0

    code_dir = Path(code_path).expanduser()
    if not code_dir.exists():
        print(f"  ⚠ {code_dir} does not exist, skipping")
        return 0

    manager = StorageManager(base_path=str(PLM_DATA))
    await manager.initialize()

    print(f"  Scanning {code_dir} for git repositories...")
    source = LocalGitSource(
        path=str(code_dir),
        include_commits=True,
        include_code=False,
        max_commits=100,
    )

    total_count = 0
    try:
        async for data_point in source.iterate():
            memory = MemoryEntry(tier=MemoryTier.PERSISTENT,
                content=data_point.content,
                memory_type=MemoryType.FACT,
                truth_category=TruthCategory.ABSOLUTE,
                confidence=ConfidenceScore(overall=0.95, source_reliability=1.0),
                tags=list(data_point.topics) + ["git", "commit"],
                domains=["code", "software_development"],
                metadata={
                    "source": "local_git",
                    **data_point.raw_data,
                },
            )
            await manager.store(memory)
            total_count += 1

            if total_count % 100 == 0:
                print(f"    Processed {total_count} commits...")
    except Exception as e:
        print(f"    Error: {e}")

    print(f"  ✓ {total_count} commits ingested")
    await manager.close()
    return total_count


async def ingest_aws(config: dict[str, Any]):
    """Ingest AWS resource metadata."""
    print("\n☁️  Ingesting AWS resources...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from memory.ingestion.sources import AWSSource
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory
    from memory.storage import StorageManager

    aws_config = config.get("cloud_services", {}).get("aws", {})
    if not aws_config.get("cli_accessible"):
        print("  ⚠ AWS CLI not configured in data_sources.json")
        return 0

    manager = StorageManager(base_path=str(PLM_DATA))
    await manager.initialize()

    profile = aws_config.get("profile", "default")
    services = aws_config.get("services_to_ingest", [])

    source = AWSSource(
        profile=profile,
        include_s3="s3" in services,
        include_lambda="lambda" in services,
        include_ec2="ec2" in services,
        include_dynamodb="dynamodb" in services,
        include_cloudwatch="cloudwatch" in services,
        include_iam="iam" in services,
    )

    if not await source.is_available():
        print("  ⚠ AWS CLI not configured. Run 'aws configure' first.")
        return 0

    total_count = 0
    try:
        async for data_point in source.iterate():
            memory = MemoryEntry(tier=MemoryTier.PERSISTENT,
                content=data_point.content,
                memory_type=MemoryType.FACT,
                truth_category=TruthCategory.ABSOLUTE,
                confidence=ConfidenceScore(overall=0.95, source_reliability=1.0),
                tags=list(data_point.topics) + ["aws", "cloud", "infrastructure"],
                domains=["cloud", "infrastructure", "devops"],
                metadata={
                    "source": "aws",
                    "source_type": data_point.source_type,
                    **data_point.raw_data,
                },
            )
            await manager.store(memory)
            total_count += 1

            if total_count % 20 == 0:
                print(f"    Processed {total_count} AWS resources...")
    except Exception as e:
        print(f"    Error: {e}")

    print(f"  ✓ {total_count} AWS resources ingested")
    await manager.close()
    return total_count


def create_identity_profile(config: dict[str, Any]):
    """Create identity profile from config."""
    print("\n👤 Creating identity profile...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from memory.schema import IdentityProfile

    identity = IdentityProfile.from_config_file(str(CONFIG_FILE))

    # Save identity profile
    identity_path = PLM_DATA / "persistent" / "identity.json"
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(identity.model_dump_json(indent=2))

    print(f"  ✓ Identity profile saved to {identity_path}")
    return identity


def print_export_instructions(config: dict[str, Any]):
    """Print instructions for manual data exports."""
    print("\n" + "=" * 70)
    print("📥 MANUAL DATA EXPORT INSTRUCTIONS")
    print("=" * 70)
    print("""
The following data sources require you to request a data export from each
platform. Once downloaded, extract to the directories specified in your
data_sources.json config file.

Supported platforms:
- Gmail: https://takeout.google.com/ (select Mail, .mbox format)
- Twitter/X: Settings > Your Account > Download an archive
- Facebook: Settings > Your Facebook Information > Download Profile
- LinkedIn: Settings > Data Privacy > Get a copy of your data

After downloading exports, run this script again to ingest them:
    python scripts/setup_plm.py --ingest-exports
""")


async def ingest_exports(config: dict[str, Any]):
    """Ingest any available data exports."""
    print("\n📦 Checking for data exports...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from memory.ingestion.sources import (
        FacebookExportSource,
        GmailExportSource,
        LinkedInExportSource,
        TwitterExportSource,
    )
    from memory.schema import ConfidenceScore, MemoryEntry, MemoryTier, MemoryType, TruthCategory
    from memory.storage import StorageManager

    manager = StorageManager(base_path=str(PLM_DATA))
    await manager.initialize()

    total_count = 0

    # Check Gmail exports
    for email_config in config.get("email", {}).get("gmail", []):
        address = email_config.get("address", "")
        export_path = Path(email_config.get("export_path", "")).expanduser()

        if not export_path.exists():
            continue

        mbox_files = list(export_path.glob("**/*.mbox"))
        if mbox_files:
            print(f"\n  Found Gmail export for {address}")
            for mbox_file in mbox_files:
                source = GmailExportSource(export_path=str(mbox_file))
                count = 0
                async for dp in source.iterate():
                    memory = MemoryEntry(tier=MemoryTier.PERSISTENT,
                        content=dp.content,
                        memory_type=MemoryType.FACT,
                        truth_category=TruthCategory.CONTEXTUAL,
                        confidence=ConfidenceScore(overall=0.85),
                        tags=list(dp.tags) + ["email"],
                        domains=["communication"],
                        metadata={"source": "gmail", **dp.metadata},
                    )
                    await manager.store(memory)
                    count += 1
                print(f"    ✓ {count} emails from {mbox_file.name}")
                total_count += count

    # Check social media exports
    social_config = config.get("social_media", {})

    for platform, platform_config in social_config.items():
        export_path = Path(platform_config.get("export_path", "")).expanduser()

        if not export_path.exists():
            continue

        source = None
        if platform == "twitter" and (export_path / "data").exists():
            source = TwitterExportSource(export_path=str(export_path))
        elif platform == "facebook" and ((export_path / "posts").exists() or (export_path / "your_posts").exists()):
            source = FacebookExportSource(export_path=str(export_path))
        elif platform == "linkedin" and any(export_path.glob("*.csv")):
            source = LinkedInExportSource(export_path=str(export_path))

        if source:
            print(f"\n  Found {platform.title()} export")
            count = 0
            async for dp in source.iterate():
                memory = MemoryEntry(tier=MemoryTier.PERSISTENT,
                    content=dp.content,
                    memory_type=MemoryType.FACT,
                    truth_category=TruthCategory.CONTEXTUAL,
                    confidence=ConfidenceScore(overall=0.7),
                    tags=list(dp.tags) + [platform, "social_media"],
                    domains=["social_media"],
                    metadata={"source": platform, **dp.metadata},
                )
                await manager.store(memory)
                count += 1
            print(f"    ✓ {count} {platform} items ingested")
            total_count += count

    await manager.close()
    return total_count


async def main():
    """Run the complete PLM setup."""
    import sys

    print("=" * 70)
    print("🧠 PERSONAL LANGUAGE MODEL SETUP")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config = load_config()
    if not config:
        return

    # Parse args
    ingest_exports_only = "--ingest-exports" in sys.argv

    if not ingest_exports_only:
        # Full setup
        setup_directories(config)
        create_identity_profile(config)

        # Ingest data sources
        github_count = await ingest_github(config)
        git_count = await ingest_git_repos(config)
        local_count = await ingest_local_files(config)
        aws_count = await ingest_aws(config)

        print("\n" + "=" * 70)
        print("📊 INITIAL INGESTION COMPLETE")
        print("=" * 70)
        print(f"  GitHub items:     {github_count:,}")
        print(f"  Git commits:      {git_count:,}")
        print(f"  Local files:      {local_count:,}")
        print(f"  AWS resources:    {aws_count:,}")
        print(f"  TOTAL:            {github_count + git_count + local_count + aws_count:,}")

        # Print export instructions
        print_export_instructions(config)

    # Check for and ingest exports
    export_count = await ingest_exports(config)
    if export_count > 0:
        print(f"\n  Export items:     {export_count:,}")

    print("\n" + "=" * 70)
    print("✅ PLM SETUP COMPLETE")
    print("=" * 70)
    print(f"\nYour memory data is stored in: {PLM_DATA}")


if __name__ == "__main__":
    asyncio.run(main())
