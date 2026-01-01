#!/usr/bin/env python3
"""
Run the PMS sync scheduler.

Usage:
    python scripts/run_sync.py              # Run once
    python scripts/run_sync.py --daemon     # Run as daemon
    python scripts/run_sync.py --daemon 30  # Run daemon with 30 min interval
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory.scheduler import run_sync_daemon, run_sync_once


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        print(f"Starting sync daemon (interval: {interval} minutes)")
        asyncio.run(run_sync_daemon(interval))
    else:
        print("Running single sync cycle...")
        result = asyncio.run(run_sync_once())
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
