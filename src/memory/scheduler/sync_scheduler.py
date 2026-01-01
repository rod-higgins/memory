"""
Sync Scheduler for Personal Memory System.

Runs periodic ingestion from configured data sources.
Can run as a background daemon or via cron.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from memory.api.memory_api import MemoryAPI


class SyncScheduler:
    """
    Scheduler for periodic data source synchronization.

    Reads connection configurations and syncs based on their intervals.
    """

    def __init__(
        self,
        connections_db: str = "~/memory/data/connections.db",
        log_path: str = "~/memory/data/sync.log",
    ):
        self.connections_db = Path(connections_db).expanduser()
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._api: MemoryAPI | None = None

    def _log(self, message: str) -> None:
        """Log a message."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] {message}\n"
        print(log_line.strip())
        with open(self.log_path, "a") as f:
            f.write(log_line)

    def get_connections_due(self) -> list[dict[str, Any]]:
        """Get connections that are due for sync."""
        if not self.connections_db.exists():
            return []

        conn = sqlite3.connect(str(self.connections_db))
        conn.row_factory = sqlite3.Row

        now = datetime.now()

        cursor = conn.execute("""
            SELECT * FROM connections
            WHERE sync_enabled = 1
            AND status != 'error'
        """)

        due_connections = []
        for row in cursor.fetchall():
            last_sync = row["last_sync"]
            interval_hours = row["sync_interval_hours"] or 24

            if last_sync:
                last_sync_dt = datetime.fromisoformat(last_sync)
                next_sync = last_sync_dt + timedelta(hours=interval_hours)
                if now >= next_sync:
                    due_connections.append(dict(row))
            else:
                # Never synced - due immediately
                due_connections.append(dict(row))

        conn.close()
        return due_connections

    async def sync_connection(self, connection: dict[str, Any]) -> dict[str, Any]:
        """Sync a single connection."""
        source_id = connection["source_id"]
        conn_id = connection["id"]
        name = connection["name"]

        self._log(f"Starting sync for {name} ({source_id})")

        try:
            # Get credentials
            credentials = json.loads(connection.get("credentials_json", "{}"))
            settings = json.loads(connection.get("settings_json", "{}"))

            # Run ingestion
            result = await self._api.ingest(
                source=source_id,
                **credentials,
                **settings,
            )

            # Update last sync time
            self._update_last_sync(conn_id)

            count = result.get("count", 0)
            self._log(f"Completed sync for {name}: {count} items ingested")

            return {"connection_id": conn_id, "success": True, "count": count}

        except Exception as e:
            error_msg = str(e)
            self._log(f"Error syncing {name}: {error_msg}")
            self._update_status(conn_id, "error", error_msg)
            return {"connection_id": conn_id, "success": False, "error": error_msg}

    def _update_last_sync(self, conn_id: str) -> None:
        """Update last sync timestamp."""
        conn = sqlite3.connect(str(self.connections_db))
        conn.execute(
            "UPDATE connections SET last_sync = ?, status = 'connected' WHERE id = ?",
            (datetime.now().isoformat(), conn_id),
        )
        conn.commit()
        conn.close()

    def _update_status(self, conn_id: str, status: str, error: str | None = None) -> None:
        """Update connection status."""
        conn = sqlite3.connect(str(self.connections_db))
        conn.execute(
            "UPDATE connections SET status = ?, last_error = ? WHERE id = ?",
            (status, error, conn_id),
        )
        conn.commit()
        conn.close()

    async def run_sync_cycle(self) -> dict[str, Any]:
        """Run a single sync cycle for all due connections."""
        if not self._api:
            self._api = MemoryAPI()
            await self._api.initialize()

        due_connections = self.get_connections_due()

        if not due_connections:
            self._log("No connections due for sync")
            return {"synced": 0, "results": []}

        self._log(f"Found {len(due_connections)} connections due for sync")

        results = []
        for connection in due_connections:
            result = await self.sync_connection(connection)
            results.append(result)

        successful = sum(1 for r in results if r.get("success"))
        total_items = sum(r.get("count", 0) for r in results if r.get("success"))

        self._log(f"Sync cycle complete: {successful}/{len(results)} successful, {total_items} items ingested")

        return {
            "synced": successful,
            "total_items": total_items,
            "results": results,
        }

    async def run_daemon(self, check_interval_minutes: int = 15) -> None:
        """Run as a background daemon."""
        self._running = True
        self._log(f"Starting sync daemon (check every {check_interval_minutes} minutes)")

        # Handle shutdown signals
        def shutdown_handler(signum, frame):
            self._log("Received shutdown signal")
            self._running = False

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler)

        while self._running:
            try:
                await self.run_sync_cycle()
            except Exception as e:
                self._log(f"Error in sync cycle: {e}")

            # Wait for next check
            await asyncio.sleep(check_interval_minutes * 60)

        self._log("Sync daemon stopped")

    async def close(self) -> None:
        """Clean up resources."""
        if self._api:
            await self._api.close()


async def run_sync_once() -> dict[str, Any]:
    """Run a single sync cycle."""
    scheduler = SyncScheduler()
    try:
        result = await scheduler.run_sync_cycle()
        return result
    finally:
        await scheduler.close()


async def run_sync_daemon(check_interval: int = 15) -> None:
    """Run the sync daemon."""
    scheduler = SyncScheduler()
    try:
        await scheduler.run_daemon(check_interval_minutes=check_interval)
    finally:
        await scheduler.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--daemon":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        asyncio.run(run_sync_daemon(interval))
    else:
        result = asyncio.run(run_sync_once())
        print(f"Sync result: {result}")
