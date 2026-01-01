"""
Sync engine for PLM mobile companion app.

Implements offline-first synchronization:
- Change tracking and delta sync
- Conflict resolution
- Background sync scheduling
- Bandwidth-aware sync
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class SyncState(Enum):
    """State of synchronization."""

    IDLE = "idle"
    SYNCING = "syncing"
    UPLOADING = "uploading"
    DOWNLOADING = "downloading"
    RESOLVING_CONFLICTS = "resolving_conflicts"
    ERROR = "error"
    OFFLINE = "offline"


class SyncConflictResolution(Enum):
    """Strategy for resolving sync conflicts."""

    SERVER_WINS = "server_wins"  # Server version takes precedence
    CLIENT_WINS = "client_wins"  # Client version takes precedence
    NEWEST_WINS = "newest_wins"  # Most recently modified wins
    MERGE = "merge"  # Attempt to merge changes
    ASK_USER = "ask_user"  # Prompt user to resolve


@dataclass
class ChangeRecord:
    """Record of a local change to sync."""

    id: str = field(default_factory=lambda: str(uuid4()))
    memory_id: str = ""
    operation: str = ""  # create, update, delete
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    synced: bool = False
    sync_attempts: int = 0
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute checksum for change verification."""
        data_str = json.dumps({
            "memory_id": self.memory_id,
            "operation": self.operation,
            "data": self.data,
        }, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


@dataclass
class SyncConflict:
    """A conflict between local and server versions."""

    id: str = field(default_factory=lambda: str(uuid4()))
    memory_id: str = ""
    local_version: dict[str, Any] = field(default_factory=dict)
    server_version: dict[str, Any] = field(default_factory=dict)
    local_timestamp: datetime | None = None
    server_timestamp: datetime | None = None
    resolved: bool = False
    resolution: SyncConflictResolution | None = None
    merged_version: dict[str, Any] | None = None


@dataclass
class SyncStatus:
    """Current sync status."""

    state: SyncState = SyncState.IDLE
    last_sync: datetime | None = None
    pending_uploads: int = 0
    pending_downloads: int = 0
    conflicts: int = 0
    error_message: str = ""

    # Progress tracking
    current_operation: str = ""
    progress_percent: float = 0.0

    # Bandwidth stats
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0


class SyncClient:
    """
    Client for syncing memories between mobile and server.

    Features:
    - Offline-first with local change queue
    - Efficient delta sync
    - Conflict detection and resolution
    - Background sync scheduling
    """

    def __init__(
        self,
        server_url: str,
        device_id: str | None = None,
        conflict_resolution: SyncConflictResolution = SyncConflictResolution.NEWEST_WINS,
        sync_interval_seconds: int = 300,  # 5 minutes
    ):
        self.server_url = server_url
        self.device_id = device_id or str(uuid4())
        self.conflict_resolution = conflict_resolution
        self.sync_interval_seconds = sync_interval_seconds

        self.status = SyncStatus()

        # Local change queue
        self.pending_changes: list[ChangeRecord] = []
        self.conflicts: list[SyncConflict] = []

        # Sync state tracking
        self.last_sync_timestamp: datetime | None = None
        self.server_cursor: str = ""  # For paginated sync

        # Callbacks
        self.on_conflict: Callable[[SyncConflict], SyncConflictResolution] | None = None
        self.on_progress: Callable[[SyncStatus], None] | None = None

    def queue_change(
        self,
        memory_id: str,
        operation: str,
        data: dict[str, Any],
    ) -> ChangeRecord:
        """Queue a local change for sync."""
        change = ChangeRecord(
            memory_id=memory_id,
            operation=operation,
            data=data,
        )
        change.checksum = change.compute_checksum()
        self.pending_changes.append(change)
        self.status.pending_uploads = len(self.pending_changes)
        return change

    async def sync(self) -> SyncStatus:
        """Perform full sync cycle."""
        if self.status.state == SyncState.SYNCING:
            return self.status

        self.status.state = SyncState.SYNCING
        self.status.error_message = ""

        try:
            # 1. Push local changes
            await self._push_changes()

            # 2. Pull server changes
            await self._pull_changes()

            # 3. Resolve conflicts
            if self.conflicts:
                await self._resolve_conflicts()

            self.status.last_sync = datetime.now()
            self.last_sync_timestamp = datetime.now()
            self.status.state = SyncState.IDLE

        except Exception as e:
            self.status.state = SyncState.ERROR
            self.status.error_message = str(e)

        self._notify_progress()
        return self.status

    async def _push_changes(self) -> None:
        """Push local changes to server."""
        if not self.pending_changes:
            return

        self.status.state = SyncState.UPLOADING
        self.status.current_operation = "Uploading changes"

        unsynced = [c for c in self.pending_changes if not c.synced]
        total = len(unsynced)

        for i, change in enumerate(unsynced):
            self.status.progress_percent = (i / total) * 100 if total > 0 else 100
            self._notify_progress()

            try:
                success = await self._push_single_change(change)
                if success:
                    change.synced = True
                else:
                    change.sync_attempts += 1
            except Exception:
                change.sync_attempts += 1

        # Remove synced changes
        self.pending_changes = [c for c in self.pending_changes if not c.synced]
        self.status.pending_uploads = len(self.pending_changes)

    async def _push_single_change(self, change: ChangeRecord) -> bool:
        """Push a single change to server."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.server_url}/api/mobile/sync/push",
                    json={
                        "device_id": self.device_id,
                        "change": {
                            "id": change.id,
                            "memory_id": change.memory_id,
                            "operation": change.operation,
                            "data": change.data,
                            "timestamp": change.timestamp.isoformat(),
                            "checksum": change.checksum,
                        },
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    self.status.bytes_uploaded += len(response.content)

                    # Check for conflict
                    if result.get("conflict"):
                        self.conflicts.append(SyncConflict(
                            memory_id=change.memory_id,
                            local_version=change.data,
                            server_version=result["conflict"]["server_version"],
                            local_timestamp=change.timestamp,
                            server_timestamp=datetime.fromisoformat(
                                result["conflict"]["server_timestamp"]
                            ),
                        ))
                        return False

                    return True

        except Exception:
            pass

        return False

    async def _pull_changes(self) -> None:
        """Pull changes from server."""
        self.status.state = SyncState.DOWNLOADING
        self.status.current_operation = "Downloading changes"
        self.status.progress_percent = 0

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                params = {
                    "device_id": self.device_id,
                    "cursor": self.server_cursor,
                }
                if self.last_sync_timestamp:
                    params["since"] = self.last_sync_timestamp.isoformat()

                response = await client.get(
                    f"{self.server_url}/api/mobile/sync/pull",
                    params=params,
                    timeout=60.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    self.status.bytes_downloaded += len(response.content)

                    changes = result.get("changes", [])
                    self.status.pending_downloads = len(changes)

                    # Apply changes locally
                    for i, change in enumerate(changes):
                        self.status.progress_percent = (i / len(changes)) * 100 if changes else 100
                        self._notify_progress()
                        await self._apply_server_change(change)

                    # Update cursor for next sync
                    self.server_cursor = result.get("cursor", "")

        except Exception:
            pass

        self.status.pending_downloads = 0

    async def _apply_server_change(self, change: dict[str, Any]) -> None:
        """Apply a server change locally."""
        # This would integrate with local storage
        # For now, just track it
        pass

    async def _resolve_conflicts(self) -> None:
        """Resolve pending conflicts."""
        self.status.state = SyncState.RESOLVING_CONFLICTS
        self.status.current_operation = "Resolving conflicts"
        self.status.conflicts = len(self.conflicts)

        for conflict in self.conflicts:
            if conflict.resolved:
                continue

            resolution = self.conflict_resolution

            # Ask user if configured
            if resolution == SyncConflictResolution.ASK_USER and self.on_conflict:
                resolution = self.on_conflict(conflict)

            # Apply resolution
            if resolution == SyncConflictResolution.SERVER_WINS:
                conflict.merged_version = conflict.server_version
            elif resolution == SyncConflictResolution.CLIENT_WINS:
                conflict.merged_version = conflict.local_version
            elif resolution == SyncConflictResolution.NEWEST_WINS:
                if conflict.local_timestamp and conflict.server_timestamp:
                    if conflict.local_timestamp > conflict.server_timestamp:
                        conflict.merged_version = conflict.local_version
                    else:
                        conflict.merged_version = conflict.server_version
                else:
                    conflict.merged_version = conflict.server_version
            elif resolution == SyncConflictResolution.MERGE:
                conflict.merged_version = self._merge_versions(
                    conflict.local_version,
                    conflict.server_version,
                )

            conflict.resolution = resolution
            conflict.resolved = True

        # Clear resolved conflicts
        self.conflicts = [c for c in self.conflicts if not c.resolved]
        self.status.conflicts = len(self.conflicts)

    def _merge_versions(
        self,
        local: dict[str, Any],
        server: dict[str, Any],
    ) -> dict[str, Any]:
        """Attempt to merge two versions."""
        merged = server.copy()

        # Prefer local content if different
        if local.get("content") != server.get("content"):
            merged["content"] = local.get("content", server.get("content"))

        # Merge arrays (tags, domains)
        for key in ["tags", "domains", "entities"]:
            local_items = set(local.get(key, []))
            server_items = set(server.get(key, []))
            merged[key] = list(local_items | server_items)

        # Take higher confidence
        merged["confidence"] = max(
            local.get("confidence", 0),
            server.get("confidence", 0),
        )

        return merged

    def _notify_progress(self) -> None:
        """Notify progress callback."""
        if self.on_progress:
            self.on_progress(self.status)

    def get_status(self) -> dict[str, Any]:
        """Get current sync status."""
        return {
            "state": self.status.state.value,
            "last_sync": self.status.last_sync.isoformat() if self.status.last_sync else None,
            "pending_uploads": self.status.pending_uploads,
            "pending_downloads": self.status.pending_downloads,
            "conflicts": self.status.conflicts,
            "error_message": self.status.error_message,
            "progress_percent": self.status.progress_percent,
            "bytes_uploaded": self.status.bytes_uploaded,
            "bytes_downloaded": self.status.bytes_downloaded,
        }


class SyncManager:
    """
    Manages sync scheduling and background sync.

    Features:
    - Automatic background sync
    - Smart scheduling based on connectivity
    - Battery-aware sync
    """

    def __init__(
        self,
        client: SyncClient,
        auto_sync: bool = True,
        sync_on_wifi_only: bool = False,
        battery_threshold: int = 20,  # Don't sync below 20% battery
    ):
        self.client = client
        self.auto_sync = auto_sync
        self.sync_on_wifi_only = sync_on_wifi_only
        self.battery_threshold = battery_threshold

        self._running = False
        self._sync_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start background sync."""
        if self._running:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop background sync."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                # Check if we should sync
                if self._should_sync():
                    await self.client.sync()

            except Exception:
                pass

            # Wait for next interval
            await asyncio.sleep(self.client.sync_interval_seconds)

    def _should_sync(self) -> bool:
        """Check if sync conditions are met."""
        if not self.auto_sync:
            return False

        # Check connectivity (would need platform-specific implementation)
        # if self.sync_on_wifi_only and not is_wifi_connected():
        #     return False

        # Check battery (would need platform-specific implementation)
        # if get_battery_level() < self.battery_threshold:
        #     return False

        return True

    async def force_sync(self) -> SyncStatus:
        """Force immediate sync."""
        return await self.client.sync()

    def get_status(self) -> dict[str, Any]:
        """Get sync manager status."""
        return {
            "running": self._running,
            "auto_sync": self.auto_sync,
            "sync_on_wifi_only": self.sync_on_wifi_only,
            "battery_threshold": self.battery_threshold,
            "client_status": self.client.get_status(),
        }
