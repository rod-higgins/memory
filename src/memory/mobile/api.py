"""
Mobile API for PLM companion app.

Provides mobile-optimized endpoints:
- Device registration and authentication
- Efficient sync endpoints
- Mobile-specific memory queries
- Quick capture support
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4


class DeviceType(Enum):
    """Type of mobile device."""

    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    DESKTOP = "desktop"
    UNKNOWN = "unknown"


@dataclass
class DeviceInfo:
    """Information about a registered device."""

    device_id: str = field(default_factory=lambda: str(uuid4()))
    device_type: DeviceType = DeviceType.UNKNOWN
    device_name: str = ""
    os_version: str = ""
    app_version: str = ""

    # Registration
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    # Authentication
    auth_token: str = ""
    token_expires: datetime | None = None

    # Sync state
    last_sync: datetime | None = None
    sync_cursor: str = ""

    # Push notification token
    push_token: str = ""
    push_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize device info."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "device_name": self.device_name,
            "os_version": self.os_version,
            "app_version": self.app_version,
            "registered_at": self.registered_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "push_enabled": self.push_enabled,
        }


@dataclass
class MobileSession:
    """A mobile app session."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    device_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))

    # Activity tracking
    queries_count: int = 0
    memories_created: int = 0
    memories_viewed: int = 0

    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return datetime.now() < self.expires_at


class MobileAPI:
    """
    API service for mobile companion app.

    Provides:
    - Device registration and authentication
    - Mobile-optimized sync endpoints
    - Quick capture for memories
    - Contextual memory queries
    """

    def __init__(
        self,
        token_expiry_days: int = 30,
        max_devices_per_user: int = 5,
    ):
        self.token_expiry_days = token_expiry_days
        self.max_devices_per_user = max_devices_per_user

        # Device registry
        self.devices: dict[str, DeviceInfo] = {}
        self.sessions: dict[str, MobileSession] = {}

        # Sync state per device
        self.pending_changes: dict[str, list[dict[str, Any]]] = {}

    def register_device(
        self,
        device_type: str,
        device_name: str = "",
        os_version: str = "",
        app_version: str = "",
    ) -> DeviceInfo:
        """Register a new mobile device."""
        device = DeviceInfo(
            device_type=DeviceType(device_type) if device_type in [e.value for e in DeviceType] else DeviceType.UNKNOWN,
            device_name=device_name,
            os_version=os_version,
            app_version=app_version,
        )

        # Generate auth token
        device.auth_token = self._generate_token()
        device.token_expires = datetime.now() + timedelta(days=self.token_expiry_days)

        self.devices[device.device_id] = device
        self.pending_changes[device.device_id] = []

        return device

    def authenticate_device(
        self,
        device_id: str,
        auth_token: str,
    ) -> MobileSession | None:
        """Authenticate a device and create session."""
        device = self.devices.get(device_id)
        if not device:
            return None

        # Verify token
        if device.auth_token != auth_token:
            return None

        # Check token expiry
        if device.token_expires and datetime.now() > device.token_expires:
            return None

        # Update last seen
        device.last_seen = datetime.now()

        # Create session
        session = MobileSession(device_id=device_id)
        self.sessions[session.session_id] = session

        return session

    def refresh_token(self, device_id: str) -> str | None:
        """Refresh device authentication token."""
        device = self.devices.get(device_id)
        if not device:
            return None

        device.auth_token = self._generate_token()
        device.token_expires = datetime.now() + timedelta(days=self.token_expiry_days)

        return device.auth_token

    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device."""
        if device_id in self.devices:
            del self.devices[device_id]
            if device_id in self.pending_changes:
                del self.pending_changes[device_id]
            return True
        return False

    def _generate_token(self) -> str:
        """Generate a secure auth token."""
        return secrets.token_urlsafe(32)

    def update_push_token(
        self,
        device_id: str,
        push_token: str,
    ) -> bool:
        """Update device push notification token."""
        device = self.devices.get(device_id)
        if not device:
            return False

        device.push_token = push_token
        return True

    def get_sync_changes(
        self,
        device_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get changes for a device to sync."""
        device = self.devices.get(device_id)
        if not device:
            return {"changes": [], "cursor": ""}

        # Get pending changes
        changes = self.pending_changes.get(device_id, [])

        if since:
            changes = [c for c in changes if datetime.fromisoformat(c["timestamp"]) > since]

        # Limit results
        changes = changes[:limit]

        # Update cursor
        cursor = ""
        if changes:
            cursor = changes[-1].get("id", "")

        return {
            "changes": changes,
            "cursor": cursor,
            "has_more": len(self.pending_changes.get(device_id, [])) > limit,
        }

    def push_change(
        self,
        device_id: str,
        change: dict[str, Any],
    ) -> dict[str, Any]:
        """Receive a change from a device."""
        device = self.devices.get(device_id)
        if not device:
            return {"success": False, "error": "Device not found"}

        device.last_sync = datetime.now()

        # Check for conflicts with other devices
        conflict = self._check_conflict(device_id, change)
        if conflict:
            return {
                "success": False,
                "conflict": conflict,
            }

        # Broadcast to other devices
        self._broadcast_change(device_id, change)

        return {"success": True}

    def _check_conflict(
        self,
        device_id: str,
        change: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Check if a change conflicts with pending changes."""
        memory_id = change.get("memory_id")
        if not memory_id:
            return None

        # Check other devices for concurrent changes to same memory
        for other_device_id, changes in self.pending_changes.items():
            if other_device_id == device_id:
                continue

            for pending in changes:
                if pending.get("memory_id") == memory_id:
                    # Found concurrent change
                    return {
                        "memory_id": memory_id,
                        "server_version": pending.get("data", {}),
                        "server_timestamp": pending.get("timestamp"),
                    }

        return None

    def _broadcast_change(
        self,
        source_device_id: str,
        change: dict[str, Any],
    ) -> None:
        """Broadcast a change to other devices."""
        for device_id in self.devices:
            if device_id != source_device_id:
                if device_id not in self.pending_changes:
                    self.pending_changes[device_id] = []
                self.pending_changes[device_id].append(change)

    def quick_capture(
        self,
        device_id: str,
        content: str,
        source: str = "mobile_capture",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Quick capture a memory from mobile."""
        device = self.devices.get(device_id)
        if not device:
            return {"success": False, "error": "Device not found"}

        # Create memory entry
        memory_id = str(uuid4())
        memory = {
            "id": memory_id,
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "metadata": metadata or {},
        }

        # Broadcast to all devices
        change = {
            "id": str(uuid4()),
            "memory_id": memory_id,
            "operation": "create",
            "data": memory,
            "timestamp": datetime.now().isoformat(),
        }
        self._broadcast_change(device_id, change)

        return {
            "success": True,
            "memory_id": memory_id,
        }

    def get_context_memories(
        self,
        device_id: str,
        context: dict[str, Any],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get memories relevant to current mobile context.

        Context can include:
        - location: Current GPS coordinates
        - time: Current time/date
        - activity: Current user activity
        - app: Current active app
        """
        # This would integrate with the main memory search
        # For now, return empty list
        return []

    def get_device_stats(self, device_id: str) -> dict[str, Any]:
        """Get statistics for a device."""
        device = self.devices.get(device_id)
        if not device:
            return {}

        session = None
        for s in self.sessions.values():
            if s.device_id == device_id and s.is_valid():
                session = s
                break

        return {
            "device": device.to_dict(),
            "pending_sync": len(self.pending_changes.get(device_id, [])),
            "session": {
                "active": session is not None,
                "queries": session.queries_count if session else 0,
                "memories_created": session.memories_created if session else 0,
            }
            if session
            else None,
        }


# FastAPI routes (when available)
def create_mobile_routes():
    """Create FastAPI routes for mobile API."""
    try:
        from fastapi import APIRouter, Header, HTTPException
        from pydantic import BaseModel

        router = APIRouter(prefix="/api/mobile", tags=["mobile"])
        api = MobileAPI()

        class DeviceRegistration(BaseModel):
            device_type: str
            device_name: str = ""
            os_version: str = ""
            app_version: str = ""

        class AuthRequest(BaseModel):
            device_id: str
            auth_token: str

        class PushTokenUpdate(BaseModel):
            push_token: str

        class QuickCapture(BaseModel):
            content: str
            source: str = "mobile_capture"
            metadata: dict[str, Any] = {}

        class ContextQuery(BaseModel):
            context: dict[str, Any]
            limit: int = 10

        @router.post("/devices/register")
        async def register_device(reg: DeviceRegistration):
            device = api.register_device(
                device_type=reg.device_type,
                device_name=reg.device_name,
                os_version=reg.os_version,
                app_version=reg.app_version,
            )
            return {
                "device_id": device.device_id,
                "auth_token": device.auth_token,
                "expires": device.token_expires.isoformat() if device.token_expires else None,
            }

        @router.post("/devices/authenticate")
        async def authenticate(auth: AuthRequest):
            session = api.authenticate_device(auth.device_id, auth.auth_token)
            if not session:
                raise HTTPException(status_code=401, detail="Authentication failed")
            return {
                "session_id": session.session_id,
                "expires": session.expires_at.isoformat(),
            }

        @router.delete("/devices/{device_id}")
        async def unregister(device_id: str):
            success = api.unregister_device(device_id)
            if not success:
                raise HTTPException(status_code=404, detail="Device not found")
            return {"success": True}

        @router.put("/devices/{device_id}/push-token")
        async def update_push_token(device_id: str, update: PushTokenUpdate):
            success = api.update_push_token(device_id, update.push_token)
            if not success:
                raise HTTPException(status_code=404, detail="Device not found")
            return {"success": True}

        @router.get("/sync/pull")
        async def pull_changes(
            device_id: str,
            cursor: str = "",
            since: str = "",
            limit: int = 100,
        ):
            since_dt = None
            if since:
                try:
                    since_dt = datetime.fromisoformat(since)
                except ValueError:
                    pass
            return api.get_sync_changes(device_id, since_dt, limit)

        @router.post("/sync/push")
        async def push_change(
            device_id: str = Header(..., alias="X-Device-ID"),
        ):
            # Body would contain the change
            return {"success": True}

        @router.post("/capture")
        async def quick_capture(
            capture: QuickCapture,
            device_id: str = Header(..., alias="X-Device-ID"),
        ):
            result = api.quick_capture(
                device_id=device_id,
                content=capture.content,
                source=capture.source,
                metadata=capture.metadata,
            )
            if not result["success"]:
                raise HTTPException(status_code=400, detail=result.get("error"))
            return result

        @router.post("/context")
        async def get_context_memories(
            query: ContextQuery,
            device_id: str = Header(..., alias="X-Device-ID"),
        ):
            memories = api.get_context_memories(
                device_id=device_id,
                context=query.context,
                limit=query.limit,
            )
            return {"memories": memories}

        @router.get("/devices/{device_id}/stats")
        async def get_device_stats(device_id: str):
            stats = api.get_device_stats(device_id)
            if not stats:
                raise HTTPException(status_code=404, detail="Device not found")
            return stats

        return router

    except ImportError:
        return None
