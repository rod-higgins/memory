"""
Mobile Companion App Foundation for PLM.

Provides the backend support for mobile applications:
- Sync API for mobile clients
- Offline-first data structures
- Push notification integration
- Mobile-optimized queries
"""

from .api import (
    DeviceInfo,
    MobileAPI,
    MobileSession,
)
from .notifications import (
    NotificationService,
    NotificationType,
    PushNotification,
)
from .sync import (
    SyncClient,
    SyncConflictResolution,
    SyncManager,
    SyncState,
)

__all__ = [
    # Sync
    "SyncClient",
    "SyncManager",
    "SyncState",
    "SyncConflictResolution",
    # API
    "MobileAPI",
    "MobileSession",
    "DeviceInfo",
    # Notifications
    "NotificationService",
    "PushNotification",
    "NotificationType",
]
