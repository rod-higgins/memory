"""
Push Notification Service for PLM mobile companion app.

Supports:
- Apple Push Notification Service (APNs)
- Firebase Cloud Messaging (FCM)
- Notification scheduling
- Rich notifications
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class NotificationType(Enum):
    """Type of notification."""

    # Memory-related
    MEMORY_REMINDER = "memory_reminder"  # Spaced repetition reminder
    MEMORY_INSIGHT = "memory_insight"  # AI-generated insight
    MEMORY_CONFLICT = "memory_conflict"  # Sync conflict

    # Sync-related
    SYNC_COMPLETE = "sync_complete"
    SYNC_ERROR = "sync_error"
    SYNC_CONFLICT = "sync_conflict"

    # Learning-related
    LEARNING_STREAK = "learning_streak"
    SKILL_LEVELUP = "skill_levelup"
    CONCEPT_MASTERED = "concept_mastered"

    # System
    SYSTEM_UPDATE = "system_update"
    SYSTEM_ALERT = "system_alert"


class NotificationPriority(Enum):
    """Priority level for notifications."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class PushNotification:
    """A push notification to send."""

    id: str = field(default_factory=lambda: str(uuid4()))
    device_id: str = ""
    notification_type: NotificationType = NotificationType.SYSTEM_ALERT
    priority: NotificationPriority = NotificationPriority.NORMAL

    # Content
    title: str = ""
    body: str = ""
    subtitle: str = ""

    # iOS specific
    badge: int | None = None
    sound: str = "default"
    category: str = ""

    # Android specific
    channel_id: str = "default"
    icon: str = ""
    color: str = ""

    # Rich content
    image_url: str = ""
    action_url: str = ""

    # Data payload
    data: dict[str, Any] = field(default_factory=dict)

    # Scheduling
    scheduled_for: datetime | None = None
    expires_at: datetime | None = None

    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: datetime | None = None
    delivered_at: datetime | None = None
    opened_at: datetime | None = None

    def to_apns_payload(self) -> dict[str, Any]:
        """Convert to APNs payload format."""
        alert: dict[str, Any] = {
            "title": self.title,
            "body": self.body,
        }
        if self.subtitle:
            alert["subtitle"] = self.subtitle

        aps: dict[str, Any] = {
            "alert": alert,
            "sound": self.sound,
        }
        if self.badge is not None:
            aps["badge"] = self.badge
        if self.category:
            aps["category"] = self.category

        payload = {"aps": aps}
        if self.data:
            payload.update(self.data)

        return payload

    def to_fcm_payload(self) -> dict[str, Any]:
        """Convert to FCM payload format."""
        notification: dict[str, Any] = {
            "title": self.title,
            "body": self.body,
        }
        if self.image_url:
            notification["image"] = self.image_url

        android: dict[str, Any] = {
            "notification": {
                "channel_id": self.channel_id,
            },
        }
        if self.icon:
            android["notification"]["icon"] = self.icon
        if self.color:
            android["notification"]["color"] = self.color

        return {
            "notification": notification,
            "android": android,
            "data": self.data,
        }


@dataclass
class NotificationChannel:
    """Android notification channel configuration."""

    id: str
    name: str
    description: str = ""
    importance: str = "default"  # min, low, default, high, max
    sound: str = "default"
    vibration: bool = True
    show_badge: bool = True


class NotificationService:
    """
    Service for sending push notifications.

    Supports:
    - APNs for iOS
    - FCM for Android
    - Scheduled notifications
    - Notification analytics
    """

    def __init__(
        self,
        apns_key_path: str | None = None,
        apns_key_id: str | None = None,
        apns_team_id: str | None = None,
        apns_bundle_id: str | None = None,
        fcm_credentials_path: str | None = None,
        use_sandbox: bool = True,
    ):
        # APNs configuration
        self.apns_key_path = apns_key_path
        self.apns_key_id = apns_key_id
        self.apns_team_id = apns_team_id
        self.apns_bundle_id = apns_bundle_id
        self.use_sandbox = use_sandbox

        # FCM configuration
        self.fcm_credentials_path = fcm_credentials_path

        # Notification queue
        self.pending: list[PushNotification] = []
        self.sent: list[PushNotification] = []
        self.failed: list[tuple[PushNotification, str]] = []

        # Scheduled notifications
        self.scheduled: list[PushNotification] = []
        self._scheduler_task: asyncio.Task | None = None

        # Device tokens
        self.device_tokens: dict[str, dict[str, str]] = {}  # device_id -> {platform, token}

        # Callbacks
        self.on_sent: Callable[[PushNotification], None] | None = None
        self.on_failed: Callable[[PushNotification, str], None] | None = None

    def register_device_token(
        self,
        device_id: str,
        platform: str,
        token: str,
    ) -> None:
        """Register a device's push token."""
        self.device_tokens[device_id] = {
            "platform": platform,
            "token": token,
        }

    def unregister_device(self, device_id: str) -> None:
        """Unregister a device's push token."""
        if device_id in self.device_tokens:
            del self.device_tokens[device_id]

    async def send(
        self,
        notification: PushNotification,
    ) -> bool:
        """Send a push notification immediately."""
        device_info = self.device_tokens.get(notification.device_id)
        if not device_info:
            self.failed.append((notification, "Device not registered"))
            return False

        platform = device_info["platform"]
        token = device_info["token"]

        try:
            if platform == "ios":
                success = await self._send_apns(notification, token)
            elif platform == "android":
                success = await self._send_fcm(notification, token)
            else:
                success = False

            if success:
                notification.sent_at = datetime.now()
                self.sent.append(notification)
                if self.on_sent:
                    self.on_sent(notification)
            else:
                self.failed.append((notification, "Send failed"))
                if self.on_failed:
                    self.on_failed(notification, "Send failed")

            return success

        except Exception as e:
            self.failed.append((notification, str(e)))
            if self.on_failed:
                self.on_failed(notification, str(e))
            return False

    async def _send_apns(
        self,
        notification: PushNotification,
        token: str,
    ) -> bool:
        """Send notification via APNs."""
        if not all(
            [
                self.apns_key_path,
                self.apns_key_id,
                self.apns_team_id,
                self.apns_bundle_id,
            ]
        ):
            return False

        try:
            import time

            import httpx
            import jwt

            # Generate JWT token
            with open(self.apns_key_path) as f:
                private_key = f.read()

            token_payload = {
                "iss": self.apns_team_id,
                "iat": int(time.time()),
            }
            jwt_token = jwt.encode(
                token_payload,
                private_key,
                algorithm="ES256",
                headers={"kid": self.apns_key_id},
            )

            # Determine endpoint
            if self.use_sandbox:
                host = "api.sandbox.push.apple.com"
            else:
                host = "api.push.apple.com"

            url = f"https://{host}/3/device/{token}"

            # Build payload
            payload = notification.to_apns_payload()

            async with httpx.AsyncClient(http2=True) as client:
                response = await client.post(
                    url,
                    headers={
                        "authorization": f"bearer {jwt_token}",
                        "apns-topic": self.apns_bundle_id,
                        "apns-priority": "10"
                        if notification.priority
                        in [
                            NotificationPriority.HIGH,
                            NotificationPriority.URGENT,
                        ]
                        else "5",
                    },
                    json=payload,
                    timeout=30.0,
                )

                return response.status_code == 200

        except ImportError:
            # jwt library not available
            return False
        except Exception:
            return False

    async def _send_fcm(
        self,
        notification: PushNotification,
        token: str,
    ) -> bool:
        """Send notification via FCM."""
        if not self.fcm_credentials_path:
            return False

        try:
            import httpx

            # Load service account credentials
            with open(self.fcm_credentials_path) as f:
                creds = json.load(f)

            project_id = creds.get("project_id")
            if not project_id:
                return False

            # Get access token (simplified - real impl would use google-auth)
            # For production, use google-auth library
            access_token = await self._get_fcm_access_token(creds)
            if not access_token:
                return False

            url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

            payload = {
                "message": {
                    "token": token,
                    **notification.to_fcm_payload(),
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30.0,
                )

                return response.status_code == 200

        except Exception:
            return False

    async def _get_fcm_access_token(
        self,
        credentials: dict[str, Any],
    ) -> str | None:
        """Get FCM access token from service account."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_info(
                credentials,
                scopes=["https://www.googleapis.com/auth/firebase.messaging"],
            )
            creds.refresh(Request())
            return creds.token

        except ImportError:
            # google-auth not available
            return None
        except Exception:
            return None

    def schedule(
        self,
        notification: PushNotification,
        send_at: datetime,
    ) -> None:
        """Schedule a notification for later."""
        notification.scheduled_for = send_at
        self.scheduled.append(notification)

    def cancel_scheduled(
        self,
        notification_id: str,
    ) -> bool:
        """Cancel a scheduled notification."""
        for i, notif in enumerate(self.scheduled):
            if notif.id == notification_id:
                self.scheduled.pop(i)
                return True
        return False

    async def start_scheduler(self) -> None:
        """Start the notification scheduler."""
        if self._scheduler_task:
            return

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self) -> None:
        """Stop the notification scheduler."""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

    async def _scheduler_loop(self) -> None:
        """Background loop for scheduled notifications."""
        while True:
            try:
                now = datetime.now()
                to_send = []

                for notif in self.scheduled:
                    if notif.scheduled_for and notif.scheduled_for <= now:
                        # Check expiry
                        if notif.expires_at and notif.expires_at < now:
                            continue
                        to_send.append(notif)

                for notif in to_send:
                    self.scheduled.remove(notif)
                    await self.send(notif)

            except Exception:
                pass

            await asyncio.sleep(60)  # Check every minute

    # Convenience methods for common notifications

    def create_memory_reminder(
        self,
        device_id: str,
        memory_id: str,
        memory_content: str,
    ) -> PushNotification:
        """Create a spaced repetition reminder notification."""
        return PushNotification(
            device_id=device_id,
            notification_type=NotificationType.MEMORY_REMINDER,
            priority=NotificationPriority.NORMAL,
            title="Memory Review",
            body=f"Time to review: {memory_content[:50]}...",
            category="MEMORY_REVIEW",
            data={
                "type": "memory_reminder",
                "memory_id": memory_id,
            },
        )

    def create_sync_complete(
        self,
        device_id: str,
        memories_synced: int,
    ) -> PushNotification:
        """Create a sync complete notification."""
        return PushNotification(
            device_id=device_id,
            notification_type=NotificationType.SYNC_COMPLETE,
            priority=NotificationPriority.LOW,
            title="Sync Complete",
            body=f"Successfully synced {memories_synced} memories",
            data={
                "type": "sync_complete",
                "count": memories_synced,
            },
        )

    def create_learning_streak(
        self,
        device_id: str,
        streak_days: int,
    ) -> PushNotification:
        """Create a learning streak notification."""
        return PushNotification(
            device_id=device_id,
            notification_type=NotificationType.LEARNING_STREAK,
            priority=NotificationPriority.NORMAL,
            title="Learning Streak!",
            body=f"You're on a {streak_days}-day learning streak!",
            data={
                "type": "learning_streak",
                "days": streak_days,
            },
        )

    def create_skill_levelup(
        self,
        device_id: str,
        skill_name: str,
        new_level: int,
    ) -> PushNotification:
        """Create a skill level-up notification."""
        return PushNotification(
            device_id=device_id,
            notification_type=NotificationType.SKILL_LEVELUP,
            priority=NotificationPriority.HIGH,
            title="Level Up!",
            body=f"You've reached level {new_level} in {skill_name}!",
            data={
                "type": "skill_levelup",
                "skill": skill_name,
                "level": new_level,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get notification statistics."""
        return {
            "registered_devices": len(self.device_tokens),
            "pending": len(self.pending),
            "scheduled": len(self.scheduled),
            "sent_today": len([n for n in self.sent if n.sent_at and n.sent_at.date() == datetime.now().date()]),
            "failed_today": len([(n, _) for n, _ in self.failed if n.created_at.date() == datetime.now().date()]),
        }
