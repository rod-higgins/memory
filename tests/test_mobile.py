"""Tests for mobile companion app module."""

from datetime import datetime, timedelta


class TestSyncClient:
    """Tests for mobile sync client."""

    def test_client_initialization(self):
        """Test sync client initialization."""
        from memory.mobile import SyncClient

        client = SyncClient(
            server_url="http://localhost:8000",
            device_id="test-device",
        )

        assert client is not None
        assert client.device_id == "test-device"
        assert client.server_url == "http://localhost:8000"

    def test_queue_change(self):
        """Test queuing a change for sync."""
        from memory.mobile import SyncClient

        client = SyncClient(server_url="http://localhost:8000")

        change = client.queue_change(
            memory_id="mem-123",
            operation="create",
            data={"content": "Test memory"},
        )

        assert change is not None
        assert change.memory_id == "mem-123"
        assert change.operation == "create"
        assert len(client.pending_changes) == 1

    def test_queue_multiple_changes(self):
        """Test queuing multiple changes."""
        from memory.mobile import SyncClient

        client = SyncClient(server_url="http://localhost:8000")

        for i in range(5):
            client.queue_change(
                memory_id=f"mem-{i}",
                operation="create",
                data={"content": f"Memory {i}"},
            )

        assert len(client.pending_changes) == 5
        assert client.status.pending_uploads == 5

    def test_get_status(self):
        """Test getting sync status."""
        from memory.mobile import SyncClient, SyncState

        client = SyncClient(server_url="http://localhost:8000")

        status = client.get_status()

        assert "state" in status
        assert status["state"] == SyncState.IDLE.value
        assert "pending_uploads" in status
        assert "pending_downloads" in status


class TestSyncConflictResolution:
    """Tests for sync conflict resolution."""

    def test_conflict_resolution_strategies(self):
        """Test all conflict resolution strategies exist."""
        from memory.mobile import SyncConflictResolution

        assert SyncConflictResolution.SERVER_WINS.value == "server_wins"
        assert SyncConflictResolution.CLIENT_WINS.value == "client_wins"
        assert SyncConflictResolution.NEWEST_WINS.value == "newest_wins"
        assert SyncConflictResolution.MERGE.value == "merge"
        assert SyncConflictResolution.ASK_USER.value == "ask_user"

    def test_default_resolution(self):
        """Test default conflict resolution."""
        from memory.mobile import SyncClient, SyncConflictResolution

        client = SyncClient(
            server_url="http://localhost:8000",
            conflict_resolution=SyncConflictResolution.NEWEST_WINS,
        )

        assert client.conflict_resolution == SyncConflictResolution.NEWEST_WINS


class TestSyncManager:
    """Tests for sync manager."""

    def test_manager_initialization(self):
        """Test sync manager initialization."""
        from memory.mobile import SyncClient, SyncManager

        client = SyncClient(server_url="http://localhost:8000")
        manager = SyncManager(
            client=client,
            auto_sync=True,
            sync_on_wifi_only=False,
        )

        assert manager is not None
        assert manager.auto_sync is True

    def test_manager_status(self):
        """Test getting manager status."""
        from memory.mobile import SyncClient, SyncManager

        client = SyncClient(server_url="http://localhost:8000")
        manager = SyncManager(client=client)

        status = manager.get_status()

        assert "running" in status
        assert "auto_sync" in status
        assert "client_status" in status


class TestMobileAPI:
    """Tests for mobile API."""

    def test_api_initialization(self):
        """Test mobile API initialization."""
        from memory.mobile import MobileAPI

        api = MobileAPI()

        assert api is not None
        assert len(api.devices) == 0

    def test_register_device(self):
        """Test device registration."""
        from memory.mobile import MobileAPI

        api = MobileAPI()

        device = api.register_device(
            device_type="ios",
            device_name="Test iPhone",
            os_version="17.0",
            app_version="1.0.0",
        )

        assert device is not None
        assert device.device_name == "Test iPhone"
        assert device.auth_token != ""
        assert device.device_id in api.devices

    def test_authenticate_device(self):
        """Test device authentication."""
        from memory.mobile import MobileAPI

        api = MobileAPI()

        # Register first
        device = api.register_device(device_type="android")

        # Authenticate
        session = api.authenticate_device(device.device_id, device.auth_token)

        assert session is not None
        assert session.device_id == device.device_id

    def test_authenticate_invalid_token(self):
        """Test authentication with invalid token."""
        from memory.mobile import MobileAPI

        api = MobileAPI()
        device = api.register_device(device_type="ios")

        session = api.authenticate_device(device.device_id, "wrong-token")
        assert session is None

    def test_unregister_device(self):
        """Test device unregistration."""
        from memory.mobile import MobileAPI

        api = MobileAPI()
        device = api.register_device(device_type="ios")

        success = api.unregister_device(device.device_id)
        assert success
        assert device.device_id not in api.devices

        # Unregistering again should fail
        success = api.unregister_device(device.device_id)
        assert not success

    def test_refresh_token(self):
        """Test token refresh."""
        from memory.mobile import MobileAPI

        api = MobileAPI()
        device = api.register_device(device_type="ios")
        old_token = device.auth_token

        new_token = api.refresh_token(device.device_id)

        assert new_token is not None
        assert new_token != old_token

    def test_quick_capture(self):
        """Test quick memory capture."""
        from memory.mobile import MobileAPI

        api = MobileAPI()
        device = api.register_device(device_type="ios")

        result = api.quick_capture(
            device_id=device.device_id,
            content="Quick thought to remember",
            source="voice_memo",
        )

        assert result["success"]
        assert "memory_id" in result

    def test_quick_capture_invalid_device(self):
        """Test quick capture with invalid device."""
        from memory.mobile import MobileAPI

        api = MobileAPI()

        result = api.quick_capture(
            device_id="nonexistent",
            content="Test",
        )

        assert not result["success"]

    def test_update_push_token(self):
        """Test updating push notification token."""
        from memory.mobile import MobileAPI

        api = MobileAPI()
        device = api.register_device(device_type="ios")

        success = api.update_push_token(
            device_id=device.device_id,
            push_token="new-push-token-123",
        )

        assert success
        assert api.devices[device.device_id].push_token == "new-push-token-123"


class TestNotificationService:
    """Tests for push notification service."""

    def test_service_initialization(self):
        """Test notification service initialization."""
        from memory.mobile import NotificationService

        service = NotificationService()

        assert service is not None
        assert len(service.device_tokens) == 0

    def test_register_device_token(self):
        """Test registering device push token."""
        from memory.mobile import NotificationService

        service = NotificationService()

        service.register_device_token(
            device_id="device-1",
            platform="ios",
            token="apns-token-123",
        )

        assert "device-1" in service.device_tokens
        assert service.device_tokens["device-1"]["platform"] == "ios"

    def test_unregister_device(self):
        """Test unregistering device."""
        from memory.mobile import NotificationService

        service = NotificationService()
        service.register_device_token("device-1", "ios", "token")

        service.unregister_device("device-1")
        assert "device-1" not in service.device_tokens

    def test_create_notification(self):
        """Test creating a notification."""
        from memory.mobile import NotificationType, PushNotification

        notification = PushNotification(
            device_id="device-1",
            notification_type=NotificationType.MEMORY_REMINDER,
            title="Memory Review",
            body="Time to review your memories",
        )

        assert notification.device_id == "device-1"
        assert notification.title == "Memory Review"

    def test_notification_to_apns_payload(self):
        """Test converting notification to APNs format."""
        from memory.mobile import NotificationType, PushNotification

        notification = PushNotification(
            device_id="device-1",
            notification_type=NotificationType.MEMORY_REMINDER,
            title="Test",
            body="Test body",
            badge=5,
        )

        payload = notification.to_apns_payload()

        assert "aps" in payload
        assert payload["aps"]["alert"]["title"] == "Test"
        assert payload["aps"]["badge"] == 5

    def test_notification_to_fcm_payload(self):
        """Test converting notification to FCM format."""
        from memory.mobile import NotificationType, PushNotification

        notification = PushNotification(
            device_id="device-1",
            notification_type=NotificationType.SYNC_COMPLETE,
            title="Sync Complete",
            body="Your memories are synced",
        )

        payload = notification.to_fcm_payload()

        assert "notification" in payload
        assert payload["notification"]["title"] == "Sync Complete"

    def test_schedule_notification(self):
        """Test scheduling a notification."""
        from memory.mobile import NotificationService, NotificationType, PushNotification

        service = NotificationService()

        notification = PushNotification(
            device_id="device-1",
            notification_type=NotificationType.MEMORY_REMINDER,
            title="Scheduled",
            body="This was scheduled",
        )

        send_at = datetime.now() + timedelta(hours=1)
        service.schedule(notification, send_at)

        assert len(service.scheduled) == 1
        assert service.scheduled[0].scheduled_for == send_at

    def test_cancel_scheduled(self):
        """Test canceling scheduled notification."""
        from memory.mobile import NotificationService, NotificationType, PushNotification

        service = NotificationService()

        notification = PushNotification(
            device_id="device-1",
            notification_type=NotificationType.MEMORY_REMINDER,
            title="To Cancel",
            body="Will be canceled",
        )

        service.schedule(notification, datetime.now() + timedelta(hours=1))
        assert len(service.scheduled) == 1

        success = service.cancel_scheduled(notification.id)
        assert success
        assert len(service.scheduled) == 0

    def test_notification_convenience_methods(self):
        """Test convenience methods for common notifications."""
        from memory.mobile import NotificationService

        service = NotificationService()

        # Memory reminder
        reminder = service.create_memory_reminder(
            device_id="device-1",
            memory_id="mem-123",
            memory_content="Python preference",
        )
        assert "Memory Review" in reminder.title

        # Sync complete
        sync = service.create_sync_complete(
            device_id="device-1",
            memories_synced=50,
        )
        assert "50" in sync.body

        # Learning streak
        streak = service.create_learning_streak(
            device_id="device-1",
            streak_days=7,
        )
        assert "7" in streak.body

    def test_notification_stats(self):
        """Test getting notification statistics."""
        from memory.mobile import NotificationService

        service = NotificationService()
        service.register_device_token("d1", "ios", "t1")
        service.register_device_token("d2", "android", "t2")

        stats = service.get_stats()

        assert stats["registered_devices"] == 2
        assert "pending" in stats
        assert "scheduled" in stats


class TestDeviceTypes:
    """Tests for device type handling."""

    def test_device_type_enum(self):
        """Test device type enumeration."""
        from memory.mobile.api import DeviceType

        assert DeviceType.IOS.value == "ios"
        assert DeviceType.ANDROID.value == "android"
        assert DeviceType.WEB.value == "web"
        assert DeviceType.DESKTOP.value == "desktop"

    def test_device_info(self):
        """Test device info dataclass."""
        from memory.mobile import DeviceInfo

        device = DeviceInfo(
            device_name="Test Device",
            os_version="17.0",
            app_version="1.0.0",
        )

        assert device.device_name == "Test Device"
        assert device.device_id is not None

    def test_device_info_to_dict(self):
        """Test device info serialization."""
        from memory.mobile import DeviceInfo

        device = DeviceInfo(device_name="Test")
        data = device.to_dict()

        assert "device_id" in data
        assert "device_name" in data
        assert "registered_at" in data
