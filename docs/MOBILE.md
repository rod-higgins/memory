# Mobile Companion App

## Overview

The mobile companion app provides on-the-go access to your personal memory system with offline-first architecture and seamless synchronization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MOBILE ARCHITECTURE                                │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      MOBILE APP                                  │   │
│   │                                                                  │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │   │
│   │   │  Quick   │  │  Memory  │  │  Search  │  │ Settings │        │   │
│   │   │ Capture  │  │  Browse  │  │  Query   │  │  Sync    │        │   │
│   │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │   │
│   │        │             │             │             │               │   │
│   │        └─────────────┴─────────────┴─────────────┘               │   │
│   │                              │                                   │   │
│   │                    ┌─────────▼─────────┐                         │   │
│   │                    │   Local Storage   │                         │   │
│   │                    │   (SQLite/Realm)  │                         │   │
│   │                    └─────────┬─────────┘                         │   │
│   └──────────────────────────────┼──────────────────────────────────┘   │
│                                  │                                      │
│                         ┌────────▼────────┐                             │
│                         │   Sync Engine   │                             │
│                         │                 │                             │
│                         │ • Offline-first │                             │
│                         │ • Conflict res. │                             │
│                         │ • Delta sync    │                             │
│                         └────────┬────────┘                             │
│                                  │                                      │
│                         ┌────────▼────────┐                             │
│                         │   Server API    │                             │
│                         │   (REST/gRPC)   │                             │
│                         └────────┬────────┘                             │
│                                  │                                      │
│                         ┌────────▼────────┐                             │
│                         │  Desktop/Cloud  │                             │
│                         │  Memory System  │                             │
│                         └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. SyncClient

Handles synchronization between mobile and server.

```python
from memory.mobile import SyncClient, SyncConflictResolution

client = SyncClient(
    server_url="http://localhost:8000",
    device_id="my-iphone",
    conflict_resolution=SyncConflictResolution.NEWEST_WINS,
)

# Queue a change for sync
change = client.queue_change(
    memory_id="mem-123",
    operation="create",
    data={"content": "Quick thought from mobile"},
)

# Get sync status
status = client.get_status()
print(f"Pending uploads: {status['pending_uploads']}")
print(f"Pending downloads: {status['pending_downloads']}")

# Sync with server
await client.sync()
```

### 2. SyncManager

Manages background sync operations.

```python
from memory.mobile import SyncClient, SyncManager

client = SyncClient(server_url="http://localhost:8000")

manager = SyncManager(
    client=client,
    auto_sync=True,
    sync_interval=300,        # Sync every 5 minutes
    sync_on_wifi_only=True,   # Conserve mobile data
)

# Start background sync
await manager.start()

# Get manager status
status = manager.get_status()
print(f"Running: {status['running']}")
print(f"Last sync: {status['last_sync']}")

# Stop when done
await manager.stop()
```

### 3. MobileAPI

Server-side API for mobile clients.

```python
from memory.mobile import MobileAPI

api = MobileAPI()

# Register a new device
device = api.register_device(
    device_type="ios",
    device_name="iPhone 15",
    os_version="17.0",
    app_version="1.0.0",
)
print(f"Device ID: {device.device_id}")
print(f"Auth token: {device.auth_token}")

# Authenticate device
session = api.authenticate_device(
    device_id=device.device_id,
    auth_token=device.auth_token,
)

# Quick capture from mobile
result = api.quick_capture(
    device_id=device.device_id,
    content="Remember to review the proposal",
    source="voice_memo",
)
print(f"Memory ID: {result['memory_id']}")

# Update push notification token
api.update_push_token(
    device_id=device.device_id,
    push_token="apns-token-xyz",
)
```

### 4. NotificationService

Push notification management.

```python
from memory.mobile import (
    NotificationService,
    PushNotification,
    NotificationType,
)

service = NotificationService()

# Register device for notifications
service.register_device_token(
    device_id="device-123",
    platform="ios",
    token="apns-token-xyz",
)

# Create and send notification
notification = PushNotification(
    device_id="device-123",
    notification_type=NotificationType.MEMORY_REMINDER,
    title="Memory Review",
    body="Time to review your memories from last week",
    badge=5,
)

await service.send(notification)

# Schedule notification
from datetime import datetime, timedelta

service.schedule(
    notification,
    send_at=datetime.now() + timedelta(hours=2),
)

# Convenience methods
reminder = service.create_memory_reminder(
    device_id="device-123",
    memory_id="mem-456",
    memory_content="Python best practices",
)

sync_complete = service.create_sync_complete(
    device_id="device-123",
    memories_synced=50,
)
```

## Sync Protocol

### Conflict Resolution Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SERVER_WINS` | Server version always wins | Centralized authority |
| `CLIENT_WINS` | Client version always wins | Mobile-first |
| `NEWEST_WINS` | Most recent timestamp wins | General purpose |
| `MERGE` | Attempt to merge changes | Complex content |
| `ASK_USER` | Prompt user to resolve | Critical data |

### Delta Sync

Only sync changes since last sync:

```python
# Client tracks last sync timestamp
last_sync = client.last_sync_timestamp

# Request only changes since then
changes = await client.fetch_changes(since=last_sync)

# Apply changes locally
for change in changes:
    await client.apply_change(change)

# Update sync timestamp
client.last_sync_timestamp = datetime.now()
```

### Offline Queue

Changes are queued when offline:

```python
# Queue changes even when offline
client.queue_change(memory_id="mem-1", operation="create", data={...})
client.queue_change(memory_id="mem-2", operation="update", data={...})

# Queue persists to disk
print(f"Pending: {len(client.pending_changes)}")

# When online, sync automatically processes queue
await client.sync()  # Processes all pending changes
```

## Notification Types

| Type | Description | Payload |
|------|-------------|---------|
| `MEMORY_REMINDER` | Reminder to review a memory | memory_id, content preview |
| `SYNC_COMPLETE` | Sync finished | count of synced items |
| `LEARNING_STREAK` | Learning streak notification | streak_days |
| `NEW_INSIGHT` | New pattern discovered | insight summary |
| `MAINTENANCE` | System maintenance | message |

### APNs Payload Format

```json
{
  "aps": {
    "alert": {
      "title": "Memory Review",
      "body": "Time to review your memories"
    },
    "badge": 5,
    "sound": "default"
  },
  "memory_id": "mem-123",
  "action": "review"
}
```

### FCM Payload Format

```json
{
  "notification": {
    "title": "Sync Complete",
    "body": "50 memories synced"
  },
  "data": {
    "type": "sync_complete",
    "count": "50"
  }
}
```

## Quick Capture

Fast memory capture from mobile devices:

```python
# Voice memo capture
result = await api.quick_capture(
    device_id=device_id,
    content="transcribed voice memo content",
    source="voice_memo",
    metadata={
        "duration": 30.5,
        "audio_quality": "high",
    },
)

# Photo capture with OCR
result = await api.quick_capture(
    device_id=device_id,
    content="text extracted from photo",
    source="photo",
    metadata={
        "image_path": "/photos/note.jpg",
        "ocr_confidence": 0.95,
    },
)

# Quick note
result = await api.quick_capture(
    device_id=device_id,
    content="Quick thought to remember",
    source="quick_note",
)
```

## Device Management

### Device Info

```python
from memory.mobile import DeviceInfo

device = DeviceInfo(
    device_name="iPhone 15 Pro",
    device_type="ios",
    os_version="17.2",
    app_version="1.0.0",
    last_sync=datetime.now(),
    is_primary=True,
)

# Serialize for storage
data = device.to_dict()

# Get device summary
print(f"Device: {device.device_name} ({device.device_type})")
print(f"Last sync: {device.last_sync}")
```

### Multi-Device Support

```python
# List all devices
devices = api.list_devices(user_id="user-123")

for device in devices:
    print(f"{device.device_name}: Last seen {device.last_seen}")

# Remove device
api.unregister_device(device_id="old-device")
```

## Security

### Authentication Flow

```
1. Device registers with server
   → Server returns device_id + auth_token

2. Device stores auth_token securely (Keychain/KeyStore)

3. Device authenticates with auth_token
   → Server returns session token

4. All API calls include session token

5. Periodic token refresh maintains session
```

### Token Management

```python
# Initial registration
device = api.register_device(device_type="ios")

# Secure storage (platform-specific)
keychain.store("plm_auth_token", device.auth_token)

# Token refresh
new_token = api.refresh_token(device.device_id)
keychain.store("plm_auth_token", new_token)

# Revoke on logout
api.unregister_device(device.device_id)
keychain.delete("plm_auth_token")
```

## Configuration Reference

### SyncClient

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_url` | str | required | Server API URL |
| `device_id` | str | auto | Device identifier |
| `conflict_resolution` | enum | NEWEST_WINS | Conflict strategy |
| `timeout` | int | 30 | Request timeout (seconds) |

### SyncManager

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | SyncClient | required | Sync client instance |
| `auto_sync` | bool | True | Enable auto sync |
| `sync_interval` | int | 300 | Sync interval (seconds) |
| `sync_on_wifi_only` | bool | False | WiFi-only sync |
| `retry_count` | int | 3 | Retry attempts |

### NotificationService

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apns_key_path` | str | None | APNs key file path |
| `apns_key_id` | str | None | APNs key ID |
| `apns_team_id` | str | None | APNs team ID |
| `fcm_credentials` | str | None | FCM credentials file |

---

*See [ARCHITECTURE.md](./ARCHITECTURE.md) for overall system architecture.*
*See [API.md](./API.md) for complete API reference.*
