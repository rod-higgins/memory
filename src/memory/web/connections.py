"""Connection and credential management for data sources."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class ConnectionStatus(str, Enum):
    """Status of a data source connection."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PENDING = "pending"
    NEEDS_AUTH = "needs_auth"


class AuthType(str, Enum):
    """Authentication type for a connection."""

    NONE = "none"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    USERNAME_PASSWORD = "username_password"
    TOKEN = "token"
    FILE_PATH = "file_path"
    EXPORT_FILE = "export_file"


@dataclass
class ConnectionConfig:
    """Configuration for a data source connection."""

    id: str
    source_id: str  # References the source registry
    name: str
    auth_type: AuthType
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    credentials: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    last_sync: datetime | None = None
    last_error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    sync_enabled: bool = True
    sync_interval_hours: int = 24


class CredentialStore:
    """Secure storage for credentials using encryption."""

    def __init__(self, db_path: str | Path, master_key: str | None = None):
        """
        Initialize the credential store.

        Args:
            db_path: Path to the SQLite database
            master_key: Master encryption key. If None, generates one and stores in keychain/file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Get or create encryption key
        self._fernet = self._get_fernet(master_key)

        # Initialize database
        self._init_db()

    def _get_fernet(self, master_key: str | None) -> Fernet:
        """Get or create the Fernet encryption instance."""
        key_file = self.db_path.parent / ".credential_key"

        if master_key:
            # Derive key from provided master key
            key = self._derive_key(master_key)
        elif key_file.exists():
            # Load existing key
            key = key_file.read_bytes()
        else:
            # Generate new key
            key = Fernet.generate_key()
            # Save with restricted permissions
            key_file.write_bytes(key)
            os.chmod(key_file, 0o600)

        return Fernet(key)

    def _derive_key(self, password: str, salt: bytes | None = None) -> bytes:
        """Derive an encryption key from a password."""
        if salt is None:
            salt = b"plm_credential_store_salt_v1"  # Static salt for simplicity

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS connections (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    auth_type TEXT NOT NULL,
                    status TEXT DEFAULT 'disconnected',
                    credentials_encrypted BLOB,
                    settings TEXT,
                    last_sync TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    sync_enabled INTEGER DEFAULT 1,
                    sync_interval_hours INTEGER DEFAULT 24
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    connection_id TEXT PRIMARY KEY,
                    access_token_encrypted BLOB,
                    refresh_token_encrypted BLOB,
                    token_type TEXT,
                    expires_at TEXT,
                    scope TEXT,
                    FOREIGN KEY (connection_id) REFERENCES connections(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_connections_source
                ON connections(source_id)
            """)

    def encrypt(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        return self._fernet.encrypt(data.encode())

    def decrypt(self, data: bytes) -> str:
        """Decrypt sensitive data."""
        return self._fernet.decrypt(data).decode()

    def save_connection(self, config: ConnectionConfig) -> ConnectionConfig:
        """Save a connection configuration."""
        # Encrypt credentials
        credentials_encrypted = None
        if config.credentials:
            credentials_json = json.dumps(config.credentials)
            credentials_encrypted = self.encrypt(credentials_json)

        settings_json = json.dumps(config.settings) if config.settings else None
        config.updated_at = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO connections
                (id, source_id, name, auth_type, status, credentials_encrypted,
                 settings, last_sync, last_error, created_at, updated_at,
                 sync_enabled, sync_interval_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.id,
                    config.source_id,
                    config.name,
                    config.auth_type.value,
                    config.status.value,
                    credentials_encrypted,
                    settings_json,
                    config.last_sync.isoformat() if config.last_sync else None,
                    config.last_error,
                    config.created_at.isoformat(),
                    config.updated_at.isoformat(),
                    1 if config.sync_enabled else 0,
                    config.sync_interval_hours,
                ),
            )

        return config

    def get_connection(self, connection_id: str) -> ConnectionConfig | None:
        """Get a connection by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM connections WHERE id = ?", (connection_id,)).fetchone()

            if not row:
                return None

            return self._row_to_config(row)

    def list_connections(self, source_id: str | None = None) -> list[ConnectionConfig]:
        """List all connections, optionally filtered by source."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if source_id:
                rows = conn.execute(
                    "SELECT * FROM connections WHERE source_id = ? ORDER BY name",
                    (source_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM connections ORDER BY name").fetchall()

            return [self._row_to_config(row) for row in rows]

    def delete_connection(self, connection_id: str) -> bool:
        """Delete a connection."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete OAuth tokens first
            conn.execute(
                "DELETE FROM oauth_tokens WHERE connection_id = ?",
                (connection_id,),
            )
            # Delete connection
            cursor = conn.execute("DELETE FROM connections WHERE id = ?", (connection_id,))
            return cursor.rowcount > 0

    def update_status(
        self,
        connection_id: str,
        status: ConnectionStatus,
        error: str | None = None,
    ):
        """Update connection status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE connections
                SET status = ?, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, error, datetime.now().isoformat(), connection_id),
            )

    def update_last_sync(self, connection_id: str):
        """Update the last sync timestamp."""
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE connections
                SET last_sync = ?, updated_at = ?
                WHERE id = ?
                """,
                (now.isoformat(), now.isoformat(), connection_id),
            )

    def save_oauth_tokens(
        self,
        connection_id: str,
        access_token: str,
        refresh_token: str | None = None,
        token_type: str = "Bearer",
        expires_at: datetime | None = None,
        scope: str | None = None,
    ):
        """Save OAuth tokens for a connection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO oauth_tokens
                (connection_id, access_token_encrypted, refresh_token_encrypted,
                 token_type, expires_at, scope)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    connection_id,
                    self.encrypt(access_token),
                    self.encrypt(refresh_token) if refresh_token else None,
                    token_type,
                    expires_at.isoformat() if expires_at else None,
                    scope,
                ),
            )

    def get_oauth_tokens(self, connection_id: str) -> dict[str, Any] | None:
        """Get OAuth tokens for a connection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM oauth_tokens WHERE connection_id = ?",
                (connection_id,),
            ).fetchone()

            if not row:
                return None

            return {
                "access_token": self.decrypt(row["access_token_encrypted"]),
                "refresh_token": (
                    self.decrypt(row["refresh_token_encrypted"]) if row["refresh_token_encrypted"] else None
                ),
                "token_type": row["token_type"],
                "expires_at": (datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None),
                "scope": row["scope"],
            }

    def _row_to_config(self, row: sqlite3.Row) -> ConnectionConfig:
        """Convert a database row to ConnectionConfig."""
        credentials = {}
        if row["credentials_encrypted"]:
            try:
                credentials_json = self.decrypt(row["credentials_encrypted"])
                credentials = json.loads(credentials_json)
            except Exception:
                pass  # Credentials corrupted or key changed

        settings = {}
        if row["settings"]:
            settings = json.loads(row["settings"])

        return ConnectionConfig(
            id=row["id"],
            source_id=row["source_id"],
            name=row["name"],
            auth_type=AuthType(row["auth_type"]),
            status=ConnectionStatus(row["status"]),
            credentials=credentials,
            settings=settings,
            last_sync=(datetime.fromisoformat(row["last_sync"]) if row["last_sync"] else None),
            last_error=row["last_error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            sync_enabled=bool(row["sync_enabled"]),
            sync_interval_hours=row["sync_interval_hours"],
        )


class ConnectionManager:
    """Manages data source connections."""

    def __init__(self, data_dir: str | Path = "~/memory/data"):
        """Initialize the connection manager."""
        self.data_dir = Path(data_dir).expanduser()
        self.store = CredentialStore(self.data_dir / "connections.db")

        # Source definitions with auth requirements
        self._source_auth_config = self._build_auth_config()

    def _build_auth_config(self) -> dict[str, dict[str, Any]]:
        """Build authentication configuration for each source type."""
        return {
            # Social Media
            "twitter": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "Twitter Export Path", "type": "file"},
                ],
                "description": "Upload your Twitter data export (data-export.zip)",
            },
            "facebook": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "Facebook Export Path", "type": "file"},
                ],
                "description": "Upload your Facebook data download",
            },
            "linkedin": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "LinkedIn Export Path", "type": "file"},
                ],
                "description": "Upload your LinkedIn data export",
            },
            "instagram": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "Instagram Export Path", "type": "file"},
                ],
                "description": "Upload your Instagram data download",
            },
            "reddit": {
                "auth_type": AuthType.API_KEY,
                "fields": [
                    {"name": "client_id", "label": "Client ID", "type": "text"},
                    {"name": "client_secret", "label": "Client Secret", "type": "password"},
                    {"name": "username", "label": "Username", "type": "text"},
                ],
                "description": "Create an app at reddit.com/prefs/apps",
            },
            # Communication
            "gmail": {
                "auth_type": AuthType.OAUTH2,
                "oauth_config": {
                    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                    "token_url": "https://oauth2.googleapis.com/token",
                    "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
                },
                "fields": [
                    {"name": "client_id", "label": "OAuth Client ID", "type": "text"},
                    {"name": "client_secret", "label": "OAuth Client Secret", "type": "password"},
                ],
                "description": "Set up OAuth in Google Cloud Console",
            },
            "slack": {
                "auth_type": AuthType.TOKEN,
                "fields": [
                    {"name": "token", "label": "Bot Token", "type": "password"},
                    {"name": "workspace", "label": "Workspace Name", "type": "text"},
                ],
                "description": "Create a Slack app and get a Bot Token",
            },
            "discord": {
                "auth_type": AuthType.TOKEN,
                "fields": [
                    {"name": "token", "label": "User Token", "type": "password"},
                ],
                "description": "Your Discord user token (use carefully)",
            },
            "imessage": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "db_path",
                        "label": "Messages DB Path",
                        "type": "file",
                        "default": "~/Library/Messages/chat.db",
                    },
                ],
                "description": "Requires Full Disk Access permission",
            },
            "whatsapp": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "WhatsApp Export Path", "type": "file"},
                ],
                "description": "Export chat from WhatsApp",
            },
            # Documents
            "notion": {
                "auth_type": AuthType.API_KEY,
                "fields": [
                    {"name": "api_key", "label": "Integration Token", "type": "password"},
                ],
                "description": "Create an integration at notion.so/my-integrations",
            },
            "obsidian": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {"name": "vault_path", "label": "Vault Path", "type": "directory"},
                ],
                "description": "Path to your Obsidian vault",
            },
            "google_docs": {
                "auth_type": AuthType.OAUTH2,
                "oauth_config": {
                    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                    "token_url": "https://oauth2.googleapis.com/token",
                    "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
                },
                "fields": [
                    {"name": "client_id", "label": "OAuth Client ID", "type": "text"},
                    {"name": "client_secret", "label": "OAuth Client Secret", "type": "password"},
                ],
                "description": "Set up OAuth in Google Cloud Console",
            },
            "apple_notes": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "db_path",
                        "label": "Notes DB Path",
                        "type": "file",
                        "default": "~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite",
                    },
                ],
                "description": "Requires Full Disk Access permission",
            },
            "local_documents": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {"name": "path", "label": "Documents Path", "type": "directory"},
                    {
                        "name": "extensions",
                        "label": "File Extensions",
                        "type": "text",
                        "default": ".txt,.md,.docx,.pdf",
                    },
                ],
                "description": "Path to your documents folder",
            },
            # Code
            "github": {
                "auth_type": AuthType.TOKEN,
                "fields": [
                    {"name": "token", "label": "Personal Access Token", "type": "password"},
                    {"name": "username", "label": "Username", "type": "text"},
                ],
                "description": "Create a token at github.com/settings/tokens",
            },
            "gitlab": {
                "auth_type": AuthType.TOKEN,
                "fields": [
                    {"name": "token", "label": "Personal Access Token", "type": "password"},
                    {"name": "url", "label": "GitLab URL", "type": "text", "default": "https://gitlab.com"},
                ],
                "description": "Create a token in GitLab settings",
            },
            "local_git": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {"name": "path", "label": "Git Repos Path", "type": "directory"},
                ],
                "description": "Path containing git repositories",
            },
            # AI History
            "claude_history": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "history_path",
                        "label": "History File",
                        "type": "file",
                        "default": "~/.claude/history.jsonl",
                    },
                ],
                "description": "Claude Code conversation history",
            },
            "chatgpt_history": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "Export Path", "type": "file"},
                ],
                "description": "Export from ChatGPT settings",
            },
            # Media
            "apple_photos": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "library_path",
                        "label": "Photos Library",
                        "type": "file",
                        "default": "~/Pictures/Photos Library.photoslibrary",
                    },
                ],
                "description": "Requires Photos access permission",
            },
            "google_photos": {
                "auth_type": AuthType.OAUTH2,
                "oauth_config": {
                    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                    "token_url": "https://oauth2.googleapis.com/token",
                    "scopes": ["https://www.googleapis.com/auth/photoslibrary.readonly"],
                },
                "fields": [
                    {"name": "client_id", "label": "OAuth Client ID", "type": "text"},
                    {"name": "client_secret", "label": "OAuth Client Secret", "type": "password"},
                ],
                "description": "Set up OAuth in Google Cloud Console",
            },
            "youtube": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "Takeout Path", "type": "file"},
                ],
                "description": "Download from Google Takeout",
            },
            "spotify": {
                "auth_type": AuthType.EXPORT_FILE,
                "fields": [
                    {"name": "export_path", "label": "Export Path", "type": "file"},
                ],
                "description": "Request data from Spotify privacy settings",
            },
            # Browsing
            "chrome_history": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "profile_path",
                        "label": "Chrome Profile",
                        "type": "directory",
                        "default": "~/Library/Application Support/Google/Chrome/Default",
                    },
                ],
                "description": "Chrome browser history",
            },
            "safari_history": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "db_path",
                        "label": "History DB",
                        "type": "file",
                        "default": "~/Library/Safari/History.db",
                    },
                ],
                "description": "Safari browser history",
            },
            "bookmarks": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "browser",
                        "label": "Browser",
                        "type": "select",
                        "options": ["chrome", "safari", "firefox"],
                    },
                ],
                "description": "Browser bookmarks",
            },
            # Calendar & Tasks
            "apple_calendar": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {
                        "name": "calendar_path",
                        "label": "Calendar Path",
                        "type": "directory",
                        "default": "~/Library/Calendars",
                    },
                ],
                "description": "Apple Calendar events",
            },
            "google_calendar": {
                "auth_type": AuthType.OAUTH2,
                "oauth_config": {
                    "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
                    "token_url": "https://oauth2.googleapis.com/token",
                    "scopes": ["https://www.googleapis.com/auth/calendar.readonly"],
                },
                "fields": [
                    {"name": "client_id", "label": "OAuth Client ID", "type": "text"},
                    {"name": "client_secret", "label": "OAuth Client Secret", "type": "password"},
                ],
                "description": "Set up OAuth in Google Cloud Console",
            },
            "apple_reminders": {
                "auth_type": AuthType.FILE_PATH,
                "fields": [
                    {"name": "db_path", "label": "Reminders DB", "type": "file"},
                ],
                "description": "Apple Reminders",
            },
        }

    def get_source_auth_config(self, source_id: str) -> dict[str, Any] | None:
        """Get authentication configuration for a source."""
        return self._source_auth_config.get(source_id)

    def create_connection(
        self,
        source_id: str,
        name: str,
        credentials: dict[str, str],
        settings: dict[str, Any] | None = None,
    ) -> ConnectionConfig:
        """Create a new connection."""
        import uuid

        auth_config = self.get_source_auth_config(source_id)
        if not auth_config:
            raise ValueError(f"Unknown source: {source_id}")

        config = ConnectionConfig(
            id=str(uuid.uuid4()),
            source_id=source_id,
            name=name,
            auth_type=auth_config["auth_type"],
            credentials=credentials,
            settings=settings or {},
            status=ConnectionStatus.PENDING,
        )

        return self.store.save_connection(config)

    def test_connection(self, connection_id: str) -> tuple[bool, str]:
        """Test a connection and update its status."""
        config = self.store.get_connection(connection_id)
        if not config:
            return False, "Connection not found"

        try:
            # Test based on auth type
            if config.auth_type == AuthType.FILE_PATH:
                path = (
                    config.credentials.get("path")
                    or config.credentials.get("db_path")
                    or config.credentials.get("vault_path")
                )
                if path:
                    expanded = Path(path).expanduser()
                    if not expanded.exists():
                        raise FileNotFoundError(f"Path not found: {path}")

            elif config.auth_type == AuthType.EXPORT_FILE:
                path = config.credentials.get("export_path")
                if path:
                    expanded = Path(path).expanduser()
                    if not expanded.exists():
                        raise FileNotFoundError(f"Export file not found: {path}")

            elif config.auth_type == AuthType.TOKEN:
                token = config.credentials.get("token")
                if not token:
                    raise ValueError("Token not provided")

            elif config.auth_type == AuthType.API_KEY:
                api_key = config.credentials.get("api_key") or config.credentials.get("client_id")
                if not api_key:
                    raise ValueError("API key not provided")

            # Connection successful
            self.store.update_status(connection_id, ConnectionStatus.CONNECTED)
            return True, "Connection successful"

        except Exception as e:
            error_msg = str(e)
            self.store.update_status(connection_id, ConnectionStatus.ERROR, error_msg)
            return False, error_msg

    def get_connection(self, connection_id: str) -> ConnectionConfig | None:
        """Get a connection by ID."""
        return self.store.get_connection(connection_id)

    def list_connections(self, source_id: str | None = None) -> list[ConnectionConfig]:
        """List all connections."""
        return self.store.list_connections(source_id)

    def update_connection(
        self,
        connection_id: str,
        name: str | None = None,
        credentials: dict[str, str] | None = None,
        settings: dict[str, Any] | None = None,
        sync_enabled: bool | None = None,
        sync_interval_hours: int | None = None,
    ) -> ConnectionConfig | None:
        """Update a connection."""
        config = self.store.get_connection(connection_id)
        if not config:
            return None

        if name is not None:
            config.name = name
        if credentials is not None:
            config.credentials = credentials
        if settings is not None:
            config.settings = settings
        if sync_enabled is not None:
            config.sync_enabled = sync_enabled
        if sync_interval_hours is not None:
            config.sync_interval_hours = sync_interval_hours

        return self.store.save_connection(config)

    def delete_connection(self, connection_id: str) -> bool:
        """Delete a connection."""
        return self.store.delete_connection(connection_id)

    def get_all_source_configs(self) -> list[dict[str, Any]]:
        """Get all source configurations with auth info."""
        from ..ingestion.sources.registry import SourceRegistry

        registry = SourceRegistry()
        sources = registry.list_sources()

        result = []
        for source in sources:
            source_id = source["id"]
            auth_config = self._source_auth_config.get(
                source_id,
                {
                    "auth_type": AuthType.NONE,
                    "fields": [],
                    "description": source.get("description", ""),
                },
            )

            result.append(
                {
                    **source,
                    "auth_type": auth_config.get("auth_type", AuthType.NONE).value
                    if isinstance(auth_config.get("auth_type"), AuthType)
                    else auth_config.get("auth_type", "none"),
                    "auth_fields": auth_config.get("fields", []),
                    "auth_description": auth_config.get("description", ""),
                    "oauth_config": auth_config.get("oauth_config"),
                }
            )

        return result
