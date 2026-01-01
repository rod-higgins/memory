"""Authentication store using SQLite."""

import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .mfa import MFAManager, init_mfa_manager
from .models import Session, User, UserCreate, UserRole
from .password import get_password_manager


class AuthStore:
    """SQLite-based storage for authentication data."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        is_active INTEGER DEFAULT 1,
        mfa_enabled INTEGER DEFAULT 0,
        mfa_secret_encrypted BLOB,
        mfa_backup_codes_json TEXT,
        data_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_login TEXT
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        ip_address TEXT,
        user_agent TEXT,
        is_valid INTEGER DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS mfa_pending (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        temp_secret TEXT NOT NULL,
        backup_codes_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS login_attempts (
        id TEXT PRIMARY KEY,
        username_or_email TEXT NOT NULL,
        ip_address TEXT,
        attempt_time TEXT NOT NULL,
        successful INTEGER DEFAULT 0,
        failure_reason TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
    CREATE INDEX IF NOT EXISTS idx_login_attempts_time ON login_attempts(attempt_time);
    """

    def __init__(self, db_path: Path):
        """Initialize the auth store."""
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._mfa_manager: MFAManager | None = None
        self._credential_key_path = db_path.parent / ".credential_key"

    async def initialize(self) -> None:
        """Initialize the database and encryption."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or load encryption key
        await self._init_encryption_key()

        # Connect to database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

        # Create schema
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    async def _init_encryption_key(self) -> None:
        """Initialize or load the encryption key for MFA secrets."""
        if self._credential_key_path.exists():
            # Load existing key
            with open(self._credential_key_path, "rb") as f:
                key_data = f.read()
        else:
            # Generate new key
            from cryptography.fernet import Fernet

            key_data = Fernet.generate_key()
            # Save with restrictive permissions
            with open(self._credential_key_path, "wb") as f:
                f.write(key_data)
            os.chmod(self._credential_key_path, 0o600)

        self._mfa_manager = init_mfa_manager(key_data)

    @property
    def mfa_manager(self) -> MFAManager:
        """Get the MFA manager."""
        if self._mfa_manager is None:
            raise RuntimeError("AuthStore not initialized")
        return self._mfa_manager

    async def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _user_from_row(self, row: sqlite3.Row) -> User:
        """Convert a database row to a User model."""
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            role=UserRole(row["role"]),
            is_active=bool(row["is_active"]),
            mfa_enabled=bool(row["mfa_enabled"]),
            data_path=row["data_path"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
        )

    def _session_from_row(self, row: sqlite3.Row) -> Session:
        """Convert a database row to a Session model."""
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            ip_address=row["ip_address"],
            user_agent=row["user_agent"],
            is_valid=bool(row["is_valid"]),
        )

    # User operations

    async def create_user(self, data: UserCreate) -> User:
        """Create a new user."""
        now = datetime.utcnow().isoformat()
        user_id = str(uuid4())
        username = data.username.lower()

        # Generate user data path
        data_path = str(Path("~/memory/data/users").expanduser() / username)

        # Hash password
        password_hash = get_password_manager().hash(data.password)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (id, username, email, password_hash, role, data_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, data.email.lower(), password_hash, data.role.value, data_path, now, now),
        )
        self.conn.commit()

        # Create user data directory
        user_data_path = Path(data_path)
        user_data_path.mkdir(parents=True, exist_ok=True)
        (user_data_path / "persistent").mkdir(exist_ok=True)
        (user_data_path / "long_term").mkdir(exist_ok=True)
        (user_data_path / "short_term").mkdir(exist_ok=True)

        return await self.get_user(user_id)

    async def get_user(self, user_id: str) -> User | None:
        """Get a user by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return self._user_from_row(row) if row else None

    async def get_user_by_username(self, username: str) -> User | None:
        """Get a user by username."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.lower(),))
        row = cursor.fetchone()
        return self._user_from_row(row) if row else None

    async def get_user_by_email(self, email: str) -> User | None:
        """Get a user by email."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
        row = cursor.fetchone()
        return self._user_from_row(row) if row else None

    async def get_user_by_username_or_email(self, identifier: str) -> User | None:
        """Get a user by username or email."""
        identifier = identifier.lower()
        user = await self.get_user_by_username(identifier)
        if not user:
            user = await self.get_user_by_email(identifier)
        return user

    async def list_users(self) -> list[User]:
        """List all users."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        return [self._user_from_row(row) for row in cursor.fetchall()]

    async def update_user(self, user_id: str, **kwargs) -> User | None:
        """Update user fields."""
        allowed_fields = {"email", "is_active", "role", "mfa_enabled", "last_login"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return await self.get_user(user_id)

        updates["updated_at"] = datetime.utcnow().isoformat()

        # Convert booleans to integers for SQLite
        for k, v in updates.items():
            if isinstance(v, bool):
                updates[k] = int(v)
            elif isinstance(v, UserRole):
                updates[k] = v.value

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [user_id]

        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
        self.conn.commit()

        return await self.get_user(user_id)

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user (soft delete - set inactive)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE users SET is_active = 0, updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), user_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    async def verify_password(self, user_id: str, password: str) -> bool:
        """Verify a user's password."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            return False
        return get_password_manager().verify(password, row["password_hash"])

    async def update_password(self, user_id: str, new_password: str) -> bool:
        """Update a user's password."""
        password_hash = get_password_manager().hash(new_password)
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (password_hash, datetime.utcnow().isoformat(), user_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # MFA operations

    async def setup_mfa_pending(self, user_id: str) -> tuple[str, str, list[str]]:
        """Start MFA setup for a user. Returns (secret, qr_uri, backup_codes)."""
        user = await self.get_user(user_id)
        if not user:
            raise ValueError("User not found")

        # Generate secret and backup codes
        secret = self.mfa_manager.generate_secret()
        backup_codes = self.mfa_manager.generate_backup_codes()
        qr_uri = self.mfa_manager.get_provisioning_uri(secret, user.username)

        # Store pending setup
        now = datetime.utcnow()
        expires = now + timedelta(minutes=15)
        pending_id = str(uuid4())

        cursor = self.conn.cursor()
        # Clear any existing pending setup
        cursor.execute("DELETE FROM mfa_pending WHERE user_id = ?", (user_id,))
        # Insert new pending setup
        cursor.execute(
            """
            INSERT INTO mfa_pending (id, user_id, temp_secret, backup_codes_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pending_id, user_id, secret, json.dumps(backup_codes), now.isoformat(), expires.isoformat()),
        )
        self.conn.commit()

        return secret, qr_uri, backup_codes

    async def complete_mfa_setup(self, user_id: str, totp_code: str) -> bool:
        """Complete MFA setup by verifying the TOTP code."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM mfa_pending WHERE user_id = ? AND expires_at > ?",
            (user_id, datetime.utcnow().isoformat()),
        )
        row = cursor.fetchone()

        if not row:
            return False

        # Verify the TOTP code
        if not self.mfa_manager.verify_totp(row["temp_secret"], totp_code):
            return False

        # Encrypt and store the secret
        encrypted_secret = self.mfa_manager.encrypt_secret(row["temp_secret"])

        # Hash backup codes
        backup_codes = json.loads(row["backup_codes_json"])
        hashed_codes = [self.mfa_manager.hash_backup_code(code) for code in backup_codes]

        # Update user
        cursor.execute(
            """
            UPDATE users SET
                mfa_enabled = 1,
                mfa_secret_encrypted = ?,
                mfa_backup_codes_json = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (encrypted_secret, json.dumps(hashed_codes), datetime.utcnow().isoformat(), user_id),
        )

        # Clean up pending
        cursor.execute("DELETE FROM mfa_pending WHERE user_id = ?", (user_id,))
        self.conn.commit()

        return True

    async def verify_mfa(self, user_id: str, totp_code: str) -> bool:
        """Verify a TOTP code for a user."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT mfa_secret_encrypted FROM users WHERE id = ? AND mfa_enabled = 1",
            (user_id,),
        )
        row = cursor.fetchone()

        if not row or not row["mfa_secret_encrypted"]:
            return False

        secret = self.mfa_manager.decrypt_secret(row["mfa_secret_encrypted"])
        return self.mfa_manager.verify_totp(secret, totp_code)

    async def verify_backup_code(self, user_id: str, backup_code: str) -> bool:
        """Verify and consume a backup code."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT mfa_backup_codes_json FROM users WHERE id = ? AND mfa_enabled = 1",
            (user_id,),
        )
        row = cursor.fetchone()

        if not row or not row["mfa_backup_codes_json"]:
            return False

        hashed_codes = json.loads(row["mfa_backup_codes_json"])

        # Find and remove matching code
        for i, hashed in enumerate(hashed_codes):
            if self.mfa_manager.verify_backup_code(backup_code, hashed):
                # Remove used code
                hashed_codes.pop(i)
                cursor.execute(
                    "UPDATE users SET mfa_backup_codes_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(hashed_codes), datetime.utcnow().isoformat(), user_id),
                )
                self.conn.commit()
                return True

        return False

    async def reset_mfa(self, user_id: str) -> bool:
        """Reset MFA for a user (admin action)."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE users SET
                mfa_enabled = 0,
                mfa_secret_encrypted = NULL,
                mfa_backup_codes_json = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (datetime.utcnow().isoformat(), user_id),
        )
        cursor.execute("DELETE FROM mfa_pending WHERE user_id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    async def get_pending_mfa_backup_codes(self, user_id: str) -> list[str] | None:
        """Get backup codes from pending MFA setup."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT backup_codes_json FROM mfa_pending WHERE user_id = ? AND expires_at > ?",
            (user_id, datetime.utcnow().isoformat()),
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row["backup_codes_json"])
        return None

    # Session operations

    async def create_session(
        self, user_id: str, ip_address: str | None = None, user_agent: str | None = None, expires_hours: int = 24
    ) -> Session:
        """Create a new session for a user."""
        now = datetime.utcnow()
        expires = now + timedelta(hours=expires_hours)
        session_id = secrets.token_urlsafe(32)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (id, user_id, created_at, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, now.isoformat(), expires.isoformat(), ip_address, user_agent),
        )

        # Update last_login
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now.isoformat(), user_id),
        )
        self.conn.commit()

        return await self.get_session(session_id)

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return self._session_from_row(row) if row else None

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE sessions SET is_valid = 0 WHERE id = ?", (session_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    async def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE sessions SET is_valid = 0 WHERE user_id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (datetime.utcnow().isoformat(),),
        )
        self.conn.commit()
        return cursor.rowcount

    # Login attempt tracking

    async def record_login_attempt(
        self,
        username_or_email: str,
        ip_address: str | None = None,
        successful: bool = False,
        failure_reason: str | None = None,
    ) -> None:
        """Record a login attempt."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO login_attempts (id, username_or_email, ip_address, attempt_time, successful, failure_reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                username_or_email.lower(),
                ip_address,
                datetime.utcnow().isoformat(),
                int(successful),
                failure_reason,
            ),
        )
        self.conn.commit()

    async def get_recent_failed_attempts(self, username_or_email: str, minutes: int = 15) -> int:
        """Get count of recent failed login attempts."""
        cutoff = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM login_attempts
            WHERE username_or_email = ? AND attempt_time > ? AND successful = 0
            """,
            (username_or_email.lower(), cutoff),
        )
        return cursor.fetchone()[0]

    async def has_any_users(self) -> bool:
        """Check if any users exist."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        return cursor.fetchone()[0] > 0

    async def has_admin(self) -> bool:
        """Check if an admin user exists."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        return cursor.fetchone()[0] > 0
