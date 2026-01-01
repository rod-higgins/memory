"""Tests for authentication module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from memory.auth.models import (
    User,
    UserCreate,
    UserUpdate,
    UserRole,
    Session,
    LoginRequest,
    MFASetupResponse,
    MFAVerifyRequest,
    TokenResponse,
    AuthState,
)
from memory.auth.password import PasswordManager, get_password_manager
from memory.auth.mfa import MFAManager, get_mfa_manager
from memory.auth.store import AuthStore


class TestPasswordManager:
    """Tests for PasswordManager."""

    def test_initialization(self):
        """Test password manager initialization."""
        pm = PasswordManager()
        assert pm.rounds == 12

    def test_custom_rounds(self):
        """Test password manager with custom rounds."""
        pm = PasswordManager(rounds=10)
        assert pm.rounds == 10

    def test_hash_password(self):
        """Test password hashing."""
        pm = PasswordManager(rounds=4)  # Use fewer rounds for speed
        password = "test_password_123"
        hashed = pm.hash(password)

        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt prefix
        assert len(hashed) == 60  # bcrypt hash length

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        pm = PasswordManager(rounds=4)
        password = "test_password_123"
        hashed = pm.hash(password)

        assert pm.verify(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        pm = PasswordManager(rounds=4)
        password = "test_password_123"
        hashed = pm.hash(password)

        assert pm.verify("wrong_password", hashed) is False

    def test_verify_invalid_hash(self):
        """Test verifying against invalid hash."""
        pm = PasswordManager(rounds=4)

        assert pm.verify("password", "invalid_hash") is False

    def test_needs_rehash(self):
        """Test needs_rehash (always returns False for now)."""
        pm = PasswordManager()
        hashed = pm.hash("password")

        assert pm.needs_rehash(hashed) is False

    def test_singleton_password_manager(self):
        """Test singleton password manager."""
        pm1 = get_password_manager()
        pm2 = get_password_manager()

        assert pm1 is pm2


class TestMFAManager:
    """Tests for MFAManager."""

    @pytest.fixture
    def encryption_key(self):
        """Generate a test encryption key."""
        salt = MFAManager.generate_salt()
        return MFAManager.generate_encryption_key("test_password", salt)

    def test_initialization(self, encryption_key):
        """Test MFA manager initialization."""
        mfa = MFAManager(encryption_key)
        assert mfa.issuer == "PersonalMemorySystem"

    def test_generate_secret(self, encryption_key):
        """Test secret generation."""
        mfa = MFAManager(encryption_key)
        secret = mfa.generate_secret()

        assert isinstance(secret, str)
        assert len(secret) == 32  # Base32 encoded

    def test_get_provisioning_uri(self, encryption_key):
        """Test provisioning URI generation."""
        mfa = MFAManager(encryption_key)
        secret = mfa.generate_secret()
        uri = mfa.get_provisioning_uri(secret, "testuser")

        assert uri.startswith("otpauth://totp/")
        assert "PersonalMemorySystem" in uri
        assert "testuser" in uri
        assert secret in uri

    def test_verify_totp_valid(self, encryption_key):
        """Test TOTP verification with valid code."""
        mfa = MFAManager(encryption_key)
        secret = mfa.generate_secret()

        # Generate a valid code
        import pyotp

        totp = pyotp.TOTP(secret)
        code = totp.now()

        assert mfa.verify_totp(secret, code) is True

    def test_verify_totp_invalid(self, encryption_key):
        """Test TOTP verification with invalid code."""
        mfa = MFAManager(encryption_key)
        secret = mfa.generate_secret()

        assert mfa.verify_totp(secret, "000000") is False
        assert mfa.verify_totp(secret, "invalid") is False

    def test_generate_backup_codes(self, encryption_key):
        """Test backup code generation."""
        mfa = MFAManager(encryption_key)
        codes = mfa.generate_backup_codes(count=8)

        assert len(codes) == 8
        for code in codes:
            # Format is XXXX-XXXX (9 chars with hyphen)
            assert len(code) == 9
            assert code[4] == "-"
            assert code.replace("-", "").isalnum()

    def test_generate_qr_code_base64(self, encryption_key):
        """Test QR code base64 PNG generation."""
        mfa = MFAManager(encryption_key)
        secret = mfa.generate_secret()
        uri = mfa.get_provisioning_uri(secret, "testuser")
        qr_base64 = mfa.generate_qr_code_base64(uri)

        # Should be valid base64-encoded PNG
        assert isinstance(qr_base64, str)
        assert len(qr_base64) > 100  # Should have substantial content
        # Verify it's valid base64
        import base64
        decoded = base64.b64decode(qr_base64)
        assert decoded[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic bytes

    def test_encrypt_decrypt_secret(self, encryption_key):
        """Test secret encryption and decryption."""
        mfa = MFAManager(encryption_key)
        secret = mfa.generate_secret()

        encrypted = mfa.encrypt_secret(secret)
        decrypted = mfa.decrypt_secret(encrypted)

        assert decrypted == secret
        assert encrypted != secret.encode()


class TestUserModels:
    """Tests for user-related models."""

    def test_user_create(self):
        """Test UserCreate model."""
        user = UserCreate(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.password == "password123"
        assert user.role == UserRole.USER  # Default

    def test_user_create_admin(self):
        """Test UserCreate with admin role."""
        user = UserCreate(
            username="admin",
            email="admin@example.com",
            password="adminpass",
            role=UserRole.ADMIN,
        )

        assert user.role == UserRole.ADMIN

    def test_user_update(self):
        """Test UserUpdate model."""
        update = UserUpdate(email="newemail@example.com", is_active=False)

        assert update.email == "newemail@example.com"
        assert update.is_active is False
        assert update.role is None

    def test_login_request(self):
        """Test LoginRequest model."""
        request = LoginRequest(
            username_or_email="testuser",
            password="password123",
        )

        assert request.username_or_email == "testuser"
        assert request.password == "password123"

    def test_auth_state_enum(self):
        """Test AuthState enum values."""
        assert AuthState.LOGIN.value == "login"
        assert AuthState.MFA_SETUP.value == "mfa_setup"
        assert AuthState.MFA_VERIFY.value == "mfa_verify"
        assert AuthState.AUTHENTICATED.value == "authenticated"


class TestAuthStore:
    """Tests for AuthStore."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_auth.sqlite"
            yield db_path

    @pytest.fixture
    async def auth_store(self, temp_db):
        """Create and initialize an auth store."""
        store = AuthStore(db_path=temp_db)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_store_initialization(self, temp_db):
        """Test auth store initialization."""
        store = AuthStore(db_path=temp_db)
        await store.initialize()

        assert temp_db.exists()
        await store.close()

    @pytest.mark.asyncio
    async def test_create_user(self, auth_store):
        """Test creating a user."""
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        user = await auth_store.create_user(user_data)

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.is_active is True
        assert user.mfa_enabled is False

    @pytest.mark.asyncio
    async def test_get_user_by_username(self, auth_store):
        """Test getting user by username."""
        user_data = UserCreate(
            username="findme",
            email="findme@example.com",
            password="password123",
        )
        created = await auth_store.create_user(user_data)

        found = await auth_store.get_user_by_username("findme")

        assert found is not None
        assert found.id == created.id
        assert found.username == "findme"

    @pytest.mark.asyncio
    async def test_get_user_by_email(self, auth_store):
        """Test getting user by email."""
        user_data = UserCreate(
            username="emailuser",
            email="email@example.com",
            password="password123",
        )
        created = await auth_store.create_user(user_data)

        found = await auth_store.get_user_by_email("email@example.com")

        assert found is not None
        assert found.id == created.id

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, auth_store):
        """Test getting non-existent user."""
        found = await auth_store.get_user_by_username("nonexistent")
        assert found is None

    @pytest.mark.asyncio
    async def test_verify_password(self, auth_store):
        """Test password verification."""
        user_data = UserCreate(
            username="passuser",
            email="pass@example.com",
            password="correctpassword",
        )
        user = await auth_store.create_user(user_data)

        assert await auth_store.verify_password(user.id, "correctpassword") is True
        assert await auth_store.verify_password(user.id, "wrongpassword") is False

    @pytest.mark.asyncio
    async def test_update_password(self, auth_store):
        """Test password update."""
        user_data = UserCreate(
            username="updatepass",
            email="updatepass@example.com",
            password="oldpassword",
        )
        user = await auth_store.create_user(user_data)

        await auth_store.update_password(user.id, "newpassword")

        assert await auth_store.verify_password(user.id, "newpassword") is True
        assert await auth_store.verify_password(user.id, "oldpassword") is False

    @pytest.mark.asyncio
    async def test_create_session(self, auth_store):
        """Test session creation."""
        user_data = UserCreate(
            username="sessionuser",
            email="session@example.com",
            password="password123",
        )
        user = await auth_store.create_user(user_data)

        session = await auth_store.create_session(user.id)

        assert session.user_id == user.id
        assert session.is_valid is True
        assert session.expires_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_get_session(self, auth_store):
        """Test getting session."""
        user_data = UserCreate(
            username="getsession",
            email="getsession@example.com",
            password="password123",
        )
        user = await auth_store.create_user(user_data)
        created = await auth_store.create_session(user.id)

        session = await auth_store.get_session(created.id)

        assert session is not None
        assert session.id == created.id

    @pytest.mark.asyncio
    async def test_invalidate_session(self, auth_store):
        """Test session invalidation."""
        user_data = UserCreate(
            username="invalidsession",
            email="invalid@example.com",
            password="password123",
        )
        user = await auth_store.create_user(user_data)
        session = await auth_store.create_session(user.id)

        await auth_store.invalidate_session(session.id)

        updated = await auth_store.get_session(session.id)
        assert updated.is_valid is False

    @pytest.mark.asyncio
    async def test_has_admin(self, auth_store):
        """Test checking for admin user."""
        # Initially no admin
        assert await auth_store.has_admin() is False

        # Create admin
        admin_data = UserCreate(
            username="admin",
            email="admin@example.com",
            password="adminpass",
            role=UserRole.ADMIN,
        )
        await auth_store.create_user(admin_data)

        assert await auth_store.has_admin() is True

    @pytest.mark.asyncio
    async def test_list_users(self, auth_store):
        """Test listing users."""
        # Create some users
        for i in range(3):
            await auth_store.create_user(
                UserCreate(
                    username=f"user{i}",
                    email=f"user{i}@example.com",
                    password="password",
                )
            )

        users = await auth_store.list_users()

        assert len(users) == 3

    @pytest.mark.asyncio
    async def test_update_user(self, auth_store):
        """Test updating user."""
        user_data = UserCreate(
            username="updateme",
            email="old@example.com",
            password="password123",
        )
        user = await auth_store.create_user(user_data)

        # Update using kwargs
        updated = await auth_store.update_user(user.id, email="new@example.com")

        assert updated.email == "new@example.com"

    @pytest.mark.asyncio
    async def test_update_user_role(self, auth_store):
        """Test updating user role."""
        user_data = UserCreate(
            username="roleuser",
            email="role@example.com",
            password="password123",
        )
        user = await auth_store.create_user(user_data)

        updated = await auth_store.update_user(user.id, role=UserRole.ADMIN)

        assert updated.role == UserRole.ADMIN

    @pytest.mark.asyncio
    async def test_update_user_active_status(self, auth_store):
        """Test updating user active status."""
        user_data = UserCreate(
            username="activeuser",
            email="active@example.com",
            password="password123",
        )
        user = await auth_store.create_user(user_data)

        updated = await auth_store.update_user(user.id, is_active=False)

        assert updated.is_active is False

    @pytest.mark.asyncio
    async def test_reset_mfa(self, auth_store):
        """Test resetting MFA."""
        user_data = UserCreate(
            username="resetmfa",
            email="resetmfa@example.com",
            password="password123",
        )
        user = await auth_store.create_user(user_data)

        # Enable MFA via update
        await auth_store.update_user(user.id, mfa_enabled=True)

        # Reset it
        await auth_store.reset_mfa(user.id)

        updated = await auth_store.get_user(user.id)
        assert updated.mfa_enabled is False
