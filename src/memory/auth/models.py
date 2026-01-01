"""Pydantic models for authentication."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """User role enumeration."""

    ADMIN = "admin"
    USER = "user"


class AuthState(str, Enum):
    """Authentication state for frontend."""

    LOGIN = "login"
    MFA_SETUP = "mfa_setup"
    MFA_VERIFY = "mfa_verify"
    AUTHENTICATED = "authenticated"


class User(BaseModel):
    """User model returned from API."""

    id: str
    username: str
    email: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    mfa_enabled: bool = False
    data_path: str
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """Request model for creating a user."""

    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.USER


class UserUpdate(BaseModel):
    """Request model for updating a user."""

    email: EmailStr | None = None
    is_active: bool | None = None
    role: UserRole | None = None


class Session(BaseModel):
    """Session model."""

    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str | None = None
    user_agent: str | None = None
    is_valid: bool = True

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Request model for login."""

    username_or_email: str
    password: str


class MFASetupResponse(BaseModel):
    """Response model for MFA setup."""

    secret: str  # Base32 encoded secret
    qr_code_uri: str  # otpauth:// URI for QR code
    qr_code_base64: str  # Base64 encoded PNG QR code
    backup_codes: list[str]  # One-time use backup codes


class MFAVerifyRequest(BaseModel):
    """Request model for MFA verification."""

    totp_code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


class BackupCodeRequest(BaseModel):
    """Request model for backup code verification."""

    backup_code: str = Field(..., min_length=8, max_length=12)


class TokenResponse(BaseModel):
    """Response model for authentication token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User
    auth_state: AuthState


class PasswordResetRequest(BaseModel):
    """Request model for password reset (admin)."""

    new_password: str = Field(..., min_length=8)
