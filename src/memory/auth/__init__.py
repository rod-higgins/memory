"""Authentication module for Personal Memory System."""

from .mfa import MFAManager
from .middleware import AuthMiddleware, get_current_user, require_admin, require_auth
from .models import (
    AuthState,
    LoginRequest,
    MFASetupResponse,
    MFAVerifyRequest,
    Session,
    TokenResponse,
    User,
    UserCreate,
    UserRole,
    UserUpdate,
)
from .password import PasswordManager
from .store import AuthStore

__all__ = [
    "User",
    "UserCreate",
    "UserUpdate",
    "UserRole",
    "Session",
    "LoginRequest",
    "MFASetupResponse",
    "MFAVerifyRequest",
    "TokenResponse",
    "AuthState",
    "AuthStore",
    "PasswordManager",
    "MFAManager",
    "AuthMiddleware",
    "get_current_user",
    "require_auth",
    "require_admin",
]
