"""FastAPI authentication middleware and dependencies."""

import os
from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer

from .models import AuthState, User
from .store import AuthStore

# Configuration
JWT_SECRET_KEY = os.environ.get("PLM_JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
SESSION_COOKIE_NAME = "plm_session"

security = HTTPBearer(auto_error=False)


def create_access_token(user_id: str, session_id: str, expires_hours: int = ACCESS_TOKEN_EXPIRE_HOURS) -> str:
    """Create a JWT access token."""
    expire = datetime.utcnow() + timedelta(hours=expires_hours)
    payload = {
        "sub": user_id,
        "sid": session_id,  # Session ID for invalidation
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token."""
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def set_session_cookie(response: Response, token: str) -> None:
    """Set the session cookie on a response."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,  # Required for cross-origin with SameSite=None
        samesite="none",  # Allow cross-origin (frontend on different subdomain)
        max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
    )


def clear_session_cookie(response: Response) -> None:
    """Clear the session cookie."""
    response.delete_cookie(key=SESSION_COOKIE_NAME)


class AuthMiddleware:
    """Authentication middleware for FastAPI."""

    def __init__(self, auth_store: AuthStore):
        self.auth_store = auth_store

    async def get_token_from_request(self, request: Request) -> str | None:
        """Extract token from request (cookie or header)."""
        # Try cookie first (web app)
        token = request.cookies.get(SESSION_COOKIE_NAME)

        # Fall back to Authorization header (API clients)
        if not token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]

        return token

    async def get_current_user(self, request: Request) -> User | None:
        """Extract and validate user from request."""
        token = await self.get_token_from_request(request)
        if not token:
            return None

        payload = decode_token(token)
        if not payload:
            return None

        user_id = payload.get("sub")
        session_id = payload.get("sid")

        if not user_id or not session_id:
            return None

        # Verify session is still valid
        session = await self.auth_store.get_session(session_id)
        if not session or not session.is_valid:
            return None

        if session.expires_at < datetime.utcnow():
            return None

        # Get user
        user = await self.auth_store.get_user(user_id)
        if not user or not user.is_active:
            return None

        return user

    async def require_auth(self, request: Request) -> User:
        """Dependency that requires authentication."""
        user = await self.get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    async def require_mfa_complete(self, request: Request) -> User:
        """Dependency that requires completed MFA."""
        user = await self.require_auth(request)
        if not user.mfa_enabled:
            raise HTTPException(
                status_code=403,
                detail="MFA setup required",
                headers={"X-Auth-State": AuthState.MFA_SETUP.value},
            )
        return user

    async def require_admin(self, request: Request) -> User:
        """Dependency that requires admin role."""
        user = await self.require_mfa_complete(request)
        if user.role.value != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return user


# Global auth store and middleware instances
_auth_store: AuthStore | None = None
_auth_middleware: AuthMiddleware | None = None


async def init_auth(db_path: str | None = None) -> AuthStore:
    """Initialize the auth system."""
    global _auth_store, _auth_middleware

    if _auth_store is None:
        from pathlib import Path

        if db_path is None:
            db_path = Path("~/memory/data/auth/auth.sqlite").expanduser()
        else:
            db_path = Path(db_path)

        _auth_store = AuthStore(db_path)
        await _auth_store.initialize()
        _auth_middleware = AuthMiddleware(_auth_store)

    return _auth_store


def get_auth_store() -> AuthStore:
    """Get the auth store instance."""
    if _auth_store is None:
        raise RuntimeError("Auth not initialized. Call init_auth first.")
    return _auth_store


def get_auth_middleware() -> AuthMiddleware:
    """Get the auth middleware instance."""
    if _auth_middleware is None:
        raise RuntimeError("Auth not initialized. Call init_auth first.")
    return _auth_middleware


# FastAPI dependencies
async def get_current_user(request: Request) -> User | None:
    """Dependency to get the current user (optional)."""
    middleware = get_auth_middleware()
    return await middleware.get_current_user(request)


async def require_auth(request: Request) -> User:
    """Dependency that requires authentication."""
    middleware = get_auth_middleware()
    return await middleware.require_auth(request)


async def require_mfa(request: Request) -> User:
    """Dependency that requires MFA to be set up."""
    middleware = get_auth_middleware()
    return await middleware.require_mfa_complete(request)


async def require_admin(request: Request) -> User:
    """Dependency that requires admin role."""
    middleware = get_auth_middleware()
    return await middleware.require_admin(request)
