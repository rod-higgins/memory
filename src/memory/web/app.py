"""FastAPI application for PLM Web UI."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv

# Try multiple locations for .env file
env_locations = [
    Path(__file__).parent.parent.parent.parent / ".env",  # /Users/rod/memory/.env
    Path.cwd() / ".env",
    Path.home() / "memory" / ".env",
]
for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path)
        break

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from ..api.memory_api import MemoryAPI  # noqa: E402
from ..auth import (  # noqa: E402
    AuthState,
    LoginRequest,
    MFAVerifyRequest,
    TokenResponse,
    User,
    UserCreate,
    UserUpdate,
)
from ..auth.middleware import (  # noqa: E402
    clear_session_cookie,
    create_access_token,
    get_auth_store,
    get_current_user,
    init_auth,
    require_admin,
    require_auth,
    require_mfa,
    set_session_cookie,
)
from ..schema import MemoryEntry, MemoryTier, MemoryType, TruthCategory  # noqa: E402


# Request/Response models
class MemoryCreate(BaseModel):
    """Request model for creating a memory."""

    content: str
    memory_type: str = "fact"
    truth_category: str = "contextual"
    confidence: float = 0.7
    domains: list[str] = []
    tags: list[str] = []
    tier: str = "persistent"  # short_term, long_term, persistent


class MemoryUpdate(BaseModel):
    """Request model for updating a memory."""

    content: str | None = None
    confidence: float | None = None
    domains: list[str] | None = None
    tags: list[str] | None = None


class SearchRequest(BaseModel):
    """Request model for searching memories."""

    query: str
    limit: int = 20
    tier: str | None = None
    memory_type: str | None = None
    semantic: bool = True


class IngestRequest(BaseModel):
    """Request model for ingestion."""

    source: str
    path: str | None = None
    options: dict[str, Any] = {}


class ChatRequest(BaseModel):
    """Request model for chat with LLM/SLM."""

    message: str
    provider: str = "claude"  # Provider ID (claude, chatgpt, ollama_model_name, etc.)
    max_memories: int = 10
    include_identity: bool = True


class CompareRequest(BaseModel):
    """Request model for LLM comparison."""

    message: str
    max_memories: int = 10
    providers: list[str] | None = None  # None = all providers


class SingleProviderRequest(BaseModel):
    """Request model for single provider query."""

    message: str
    provider: str
    with_context: bool = True
    max_memories: int = 10


class ConnectionCreate(BaseModel):
    """Request model for creating a connection."""

    source_id: str
    name: str
    credentials: dict[str, str]
    settings: dict[str, Any] = {}


class ConnectionUpdate(BaseModel):
    """Request model for updating a connection."""

    name: str | None = None
    credentials: dict[str, str] | None = None
    settings: dict[str, Any] | None = None
    sync_enabled: bool | None = None
    sync_interval_hours: int | None = None


class StatsResponse(BaseModel):
    """Response model for statistics."""

    total_memories: int
    by_tier: dict[str, int]
    by_type: dict[str, int]
    by_truth_category: dict[str, int]
    top_domains: list[dict[str, Any]]
    recent_activity: list[dict[str, Any]]


class LLMCreate(BaseModel):
    """Request model for creating an LLM configuration."""

    name: str
    provider: str  # anthropic, openai, google, cohere, mistral, bedrock, azure, custom
    model: str
    api_key: str | None = None
    base_url: str | None = None
    description: str | None = None


class LLMUpdate(BaseModel):
    """Request model for updating an LLM."""

    name: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    description: str | None = None
    enabled: bool | None = None


class SLMCreate(BaseModel):
    """Request model for creating an SLM configuration."""

    name: str
    runtime: str  # ollama, lmstudio, llamacpp, vllm, custom
    model: str
    endpoint: str | None = None


class SLMUpdate(BaseModel):
    """Request model for updating an SLM."""

    name: str | None = None
    runtime: str | None = None
    model: str | None = None
    endpoint: str | None = None
    enabled: bool | None = None


# Global API instance
_memory_api: MemoryAPI | None = None


async def get_api() -> MemoryAPI:
    """Get or create the MemoryAPI instance."""
    global _memory_api
    if _memory_api is None:
        _memory_api = MemoryAPI()
        await _memory_api.initialize()
    return _memory_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await init_auth()  # Initialize auth system
    await get_api()
    yield
    # Shutdown
    global _memory_api
    if _memory_api:
        await _memory_api.close()
        _memory_api = None
    # Close auth store
    try:
        auth_store = get_auth_store()
        await auth_store.close()
    except RuntimeError:
        pass  # Auth not initialized


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="PLM Memory Manager",
        description="Web interface for Personal Language Model memory management",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # =========================================================================
    # API Routes
    # =========================================================================

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main UI."""
        return get_index_html()

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "plm-memory"}

    # =========================================================================
    # Help Documentation Routes
    # =========================================================================

    @app.get("/api/help")
    async def list_help_pages():
        """List available help documentation pages."""
        help_dir = Path(__file__).parent.parent.parent.parent / "docs" / "help"
        pages = []
        if help_dir.exists():
            for md_file in sorted(help_dir.glob("*.md")):
                if md_file.name != "INDEX.md":
                    name = md_file.stem
                    title = name.replace("_", " ").title()
                    pages.append(
                        {
                            "id": name.lower(),
                            "title": title,
                            "file": md_file.name,
                        }
                    )
        return {"pages": pages}

    @app.get("/api/help/{page_id}")
    async def get_help_page(page_id: str):
        """Get a specific help documentation page."""
        help_dir = Path(__file__).parent.parent.parent.parent / "docs" / "help"

        # Map page_id to filename
        filename = f"{page_id.upper()}.md"
        if page_id.lower() == "index":
            filename = "INDEX.md"

        file_path = help_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Help page '{page_id}' not found")

        content = file_path.read_text()
        title = page_id.replace("_", " ").title()

        return {
            "id": page_id,
            "title": title,
            "content": content,
        }

    # =========================================================================
    # Authentication Routes
    # =========================================================================

    @app.get("/api/auth/status")
    async def auth_status(request: Request):
        """Check authentication status."""
        auth_store = get_auth_store()
        user = await get_current_user(request)

        # Check if any users exist (for initial setup)
        has_users = await auth_store.has_any_users()

        if not has_users:
            return {
                "authenticated": False,
                "auth_state": "setup_required",
                "message": "No users configured. Run setup script.",
            }

        if not user:
            return {
                "authenticated": False,
                "auth_state": AuthState.LOGIN.value,
            }

        return {
            "authenticated": True,
            "auth_state": AuthState.AUTHENTICATED.value if user.mfa_enabled else AuthState.MFA_SETUP.value,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "mfa_enabled": user.mfa_enabled,
            },
        }

    @app.post("/api/auth/login")
    async def login(request: LoginRequest, req: Request, response: Response):
        """Authenticate user with username/email and password."""
        auth_store = get_auth_store()

        # Get client info for logging
        ip_address = req.client.host if req.client else None
        user_agent = req.headers.get("user-agent")

        # Check for rate limiting
        failed_attempts = await auth_store.get_recent_failed_attempts(request.username_or_email)
        if failed_attempts >= 5:
            await auth_store.record_login_attempt(request.username_or_email, ip_address, False, "rate_limited")
            raise HTTPException(
                status_code=429,
                detail="Too many failed attempts. Please try again later.",
            )

        # Find user
        user = await auth_store.get_user_by_username_or_email(request.username_or_email)
        if not user:
            await auth_store.record_login_attempt(request.username_or_email, ip_address, False, "user_not_found")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not user.is_active:
            await auth_store.record_login_attempt(request.username_or_email, ip_address, False, "user_inactive")
            raise HTTPException(status_code=401, detail="Account is disabled")

        # Verify password
        if not await auth_store.verify_password(user.id, request.password):
            await auth_store.record_login_attempt(request.username_or_email, ip_address, False, "invalid_password")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Password verified - create session
        session = await auth_store.create_session(user.id, ip_address, user_agent)

        # Create JWT token
        token = create_access_token(user.id, session.id)
        set_session_cookie(response, token)

        # Record successful login
        await auth_store.record_login_attempt(request.username_or_email, ip_address, True)

        # Determine auth state
        auth_state = AuthState.AUTHENTICATED if user.mfa_enabled else AuthState.MFA_SETUP

        return TokenResponse(
            access_token=token,
            expires_in=24 * 3600,
            user=user,
            auth_state=auth_state,
        )

    @app.post("/api/auth/logout")
    async def logout(request: Request, response: Response):
        """Log out and invalidate session."""
        from ..auth.middleware import SESSION_COOKIE_NAME, decode_token

        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            payload = decode_token(token)
            if payload and "sid" in payload:
                auth_store = get_auth_store()
                await auth_store.invalidate_session(payload["sid"])

        clear_session_cookie(response)
        return {"status": "logged_out"}

    @app.get("/api/auth/mfa/setup")
    async def mfa_setup(user: User = Depends(require_auth)):
        """Get MFA setup data (QR code and backup codes)."""
        if user.mfa_enabled:
            raise HTTPException(status_code=400, detail="MFA already enabled")

        auth_store = get_auth_store()
        secret, qr_uri, backup_codes = await auth_store.setup_mfa_pending(user.id)

        # Generate QR code
        qr_base64 = auth_store.mfa_manager.generate_qr_code_base64(qr_uri)

        return {
            "secret": secret,
            "qr_code_uri": qr_uri,
            "qr_code_base64": qr_base64,
            "backup_codes": backup_codes,
        }

    @app.post("/api/auth/mfa/verify")
    async def mfa_verify(
        request: MFAVerifyRequest,
        response: Response,
        user: User = Depends(require_auth),
    ):
        """Verify TOTP code to complete MFA setup or login."""
        auth_store = get_auth_store()

        if not user.mfa_enabled:
            # This is MFA setup completion
            success = await auth_store.complete_mfa_setup(user.id, request.totp_code)
            if not success:
                raise HTTPException(status_code=400, detail="Invalid TOTP code")

            # Refresh user data
            user = await auth_store.get_user(user.id)
            return {
                "status": "mfa_enabled",
                "message": "Two-factor authentication enabled successfully",
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "mfa_enabled": user.mfa_enabled,
                },
            }
        else:
            # This is MFA verification for login
            success = await auth_store.verify_mfa(user.id, request.totp_code)
            if not success:
                raise HTTPException(status_code=400, detail="Invalid TOTP code")

            return {
                "status": "verified",
                "message": "MFA verification successful",
            }

    @app.post("/api/auth/mfa/backup")
    async def use_backup_code(
        backup_code: str,
        user: User = Depends(require_auth),
    ):
        """Use a backup code for MFA verification."""
        if not user.mfa_enabled:
            raise HTTPException(status_code=400, detail="MFA not enabled")

        auth_store = get_auth_store()
        success = await auth_store.verify_backup_code(user.id, backup_code)

        if not success:
            raise HTTPException(status_code=400, detail="Invalid backup code")

        return {"status": "verified", "message": "Backup code accepted"}

    @app.post("/api/auth/change-password")
    async def change_password(
        current_password: str = Query(...),
        new_password: str = Query(..., min_length=8),
        user: User = Depends(require_mfa),
    ):
        """Change the current user's password."""
        auth_store = get_auth_store()

        # Verify current password
        if not await auth_store.verify_password(user.id, current_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Update password
        await auth_store.update_password(user.id, new_password)

        # Invalidate other sessions (optional security measure)
        # await auth_store.invalidate_user_sessions(user.id)

        return {"status": "success", "message": "Password changed successfully"}

    # =========================================================================
    # Admin User Management Routes
    # =========================================================================

    @app.get("/api/admin/users")
    async def list_users(user: User = Depends(require_admin)):
        """List all users (admin only)."""
        auth_store = get_auth_store()
        users = await auth_store.list_users()
        return {
            "users": [
                {
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role.value,
                    "is_active": u.is_active,
                    "mfa_enabled": u.mfa_enabled,
                    "created_at": u.created_at.isoformat(),
                    "last_login": u.last_login.isoformat() if u.last_login else None,
                }
                for u in users
            ]
        }

    @app.post("/api/admin/users")
    async def create_user(request: UserCreate, user: User = Depends(require_admin)):
        """Create a new user (admin only)."""
        auth_store = get_auth_store()

        # Check if username or email already exists
        existing = await auth_store.get_user_by_username(request.username)
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")

        existing = await auth_store.get_user_by_email(request.email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already exists")

        new_user = await auth_store.create_user(request)
        return {
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "role": new_user.role.value,
            "data_path": new_user.data_path,
            "message": "User created successfully",
        }

    @app.get("/api/admin/users/{user_id}")
    async def get_user_admin(user_id: str, user: User = Depends(require_admin)):
        """Get user details (admin only)."""
        auth_store = get_auth_store()
        target_user = await auth_store.get_user(user_id)

        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": target_user.id,
            "username": target_user.username,
            "email": target_user.email,
            "role": target_user.role.value,
            "is_active": target_user.is_active,
            "mfa_enabled": target_user.mfa_enabled,
            "data_path": target_user.data_path,
            "created_at": target_user.created_at.isoformat(),
            "updated_at": target_user.updated_at.isoformat(),
            "last_login": target_user.last_login.isoformat() if target_user.last_login else None,
        }

    @app.patch("/api/admin/users/{user_id}")
    async def update_user_admin(
        user_id: str,
        request: UserUpdate,
        user: User = Depends(require_admin),
    ):
        """Update user (admin only)."""
        auth_store = get_auth_store()

        # Prevent admin from deactivating themselves
        if user_id == user.id and request.is_active is False:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

        updates = {}
        if request.email is not None:
            updates["email"] = request.email
        if request.is_active is not None:
            updates["is_active"] = request.is_active
        if request.role is not None:
            updates["role"] = request.role

        updated_user = await auth_store.update_user(user_id, **updates)
        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "id": updated_user.id,
            "username": updated_user.username,
            "email": updated_user.email,
            "role": updated_user.role.value,
            "is_active": updated_user.is_active,
            "message": "User updated successfully",
        }

    @app.delete("/api/admin/users/{user_id}")
    async def delete_user_admin(user_id: str, user: User = Depends(require_admin)):
        """Deactivate a user (admin only)."""
        if user_id == user.id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")

        auth_store = get_auth_store()
        success = await auth_store.delete_user(user_id)

        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        # Invalidate all user sessions
        await auth_store.invalidate_user_sessions(user_id)

        return {"status": "deleted", "message": "User deactivated"}

    @app.post("/api/admin/users/{user_id}/reset-mfa")
    async def reset_user_mfa(user_id: str, user: User = Depends(require_admin)):
        """Reset MFA for a user (admin only)."""
        auth_store = get_auth_store()
        success = await auth_store.reset_mfa(user_id)

        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        return {"status": "reset", "message": "MFA reset successfully. User will need to set up MFA on next login."}

    @app.post("/api/admin/users/{user_id}/reset-password")
    async def reset_user_password(
        user_id: str,
        new_password: str = Query(..., min_length=8),
        user: User = Depends(require_admin),
    ):
        """Reset password for a user (admin only)."""
        auth_store = get_auth_store()
        success = await auth_store.update_password(user_id, new_password)

        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        # Invalidate all user sessions to force re-login
        await auth_store.invalidate_user_sessions(user_id)

        return {"status": "reset", "message": "Password reset successfully. User will need to log in again."}

    # =========================================================================
    # Memory Routes (Protected)
    # =========================================================================

    @app.get("/api/stats")
    async def get_stats() -> StatsResponse:
        """Get memory statistics."""
        api = await get_api()
        stats = await api.get_stats()
        return StatsResponse(**stats)

    @app.get("/api/memories")
    async def list_memories(
        tier: str | None = None,
        memory_type: str | None = None,
        domain: str | None = None,
        limit: int = Query(default=50, le=500),
        offset: int = 0,
    ):
        """List memories with optional filtering."""
        api = await get_api()

        # Build filter criteria
        filters = {}
        if tier:
            filters["tier"] = tier
        if memory_type:
            filters["memory_type"] = memory_type
        if domain:
            filters["domain"] = domain

        memories, total = await api.list_memories(
            filters=filters if filters else None,
            limit=limit,
            offset=offset,
            return_total=True,
        )

        return {
            "memories": [_serialize_memory(m) for m in memories],
            "count": len(memories),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(memories) < total,
        }

    @app.get("/api/memories/{memory_id}")
    async def get_memory(memory_id: str):
        """Get a specific memory by ID."""
        api = await get_api()
        memory = await api.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        return _serialize_memory(memory)

    @app.post("/api/memories")
    async def create_memory(request: MemoryCreate):
        """Create a new memory."""
        api = await get_api()

        # Enum values are lowercase, so convert to lowercase for lookup
        memory_type = MemoryType(request.memory_type.lower())
        truth_category = TruthCategory(request.truth_category.lower())

        # Parse tier
        tier_map = {
            "short_term": MemoryTier.SHORT_TERM,
            "long_term": MemoryTier.LONG_TERM,
            "persistent": MemoryTier.PERSISTENT,
        }
        tier = tier_map.get(request.tier.lower(), MemoryTier.PERSISTENT)

        memory = await api.remember(
            content=request.content,
            memory_type=memory_type,
            truth_category=truth_category,
            domains=request.domains if request.domains else None,
            tags=request.tags if request.tags else None,
            tier=tier,
        )

        return _serialize_memory(memory)

    @app.put("/api/memories/{memory_id}")
    async def update_memory(memory_id: str, request: MemoryUpdate):
        """Update an existing memory."""
        api = await get_api()

        updates = {}
        if request.content is not None:
            updates["content"] = request.content
        if request.confidence is not None:
            updates["confidence"] = request.confidence
        if request.domains is not None:
            updates["domains"] = request.domains
        if request.tags is not None:
            updates["tags"] = request.tags

        memory = await api.update_memory(memory_id, **updates)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        return _serialize_memory(memory)

    @app.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        """Delete a memory."""
        api = await get_api()
        success = await api.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted", "id": memory_id}

    @app.post("/api/search")
    async def search_memories(request: SearchRequest):
        """Search memories."""
        api = await get_api()

        result = await api.search(
            query=request.query,
            limit=request.limit,
            semantic=request.semantic,
        )

        return {
            "query": request.query,
            "memories": [_serialize_memory(m) for m in result.memories],
            "count": len(result.memories),
            "search_time_ms": result.search_time_ms if hasattr(result, "search_time_ms") else None,
        }

    @app.get("/api/context")
    async def get_context(
        query: str = "",
        format: str = "claude",
        max_memories: int = 10,
    ):
        """Get context injection for a query. If no query, returns general context."""
        api = await get_api()
        # If no query, use a general one to get user profile context
        effective_query = query if query else "user profile and background"
        context = await api.get_context(
            query=effective_query,
            format=format,
            max_memories=max_memories,
        )
        return {"query": query, "context": context, "format": format}

    @app.get("/api/sources")
    async def list_sources():
        """List available data sources."""
        from ..ingestion.sources.registry import SourceRegistry

        registry = SourceRegistry()
        sources = registry.list_sources()

        return {
            "sources": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "category": s["category"],
                    "description": s["description"],
                }
                for s in sources
            ]
        }

    @app.post("/api/ingest")
    async def ingest_source(request: IngestRequest):
        """Trigger ingestion from a data source."""
        api = await get_api()

        try:
            result = await api.ingest(
                source=request.source,
                path=request.path,
                **request.options,
            )
            return {
                "status": "completed",
                "source": request.source,
                "memories_created": result.get("count", 0),
                "details": result,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/synthesis")
    async def run_synthesis():
        """Analyze memories and extract insights."""
        api = await get_api()

        # Get all memories for analysis
        memories, total = await api.list_memories(limit=1000, return_total=True)

        # Extract skills from skill-type memories
        skills = set()
        for m in memories:
            if m.memory_type.value == "skill" or "skill" in (m.tags or []):
                # Extract skill keywords from content
                content = m.content.lower()
                skill_keywords = [
                    "python",
                    "javascript",
                    "typescript",
                    "react",
                    "node",
                    "aws",
                    "azure",
                    "docker",
                    "kubernetes",
                    "devops",
                    "sql",
                    "api",
                    "drupal",
                    "php",
                    "java",
                    "go",
                    "rust",
                    "c++",
                    "machine learning",
                    "terraform",
                    "ansible",
                    "jenkins",
                    "git",
                    "agile",
                    "scrum",
                ]
                for skill in skill_keywords:
                    if skill in content:
                        skills.add(skill.title())

        # Extract topics from domains
        topic_counts = {}
        for m in memories:
            for d in m.domains or []:
                topic_counts[d] = topic_counts.get(d, 0) + 1
        topics = sorted(topic_counts.keys(), key=lambda x: -topic_counts[x])[:15]

        # Count entities
        entity_count = 0
        for m in memories:
            entity_count += len(m.entities or [])

        # Generate summary
        type_counts = {}
        for m in memories:
            t = m.memory_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        summary = f"Your memory contains {total} entries across {len(set(d for m in memories for d in (m.domains or [])))} domains. "
        if type_counts:
            top_type = max(type_counts, key=type_counts.get)
            summary += f"Most common type: {top_type} ({type_counts[top_type]} entries). "
        if skills:
            summary += f"Key skills identified: {', '.join(list(skills)[:5])}."

        # Count connections
        try:
            connections = await api.list_connections()
            connection_count = len(connections)
        except Exception:
            connection_count = 0

        return {
            "skills": list(skills),
            "topics": topics,
            "summary": summary,
            "stats": {
                "total": total,
                "domains": len(set(d for m in memories for d in (m.domains or []))),
                "entities": entity_count,
                "connections": connection_count,
            },
        }

    @app.post("/api/promote")
    async def run_promotion():
        """Run the promotion cycle."""
        api = await get_api()
        result = await api.run_promotion_cycle()
        return {
            "status": "completed",
            "promoted": result.get("promoted", 0),
            "demoted": result.get("demoted", 0),
        }

    @app.get("/api/identity")
    async def get_identity():
        """Get the user identity profile."""
        api = await get_api()
        identity = await api.get_identity()
        if not identity:
            return {"identity": None}
        return {"identity": identity}

    @app.put("/api/identity")
    async def update_identity(request: dict[str, Any]):
        """Update the user identity profile."""
        api = await get_api()
        identity = await api.update_identity(request)
        return {"identity": identity}

    # =========================================================================
    # Claude Integration Routes
    # =========================================================================

    @app.post("/api/chat")
    async def chat_with_provider(request: ChatRequest):
        """
        Chat with any LLM/SLM provider using memory context.

        Retrieves relevant memories based on the user's message,
        injects them as context, and calls the selected provider.
        """

        api = await get_api()

        # Get memory context
        context = await api.get_context(
            query=request.message,
            format="claude",
            max_memories=request.max_memories,
        )

        # Build system prompt with memory context
        system_parts = []
        if request.include_identity:
            identity = await api.get_identity()
            if identity and identity.get("name"):
                system_parts.append(f"You are helping {identity.get('name')}.")

        system_parts.append(
            "You have access to the user's personal memories and context. "
            "Use this information to provide personalized, relevant responses."
        )

        if context:
            system_parts.append(f"\n\n{context}")

        system_prompt = "\n".join(system_parts)

        provider_id = request.provider

        # Handle Ollama/SLM providers
        if provider_id.startswith("ollama_") or provider_id.startswith("slm_"):
            return await _chat_with_ollama(request, system_prompt, context, provider_id)

        # Handle built-in LLM providers
        if provider_id in ("claude", "chatgpt", "copilot", "amazonq"):
            return await _chat_with_builtin_llm(request, system_prompt, context, provider_id)

        # Handle user-configured LLMs
        if provider_id.startswith("llm_"):
            return await _chat_with_custom_llm(request, system_prompt, context, provider_id)

        # Default to Claude
        return await _chat_with_builtin_llm(request, system_prompt, context, "claude")

    async def _chat_with_ollama(request: ChatRequest, system_prompt: str, context: str, provider_id: str):
        """Chat with a local Ollama model."""
        import httpx

        # Extract model name from provider ID
        if provider_id.startswith("ollama_"):
            model_name = provider_id.replace("ollama_", "").replace("_", ":")
        else:
            # SLM from config
            config = _load_models_config()
            slm_id = provider_id.replace("slm_", "")
            slm = next((s for s in config.get("slms", []) if s.get("id") == slm_id), None)
            if not slm:
                raise HTTPException(status_code=404, detail=f"SLM not found: {provider_id}")
            model_name = slm.get("model")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": request.message},
                        ],
                        "stream": False,
                    },
                )

                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {response.text}")

                result = response.json()
                assistant_response = result.get("message", {}).get("content", "No response")

                return {
                    "response": assistant_response,
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                    "provider": model_name,
                }

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Ollama timeout - model may be loading")
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama not running. Start with: ollama serve")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def _chat_with_builtin_llm(request: ChatRequest, system_prompt: str, context: str, provider_id: str):
        """Chat with a built-in LLM provider (Claude, ChatGPT, etc.)."""
        import httpx

        # Load any stored API key overrides
        config = _load_models_config()
        builtin_overrides = config.get("builtin_llms", {})
        override = builtin_overrides.get(provider_id, {})

        if provider_id == "claude":
            api_key = override.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            model = override.get("model", "claude-sonnet-4-20250514")

            if not api_key:
                return {
                    "response": "Claude API key not configured. Go to LLMs settings to configure.",
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                }

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": model,
                            "max_tokens": 2048,
                            "system": system_prompt,
                            "messages": [{"role": "user", "content": request.message}],
                        },
                    )

                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code, detail=f"Claude API error: {response.text}"
                        )

                    result = response.json()
                    return {
                        "response": result["content"][0]["text"],
                        "memories_used": request.max_memories,
                        "context_preview": context[:200] if context else None,
                        "provider": "Claude",
                    }
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Claude API timeout")

        elif provider_id == "chatgpt":
            api_key = override.get("api_key") or os.environ.get("OPENAI_API_KEY")
            model = override.get("model", "gpt-4o")

            if not api_key:
                return {
                    "response": "OpenAI API key not configured. Go to LLMs settings to configure.",
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                }

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "max_tokens": 2048,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": request.message},
                            ],
                        },
                    )

                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code, detail=f"OpenAI API error: {response.text}"
                        )

                    result = response.json()
                    return {
                        "response": result["choices"][0]["message"]["content"],
                        "memories_used": request.max_memories,
                        "context_preview": context[:200] if context else None,
                        "provider": "ChatGPT",
                    }
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="OpenAI API timeout")

        elif provider_id == "copilot":
            # GitHub Copilot via gh CLI
            import subprocess

            try:
                full_prompt = f"{system_prompt}\n\nUser: {request.message}"
                result = subprocess.run(
                    ["gh", "copilot", "explain", full_prompt], capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    return {
                        "response": f"Copilot error: {result.stderr or 'Unknown error'}. Make sure gh CLI is authenticated.",
                        "memories_used": request.max_memories,
                        "context_preview": context[:200] if context else None,
                    }
                return {
                    "response": result.stdout.strip(),
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                    "provider": "GitHub Copilot",
                }
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail="Copilot timeout")
            except FileNotFoundError:
                return {
                    "response": "GitHub CLI (gh) not found. Install it to use Copilot.",
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                }

        elif provider_id == "amazonq":
            # Amazon Bedrock
            try:
                import boto3

                client = boto3.client("bedrock-runtime", region_name="us-east-1")
                model_id = override.get("model", "amazon.titan-text-express-v1")

                body = {
                    "inputText": f"{system_prompt}\n\nUser: {request.message}",
                    "textGenerationConfig": {"maxTokenCount": 2048, "temperature": 0.7},
                }

                response = client.invoke_model(
                    modelId=model_id, body=str(body), contentType="application/json", accept="application/json"
                )

                result = response["body"].read().decode()
                import json

                result_json = json.loads(result)

                return {
                    "response": result_json.get("results", [{}])[0].get("outputText", "No response"),
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                    "provider": "Amazon Bedrock",
                }
            except ImportError:
                return {
                    "response": "boto3 not installed. Run: pip install boto3",
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                }
            except Exception as e:
                return {
                    "response": f"Bedrock error: {str(e)}",
                    "memories_used": request.max_memories,
                    "context_preview": context[:200] if context else None,
                }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_id}")

    async def _chat_with_custom_llm(request: ChatRequest, system_prompt: str, context: str, provider_id: str):
        """Chat with a user-configured custom LLM."""
        import httpx

        config = _load_models_config()
        llm_id = provider_id.replace("llm_", "")
        llm = next((l for l in config.get("llms", []) if l.get("id") == llm_id), None)

        if not llm:
            raise HTTPException(status_code=404, detail=f"LLM not found: {provider_id}")

        provider = llm.get("provider", "").lower()
        api_key = llm.get("api_key")
        model = llm.get("model")
        base_url = llm.get("base_url")

        # Route to appropriate API based on provider type
        if provider in ("anthropic", "claude"):
            if not api_key:
                return {"response": "API key not configured for this LLM.", "memories_used": request.max_memories}

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url or 'https://api.anthropic.com'}/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 2048,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": request.message}],
                    },
                )
                if response.status_code == 200:
                    return {
                        "response": response.json()["content"][0]["text"],
                        "memories_used": request.max_memories,
                        "provider": llm.get("name"),
                    }
                raise HTTPException(status_code=response.status_code, detail=response.text)

        elif provider in ("openai", "chatgpt"):
            if not api_key:
                return {"response": "API key not configured for this LLM.", "memories_used": request.max_memories}

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url or 'https://api.openai.com'}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "max_tokens": 2048,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": request.message},
                        ],
                    },
                )
                if response.status_code == 200:
                    return {
                        "response": response.json()["choices"][0]["message"]["content"],
                        "memories_used": request.max_memories,
                        "provider": llm.get("name"),
                    }
                raise HTTPException(status_code=response.status_code, detail=response.text)

        else:
            # Generic OpenAI-compatible API
            async with httpx.AsyncClient(timeout=60.0) as client:
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": request.message},
                        ],
                    },
                )
                if response.status_code == 200:
                    return {
                        "response": response.json()["choices"][0]["message"]["content"],
                        "memories_used": request.max_memories,
                        "provider": llm.get("name"),
                    }
                raise HTTPException(status_code=response.status_code, detail=response.text)

    # =========================================================================
    # LLM Comparison Routes
    # =========================================================================

    @app.get("/api/compare/providers")
    async def get_providers():
        """Get available LLM providers and their status."""
        import httpx

        from .llm_compare import LLMCompare

        compare = LLMCompare()
        providers = compare.get_available_providers()

        # Add configured LLMs and SLMs
        config = _load_models_config()
        slm_overrides = config.get("slm_overrides", {})

        # Add user-configured LLMs
        for llm in config.get("llms", []):
            if llm.get("enabled", True):
                providers.append(
                    {
                        "id": f"llm_{llm['id']}",
                        "name": llm["name"],
                        "model": llm["model"],
                        "available": True,
                        "config_help": f"{llm['provider']} - {llm.get('description', 'Custom configured')}",
                        "type": "llm",
                        "provider": llm["provider"],
                    }
                )

        # Add user-configured SLMs
        configured_models = set()
        for slm in config.get("slms", []):
            configured_models.add(slm.get("model"))
            if slm.get("enabled", True):
                providers.append(
                    {
                        "id": f"slm_{slm['id']}",
                        "name": slm["name"],
                        "model": slm["model"],
                        "available": True,
                        "config_help": f"{slm['runtime']} - Local model",
                        "type": "slm",
                        "runtime": slm["runtime"],
                        "endpoint": slm["endpoint"],
                    }
                )

        # Add auto-detected Ollama models
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        model_name = model.get("name", "")
                        if model_name and model_name not in configured_models:
                            override = slm_overrides.get(model_name, {})
                            if override.get("enabled", True):
                                display_name = override.get("name", model_name.split(":")[0].title())
                                providers.append(
                                    {
                                        "id": f"ollama_{model_name.replace(':', '_')}",
                                        "name": f"{display_name} (Local)",
                                        "model": model_name,
                                        "available": True,
                                        "config_help": "Ollama - Local model",
                                        "type": "slm",
                                        "runtime": "ollama",
                                        "endpoint": "http://localhost:11434",
                                    }
                                )
        except Exception:
            pass  # Ollama not running

        return {"providers": providers}

    @app.post("/api/compare")
    async def compare_llms(request: CompareRequest):
        """
        Compare LLM responses with and without PLM context.

        Sends the same query to all (or specified) LLMs twice:
        - Once with memory context injected
        - Once without any context

        Returns responses from all providers for comparison.
        """
        from .llm_compare import LLMCompare

        api = await get_api()
        compare = LLMCompare()

        # Get memory context
        context = await api.get_context(
            query=request.message,
            format="claude",
            max_memories=request.max_memories,
        )

        # System prompt for all LLMs
        system_prompt = "You are a helpful AI assistant. Answer the user's question directly and concisely."

        # Run comparison
        results = await compare.compare(
            message=request.message,
            context=context,
            system_prompt=system_prompt,
            providers=request.providers,
        )

        # Format response
        def format_response(resp):
            return {
                "provider": resp.provider,
                "model": resp.model,
                "response": resp.response,
                "latency_ms": round(resp.latency_ms, 2),
                "error": resp.error,
            }

        return {
            "query": request.message,
            "context_length": len(context) if context else 0,
            "context_preview": context[:300] if context else None,
            "with_context": {k: format_response(v) for k, v in results["with_context"].items()},
            "without_context": {k: format_response(v) for k, v in results["without_context"].items()},
        }

    @app.post("/api/compare/single")
    async def compare_single_provider(request: SingleProviderRequest):
        """Query a single provider with or without context."""
        from .llm_compare import LLMCompare

        api = await get_api()
        compare = LLMCompare()

        context = None
        if request.with_context:
            context = await api.get_context(
                query=request.message,
                format="claude",
                max_memories=request.max_memories,
            )

        result = await compare.query_single(
            provider_id=request.provider,
            message=request.message,
            context=context,
        )

        return {
            "provider": result.provider,
            "model": result.model,
            "response": result.response,
            "with_context": result.with_context,
            "latency_ms": round(result.latency_ms, 2),
            "error": result.error,
            "context_preview": context[:200] if context else None,
        }

    # =========================================================================
    # Connection Management Routes
    # =========================================================================

    @app.get("/api/connections")
    async def list_connections(source_id: str | None = None):
        """List all connections."""
        from .connections import ConnectionManager

        manager = ConnectionManager()
        connections = manager.list_connections(source_id)

        return {
            "connections": [
                {
                    "id": c.id,
                    "source_id": c.source_id,
                    "name": c.name,
                    "auth_type": c.auth_type.value,
                    "status": c.status.value,
                    "last_sync": c.last_sync.isoformat() if c.last_sync else None,
                    "last_error": c.last_error,
                    "sync_enabled": c.sync_enabled,
                    "sync_interval_hours": c.sync_interval_hours,
                    "created_at": c.created_at.isoformat(),
                    "settings": c.settings,
                }
                for c in connections
            ]
        }

    @app.get("/api/connections/{connection_id}")
    async def get_connection(connection_id: str):
        """Get a specific connection."""
        from .connections import ConnectionManager

        manager = ConnectionManager()
        conn = manager.get_connection(connection_id)

        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        return {
            "id": conn.id,
            "source_id": conn.source_id,
            "name": conn.name,
            "auth_type": conn.auth_type.value,
            "status": conn.status.value,
            "last_sync": conn.last_sync.isoformat() if conn.last_sync else None,
            "last_error": conn.last_error,
            "sync_enabled": conn.sync_enabled,
            "sync_interval_hours": conn.sync_interval_hours,
            "created_at": conn.created_at.isoformat(),
            "settings": conn.settings,
            # Don't expose credentials
        }

    @app.post("/api/connections")
    async def create_connection(request: ConnectionCreate):
        """Create a new connection."""
        from .connections import ConnectionManager

        manager = ConnectionManager()

        try:
            conn = manager.create_connection(
                source_id=request.source_id,
                name=request.name,
                credentials=request.credentials,
                settings=request.settings,
            )

            return {
                "id": conn.id,
                "source_id": conn.source_id,
                "name": conn.name,
                "status": conn.status.value,
                "message": "Connection created",
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.put("/api/connections/{connection_id}")
    async def update_connection(connection_id: str, request: ConnectionUpdate):
        """Update a connection."""
        from .connections import ConnectionManager

        manager = ConnectionManager()
        conn = manager.update_connection(
            connection_id=connection_id,
            name=request.name,
            credentials=request.credentials,
            settings=request.settings,
            sync_enabled=request.sync_enabled,
            sync_interval_hours=request.sync_interval_hours,
        )

        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        return {
            "id": conn.id,
            "name": conn.name,
            "status": conn.status.value,
            "message": "Connection updated",
        }

    @app.delete("/api/connections/{connection_id}")
    async def delete_connection(connection_id: str):
        """Delete a connection."""
        from .connections import ConnectionManager

        manager = ConnectionManager()
        success = manager.delete_connection(connection_id)

        if not success:
            raise HTTPException(status_code=404, detail="Connection not found")

        return {"status": "deleted", "id": connection_id}

    @app.post("/api/connections/{connection_id}/test")
    async def test_connection(connection_id: str):
        """Test a connection."""
        from .connections import ConnectionManager

        manager = ConnectionManager()
        success, message = manager.test_connection(connection_id)

        return {
            "success": success,
            "message": message,
            "connection_id": connection_id,
        }

    @app.post("/api/connections/{connection_id}/sync")
    async def sync_connection(connection_id: str):
        """Trigger sync for a connection."""
        from .connections import ConnectionManager

        manager = ConnectionManager()
        conn = manager.get_connection(connection_id)

        if not conn:
            raise HTTPException(status_code=404, detail="Connection not found")

        # Get the API and trigger ingestion
        api = await get_api()

        try:
            # Build ingestion parameters from connection
            ingest_params = {**conn.credentials, **conn.settings}
            result = await api.ingest(
                source=conn.source_id,
                **ingest_params,
            )

            # Update last sync
            manager.store.update_last_sync(connection_id)

            return {
                "status": "completed",
                "connection_id": connection_id,
                "memories_created": result.get("count", 0),
            }
        except Exception as e:
            from .connections import ConnectionStatus

            manager.store.update_status(connection_id, ConnectionStatus.ERROR, str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sources/config")
    async def get_sources_config():
        """Get all source configurations with auth requirements."""
        from .connections import ConnectionManager

        manager = ConnectionManager()
        return {"sources": manager.get_all_source_configs()}

    # =========================================================================
    # LLM and SLM Management Endpoints
    # =========================================================================

    def _get_models_db_path() -> Path:
        """Get path to models configuration database."""
        return Path.home() / "memory" / "data" / "persistent" / "models.json"

    def _load_models_config() -> dict:
        """Load LLM and SLM configurations from disk."""
        import json

        path = _get_models_db_path()
        if path.exists():
            return json.loads(path.read_text())
        return {"llms": [], "slms": []}

    def _save_models_config(config: dict):
        """Save LLM and SLM configurations to disk."""
        import json

        path = _get_models_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))

    @app.get("/api/llms")
    async def list_llms():
        """List all configured LLMs, including built-in providers."""
        from .llm_compare import LLMCompare

        config = _load_models_config()
        builtin_overrides = config.get("builtin_llms", {})

        # Get built-in providers
        compare = LLMCompare()
        built_in = []
        for name, provider in compare.providers.items():
            override = builtin_overrides.get(name, {})
            # Check if API key is configured (either in override or env)
            has_api_key = bool(override.get("api_key")) or provider.is_available()
            built_in.append(
                {
                    "id": f"builtin_{name}",
                    "name": provider.display_name,
                    "provider": name,
                    "model": override.get("model", provider.model),
                    "enabled": has_api_key,
                    "description": provider.get_config_help(),
                    "built_in": True,
                }
            )

        # Get user-configured LLMs
        user_llms = config.get("llms", [])
        for llm in user_llms:
            llm["built_in"] = False

        return built_in + user_llms

    @app.post("/api/llms")
    async def create_llm(request: LLMCreate):
        """Add a new LLM configuration."""
        import uuid

        config = _load_models_config()
        llm = {
            "id": str(uuid.uuid4()),
            "name": request.name,
            "provider": request.provider,
            "model": request.model,
            "api_key": request.api_key,
            "base_url": request.base_url,
            "description": request.description,
            "enabled": True,
            "created_at": __import__("datetime").datetime.now().isoformat(),
        }
        config.setdefault("llms", []).append(llm)
        _save_models_config(config)
        return llm

    @app.patch("/api/llms/{llm_id}")
    async def update_llm(llm_id: str, request: LLMUpdate):
        """Update an LLM configuration."""
        config = _load_models_config()

        # Handle built-in provider updates
        if llm_id.startswith("builtin_"):
            provider_name = llm_id.replace("builtin_", "")
            builtin_overrides = config.setdefault("builtin_llms", {})
            override = builtin_overrides.setdefault(provider_name, {})

            if request.model is not None:
                override["model"] = request.model
            if request.api_key:  # Only update if non-empty
                override["api_key"] = request.api_key
                # Also set as environment variable for this session
                env_key_map = {
                    "claude": "ANTHROPIC_API_KEY",
                    "chatgpt": "OPENAI_API_KEY",
                    "copilot": "GITHUB_TOKEN",
                    "amazonq": "AWS_ACCESS_KEY_ID",
                }
                if provider_name in env_key_map:
                    os.environ[env_key_map[provider_name]] = request.api_key

            _save_models_config(config)
            return {"id": llm_id, "provider": provider_name, **override, "built_in": True}

        # Handle user-configured LLM updates
        for llm in config.get("llms", []):
            if llm["id"] == llm_id:
                if request.name is not None:
                    llm["name"] = request.name
                if request.model is not None:
                    llm["model"] = request.model
                if request.api_key is not None:
                    llm["api_key"] = request.api_key
                if request.base_url is not None:
                    llm["base_url"] = request.base_url
                if request.description is not None:
                    llm["description"] = request.description
                if request.enabled is not None:
                    llm["enabled"] = request.enabled
                _save_models_config(config)
                return llm
        raise HTTPException(status_code=404, detail="LLM not found")

    @app.delete("/api/llms/{llm_id}")
    async def delete_llm(llm_id: str):
        """Delete an LLM configuration."""
        config = _load_models_config()
        llms = config.get("llms", [])
        config["llms"] = [l for l in llms if l["id"] != llm_id]
        _save_models_config(config)
        return {"status": "deleted"}

    @app.post("/api/llms/{llm_id}/test")
    async def test_llm(llm_id: str):
        """Test an LLM configuration."""
        import time

        from .llm_compare import LLMCompare

        start = time.time()

        # Check if it's a built-in provider
        if llm_id.startswith("builtin_"):
            provider_name = llm_id.replace("builtin_", "")

            # Load stored API key if available
            config = _load_models_config()
            builtin_overrides = config.get("builtin_llms", {})
            override = builtin_overrides.get(provider_name, {})

            # Set API key in environment if stored
            if override.get("api_key"):
                env_key_map = {
                    "claude": "ANTHROPIC_API_KEY",
                    "chatgpt": "OPENAI_API_KEY",
                    "copilot": "GITHUB_TOKEN",
                    "amazonq": "AWS_ACCESS_KEY_ID",
                }
                if provider_name in env_key_map:
                    os.environ[env_key_map[provider_name]] = override["api_key"]

            compare = LLMCompare()
            provider = compare.providers.get(provider_name)
            if not provider:
                raise HTTPException(status_code=404, detail="Provider not found")

            try:
                response = await provider.query("Say 'Hello!' in exactly one word.")
                latency = int((time.time() - start) * 1000)
                if response.error:
                    return {"success": False, "error": response.error}
                return {
                    "success": True,
                    "message": f"Connected to {provider.display_name}",
                    "response": response.response[:100],
                    "latency": latency,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # User-configured LLM
        config = _load_models_config()
        llm = next((l for l in config.get("llms", []) if l["id"] == llm_id), None)
        if not llm:
            raise HTTPException(status_code=404, detail="LLM not found")

        try:
            provider = llm["provider"]
            model = llm["model"]
            api_key = (
                llm.get("api_key")
                or os.environ.get(f"{provider.upper()}_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            )

            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model,
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Say 'Hello!' in exactly one word."}],
                )
                text = response.content[0].text if response.content else "OK"
            elif provider == "openai":
                import openai

                client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Say 'Hello!' in exactly one word."}],
                )
                text = response.choices[0].message.content if response.choices else "OK"
            else:
                text = "Provider test not implemented"

            latency = int((time.time() - start) * 1000)
            return {
                "success": True,
                "message": f"Connected to {llm['name']}",
                "response": text[:100],
                "latency": latency,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/slms")
    async def list_slms():
        """List all configured SLMs, including detected Ollama models."""
        import httpx

        config = _load_models_config()
        user_slms = config.get("slms", [])
        slm_overrides = config.get("slm_overrides", {})

        # Get configured model names to avoid duplicates
        configured_models = {s.get("model") for s in user_slms}

        # Detect Ollama models
        ollama_models = []
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        model_name = model.get("name", "")
                        if model_name and model_name not in configured_models:
                            override = slm_overrides.get(model_name, {})
                            ollama_models.append(
                                {
                                    "id": f"ollama_{model_name.replace(':', '_')}",
                                    "name": override.get("name", model_name.split(":")[0].title()),
                                    "runtime": "ollama",
                                    "model": model_name,
                                    "endpoint": "http://localhost:11434",
                                    "enabled": override.get("enabled", True),
                                    "size": model.get("size"),
                                    "detected": True,
                                }
                            )
        except Exception:
            pass  # Ollama not running or not accessible

        # Mark user SLMs as not detected (user-configured)
        for slm in user_slms:
            slm["detected"] = False

        return ollama_models + user_slms

    @app.post("/api/slms")
    async def create_slm(request: SLMCreate):
        """Add a new SLM configuration."""
        import uuid

        config = _load_models_config()
        # Set default endpoints based on runtime
        endpoint = request.endpoint
        if not endpoint:
            if request.runtime == "ollama":
                endpoint = "http://localhost:11434"
            elif request.runtime == "lmstudio":
                endpoint = "http://localhost:1234/v1"
            else:
                endpoint = "http://localhost:8000"

        slm = {
            "id": str(uuid.uuid4()),
            "name": request.name,
            "runtime": request.runtime,
            "model": request.model,
            "endpoint": endpoint,
            "enabled": True,
            "created_at": __import__("datetime").datetime.now().isoformat(),
        }
        config.setdefault("slms", []).append(slm)
        _save_models_config(config)
        return slm

    @app.patch("/api/slms/{slm_id}")
    async def update_slm(slm_id: str, request: SLMUpdate):
        """Update an SLM configuration."""
        config = _load_models_config()

        # Handle detected Ollama models
        if slm_id.startswith("ollama_"):
            model_name = slm_id.replace("ollama_", "").replace("_", ":")
            slm_overrides = config.setdefault("slm_overrides", {})
            override = slm_overrides.setdefault(model_name, {})

            if request.name is not None:
                override["name"] = request.name
            if request.enabled is not None:
                override["enabled"] = request.enabled

            _save_models_config(config)
            return {"id": slm_id, "model": model_name, **override, "detected": True}

        # Handle user-configured SLMs
        for slm in config.get("slms", []):
            if slm["id"] == slm_id:
                if request.name is not None:
                    slm["name"] = request.name
                if request.runtime is not None:
                    slm["runtime"] = request.runtime
                if request.model is not None:
                    slm["model"] = request.model
                if request.endpoint is not None:
                    slm["endpoint"] = request.endpoint
                if request.enabled is not None:
                    slm["enabled"] = request.enabled
                _save_models_config(config)
                return slm
        raise HTTPException(status_code=404, detail="SLM not found")

    @app.delete("/api/slms/{slm_id}")
    async def delete_slm(slm_id: str):
        """Delete an SLM configuration."""
        config = _load_models_config()
        slms = config.get("slms", [])
        config["slms"] = [s for s in slms if s["id"] != slm_id]
        _save_models_config(config)
        return {"status": "deleted"}

    @app.post("/api/slms/{slm_id}/test")
    async def test_slm(slm_id: str):
        """Test an SLM configuration."""
        import time

        import httpx

        start = time.time()

        # Handle detected Ollama models
        if slm_id.startswith("ollama_"):
            model_name = slm_id.replace("ollama_", "").replace("_", ":")
            endpoint = "http://localhost:11434"
            runtime = "ollama"
            model = model_name
        else:
            # User-configured SLM
            config = _load_models_config()
            slm = next((s for s in config.get("slms", []) if s["id"] == slm_id), None)
            if not slm:
                raise HTTPException(status_code=404, detail="SLM not found")
            endpoint = slm["endpoint"]
            model = slm["model"]
            runtime = slm["runtime"]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if runtime == "ollama":
                    response = await client.post(
                        f"{endpoint}/api/generate", json={"model": model, "prompt": "Say hello!", "stream": False}
                    )
                    data = response.json()
                    text = data.get("response", "OK")[:100]
                else:
                    # OpenAI-compatible API
                    response = await client.post(
                        f"{endpoint}/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Say hello!"}],
                            "max_tokens": 50,
                        },
                    )
                    data = response.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "OK")[:100]

            latency = int((time.time() - start) * 1000)
            return {"success": True, "message": f"Connected to {slm['name']}", "response": text, "latency": latency}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/slms/ollama/status")
    async def get_ollama_status():
        """Get Ollama server status and available models."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if Ollama is running
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return {
                        "running": True,
                        "url": "http://localhost:11434",
                        "models": models,
                        "models_count": len(models),
                    }
        except Exception:
            pass
        return {"running": False, "url": "http://localhost:11434", "models": [], "models_count": 0}

    @app.get("/api/slms/ollama/models")
    async def list_ollama_models():
        """List available Ollama models."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    @app.get("/api/slms/ollama/available")
    async def list_available_ollama_models():
        """List popular open-source models available to pull from Ollama."""
        # Comprehensive list of popular open-source models up to 7B
        # Organized by model family, all with permissive licenses
        models = [
            # === Qwen Family (Alibaba) - Apache 2.0 ===
            {
                "name": "qwen2.5:0.5b",
                "display_name": "Qwen 2.5 0.5B",
                "size": "398 MB",
                "description": "Tiny but capable, great for edge devices",
                "license": "Apache 2.0",
                "category": "qwen",
            },
            {
                "name": "qwen2.5:1.5b",
                "display_name": "Qwen 2.5 1.5B",
                "size": "986 MB",
                "description": "Lightweight multilingual model",
                "license": "Apache 2.0",
                "category": "qwen",
            },
            {
                "name": "qwen2.5:3b",
                "display_name": "Qwen 2.5 3B",
                "size": "1.9 GB",
                "description": "Balanced size and capability",
                "license": "Apache 2.0",
                "category": "qwen",
            },
            {
                "name": "qwen2.5:7b",
                "display_name": "Qwen 2.5 7B",
                "size": "4.7 GB",
                "description": "Full-featured multilingual model",
                "license": "Apache 2.0",
                "category": "qwen",
            },
            {
                "name": "qwen2.5-coder:1.5b",
                "display_name": "Qwen 2.5 Coder 1.5B",
                "size": "986 MB",
                "description": "Code-optimized, lightweight",
                "license": "Apache 2.0",
                "category": "qwen",
            },
            {
                "name": "qwen2.5-coder:3b",
                "display_name": "Qwen 2.5 Coder 3B",
                "size": "1.9 GB",
                "description": "Code-optimized, balanced",
                "license": "Apache 2.0",
                "category": "qwen",
            },
            {
                "name": "qwen2.5-coder:7b",
                "display_name": "Qwen 2.5 Coder 7B",
                "size": "4.7 GB",
                "description": "Excellent code generation",
                "license": "Apache 2.0",
                "category": "qwen",
            },
            # === Llama Family (Meta) ===
            {
                "name": "llama3.2:1b",
                "display_name": "Llama 3.2 1B",
                "size": "1.3 GB",
                "description": "Ultra-lightweight, fast responses",
                "license": "Llama 3.2 Community",
                "category": "llama",
            },
            {
                "name": "llama3.2:3b",
                "display_name": "Llama 3.2 3B",
                "size": "2.0 GB",
                "description": "Great balance of speed and quality",
                "license": "Llama 3.2 Community",
                "category": "llama",
            },
            {
                "name": "llama3.1:8b",
                "display_name": "Llama 3.1 8B",
                "size": "4.7 GB",
                "description": "Powerful general-purpose model",
                "license": "Llama 3.1 Community",
                "category": "llama",
            },
            {
                "name": "codellama:7b",
                "display_name": "Code Llama 7B",
                "size": "3.8 GB",
                "description": "Specialized for code generation",
                "license": "Llama 2 Community",
                "category": "llama",
            },
            {
                "name": "tinyllama:latest",
                "display_name": "TinyLlama 1.1B",
                "size": "637 MB",
                "description": "Extremely small and fast",
                "license": "Apache 2.0",
                "category": "llama",
            },
            # === Phi Family (Microsoft) - MIT ===
            {
                "name": "phi:2.7b",
                "display_name": "Phi-2 2.7B",
                "size": "1.6 GB",
                "description": "Compact reasoning model",
                "license": "MIT",
                "category": "phi",
            },
            {
                "name": "phi3:mini",
                "display_name": "Phi-3 Mini 3.8B",
                "size": "2.2 GB",
                "description": "Efficient small language model",
                "license": "MIT",
                "category": "phi",
            },
            {
                "name": "phi3.5:latest",
                "display_name": "Phi-3.5 Mini",
                "size": "2.2 GB",
                "description": "Latest Phi with improved capabilities",
                "license": "MIT",
                "category": "phi",
            },
            # === Gemma Family (Google) ===
            {
                "name": "gemma:2b",
                "display_name": "Gemma 2B",
                "size": "1.4 GB",
                "description": "Google's lightweight model",
                "license": "Gemma Terms",
                "category": "gemma",
            },
            {
                "name": "gemma:7b",
                "display_name": "Gemma 7B",
                "size": "4.8 GB",
                "description": "Full-featured Gemma",
                "license": "Gemma Terms",
                "category": "gemma",
            },
            {
                "name": "gemma2:2b",
                "display_name": "Gemma 2 2B",
                "size": "1.6 GB",
                "description": "Improved lightweight model",
                "license": "Gemma Terms",
                "category": "gemma",
            },
            {
                "name": "codegemma:7b",
                "display_name": "CodeGemma 7B",
                "size": "4.8 GB",
                "description": "Code-optimized Gemma",
                "license": "Gemma Terms",
                "category": "gemma",
            },
            # === Mistral Family - Apache 2.0 ===
            {
                "name": "mistral:7b",
                "display_name": "Mistral 7B",
                "size": "4.1 GB",
                "description": "Excellent reasoning and instruction following",
                "license": "Apache 2.0",
                "category": "mistral",
            },
            {
                "name": "mistral-nemo:latest",
                "display_name": "Mistral Nemo 12B",
                "size": "7.1 GB",
                "description": "State-of-the-art small model",
                "license": "Apache 2.0",
                "category": "mistral",
            },
            # === DeepSeek Family ===
            {
                "name": "deepseek-coder:1.3b",
                "display_name": "DeepSeek Coder 1.3B",
                "size": "776 MB",
                "description": "Lightweight code model",
                "license": "DeepSeek License",
                "category": "deepseek",
            },
            {
                "name": "deepseek-coder:6.7b",
                "display_name": "DeepSeek Coder 6.7B",
                "size": "3.8 GB",
                "description": "Excellent code completion",
                "license": "DeepSeek License",
                "category": "deepseek",
            },
            # === StarCoder Family - BigCode OpenRAIL-M ===
            {
                "name": "starcoder2:3b",
                "display_name": "StarCoder2 3B",
                "size": "1.7 GB",
                "description": "Code model trained on The Stack v2",
                "license": "BigCode OpenRAIL-M",
                "category": "starcoder",
            },
            {
                "name": "starcoder2:7b",
                "display_name": "StarCoder2 7B",
                "size": "4.0 GB",
                "description": "Powerful code generation",
                "license": "BigCode OpenRAIL-M",
                "category": "starcoder",
            },
            # === Stable LM Family (Stability AI) ===
            {
                "name": "stablelm2:1.6b",
                "display_name": "StableLM 2 1.6B",
                "size": "984 MB",
                "description": "Stability AI's compact model",
                "license": "Stability AI License",
                "category": "stablelm",
            },
            # === Other Notable Models ===
            {
                "name": "orca-mini:3b",
                "display_name": "Orca Mini 3B",
                "size": "1.9 GB",
                "description": "Reasoning-focused small model",
                "license": "CC-BY-NC-SA-4.0",
                "category": "other",
            },
            {
                "name": "neural-chat:7b",
                "display_name": "Neural Chat 7B",
                "size": "4.1 GB",
                "description": "Intel's conversational model",
                "license": "Apache 2.0",
                "category": "other",
            },
            {
                "name": "openchat:7b",
                "display_name": "OpenChat 7B",
                "size": "4.1 GB",
                "description": "High-quality chat model",
                "license": "Apache 2.0",
                "category": "other",
            },
            {
                "name": "yi:6b",
                "display_name": "Yi 6B",
                "size": "3.5 GB",
                "description": "01.AI's bilingual model",
                "license": "Apache 2.0",
                "category": "other",
            },
            {
                "name": "falcon:7b",
                "display_name": "Falcon 7B",
                "size": "4.2 GB",
                "description": "TII's powerful open model",
                "license": "Apache 2.0",
                "category": "other",
            },
            {
                "name": "zephyr:7b",
                "display_name": "Zephyr 7B",
                "size": "4.1 GB",
                "description": "HuggingFace's instruction-tuned model",
                "license": "MIT",
                "category": "other",
            },
            {
                "name": "vicuna:7b",
                "display_name": "Vicuna 7B",
                "size": "3.8 GB",
                "description": "Fine-tuned LLaMA for chat",
                "license": "Llama 2 Community",
                "category": "other",
            },
            {
                "name": "smollm:1.7b",
                "display_name": "SmolLM 1.7B",
                "size": "1.0 GB",
                "description": "HuggingFace's tiny but capable model",
                "license": "Apache 2.0",
                "category": "other",
            },
        ]

        # Check which models are already installed
        import httpx

        installed = set()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    installed = {m["name"] for m in data.get("models", [])}
        except Exception:
            pass

        # Mark installed models
        for model in models:
            model["installed"] = model["name"] in installed

        return models

    @app.post("/api/slms/ollama/pull")
    async def pull_ollama_model(request: dict):
        """Pull/install an Ollama model."""

        import httpx

        model_name = request.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name required")

        try:
            # Start the pull - this streams progress
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/pull", json={"name": model_name, "stream": False}
                )
                if response.status_code == 200:
                    return {"success": True, "message": f"Successfully pulled {model_name}"}
                else:
                    return {"success": False, "error": response.text}
        except httpx.TimeoutException:
            return {"success": False, "error": "Pull timed out - model may still be downloading"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.delete("/api/slms/ollama/{model_name}")
    async def delete_ollama_model(model_name: str):
        """Delete an Ollama model."""
        # URL decode the model name (handles colons)
        from urllib.parse import unquote

        import httpx

        model_name = unquote(model_name)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete("http://localhost:11434/api/delete", json={"name": model_name})
                if response.status_code == 200:
                    return {"success": True, "message": f"Deleted {model_name}"}
                else:
                    return {"success": False, "error": response.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Include federated learning routes
    try:
        from .federated_api import router as federated_router

        app.include_router(federated_router)
    except ImportError:
        pass  # Federated module not available

    # Include mobile API routes
    try:
        from ..mobile.api import create_mobile_routes

        mobile_router = create_mobile_routes()
        if mobile_router:
            app.include_router(mobile_router)
    except ImportError:
        pass  # Mobile module not available

    return app


def _serialize_memory(memory: MemoryEntry) -> dict[str, Any]:
    """Serialize a memory entry for JSON response."""
    return {
        "id": memory.id,
        "content": memory.content,
        "summary": memory.summary,
        "tier": memory.tier.value if memory.tier else None,
        "memory_type": memory.memory_type.value if memory.memory_type else None,
        "truth_category": memory.truth_category.value if memory.truth_category else None,
        "confidence": memory.confidence,
        "domains": memory.domains,
        "tags": memory.tags,
        "entities": memory.entities,
        "sources": [{"type": s.source_type, "identifier": s.identifier} for s in (memory.sources or [])],
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
        "access_count": memory.access_count,
    }


def get_index_html() -> str:
    """Return the main HTML page from template file."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text()

    # Fallback to minimal inline HTML if template not found
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PLM Memory Manager</title>
    <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        [v-cloak] { display: none; }
        .memory-card { transition: all 0.2s ease; }
        .memory-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .tier-short_term { border-left-color: #f59e0b; }
        .tier-long_term { border-left-color: #3b82f6; }
        .tier-persistent { border-left-color: #10b981; }
        .fade-enter-active, .fade-leave-active { transition: opacity 0.3s ease; }
        .fade-enter-from, .fade-leave-to { opacity: 0; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
    <div id="app" v-cloak>
        <!-- Navigation -->
        <nav class="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
            <div class="max-w-7xl mx-auto px-4">
                <div class="flex items-center justify-between h-16">
                    <div class="flex items-center space-x-4">
                        <h1 class="text-xl font-bold text-white">
                            <i class="fas fa-brain mr-2 text-purple-400"></i>PLM Memory
                        </h1>
                        <div class="flex space-x-1">
                            <button @click="view = 'dashboard'"
                                :class="view === 'dashboard' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-chart-line mr-1"></i> Dashboard
                            </button>
                            <button @click="view = 'memories'"
                                :class="view === 'memories' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-database mr-1"></i> Memories
                            </button>
                            <button @click="view = 'search'"
                                :class="view === 'search' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-search mr-1"></i> Search
                            </button>
                            <button @click="view = 'connections'"
                                :class="view === 'connections' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-plug mr-1"></i> Connections
                            </button>
                            <button @click="view = 'ingest'"
                                :class="view === 'ingest' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-upload mr-1"></i> Ingest
                            </button>
                            <button @click="view = 'context'"
                                :class="view === 'context' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-code mr-1"></i> Context
                            </button>
                            <button @click="view = 'chat'"
                                :class="view === 'chat' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-comments mr-1"></i> Chat
                            </button>
                            <button @click="view = 'compare'"
                                :class="view === 'compare' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'"
                                class="px-4 py-2 rounded-lg text-sm font-medium transition">
                                <i class="fas fa-balance-scale mr-1"></i> Compare
                            </button>
                        </div>
                    </div>
                    <button @click="showAddModal = true" class="bg-green-600 hover:bg-green-500 px-4 py-2 rounded-lg text-sm font-medium transition">
                        <i class="fas fa-plus mr-1"></i> Add Memory
                    </button>
                </div>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto px-4 py-6">
            <!-- Dashboard View -->
            <div v-if="view === 'dashboard'" class="space-y-6">
                <h2 class="text-2xl font-bold">Dashboard</h2>

                <!-- Stats Cards -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                        <div class="flex items-center justify-between">
                            <div>
                                <p class="text-gray-400 text-sm">Total Memories</p>
                                <p class="text-3xl font-bold text-white">{{ stats.total_memories || 0 }}</p>
                            </div>
                            <i class="fas fa-brain text-4xl text-purple-400 opacity-50"></i>
                        </div>
                    </div>
                    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 border-l-4 border-l-amber-500">
                        <div class="flex items-center justify-between">
                            <div>
                                <p class="text-gray-400 text-sm">Short-term</p>
                                <p class="text-3xl font-bold text-amber-400">{{ stats.by_tier?.short_term || 0 }}</p>
                            </div>
                            <i class="fas fa-clock text-4xl text-amber-400 opacity-50"></i>
                        </div>
                    </div>
                    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 border-l-4 border-l-blue-500">
                        <div class="flex items-center justify-between">
                            <div>
                                <p class="text-gray-400 text-sm">Long-term</p>
                                <p class="text-3xl font-bold text-blue-400">{{ stats.by_tier?.long_term || 0 }}</p>
                            </div>
                            <i class="fas fa-archive text-4xl text-blue-400 opacity-50"></i>
                        </div>
                    </div>
                    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 border-l-4 border-l-green-500">
                        <div class="flex items-center justify-between">
                            <div>
                                <p class="text-gray-400 text-sm">Persistent</p>
                                <p class="text-3xl font-bold text-green-400">{{ stats.by_tier?.persistent || 0 }}</p>
                            </div>
                            <i class="fas fa-gem text-4xl text-green-400 opacity-50"></i>
                        </div>
                    </div>
                </div>

                <!-- Charts Row -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- By Type -->
                    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                        <h3 class="text-lg font-semibold mb-4">By Type</h3>
                        <div class="space-y-3">
                            <div v-for="(count, type) in stats.by_type" :key="type" class="flex items-center">
                                <span class="w-24 text-sm text-gray-400">{{ type }}</span>
                                <div class="flex-1 bg-gray-700 rounded-full h-4 mx-3">
                                    <div class="bg-purple-500 h-4 rounded-full"
                                        :style="{width: (count / stats.total_memories * 100) + '%'}"></div>
                                </div>
                                <span class="text-sm font-medium w-12 text-right">{{ count }}</span>
                            </div>
                        </div>
                    </div>

                    <!-- Top Domains -->
                    <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                        <h3 class="text-lg font-semibold mb-4">Top Domains</h3>
                        <div class="space-y-2">
                            <div v-for="domain in (stats.top_domains || []).slice(0, 8)" :key="domain.domain"
                                class="flex items-center justify-between py-2 border-b border-gray-700">
                                <span class="text-gray-300">{{ domain.domain }}</span>
                                <span class="text-purple-400 font-medium">{{ domain.count }}</span>
                            </div>
                            <p v-if="!stats.top_domains?.length" class="text-gray-500 text-center py-4">
                                No domain data available
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Memories View -->
            <div v-if="view === 'memories'" class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Memories</h2>
                    <div class="flex items-center space-x-3">
                        <select v-model="filterTier" class="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm">
                            <option value="">All Tiers</option>
                            <option value="SHORT_TERM">Short-term</option>
                            <option value="LONG_TERM">Long-term</option>
                            <option value="PERSISTENT">Persistent</option>
                        </select>
                        <select v-model="filterType" class="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm">
                            <option value="">All Types</option>
                            <option value="FACT">Fact</option>
                            <option value="BELIEF">Belief</option>
                            <option value="PREFERENCE">Preference</option>
                            <option value="SKILL">Skill</option>
                            <option value="EVENT">Event</option>
                        </select>
                        <button @click="loadMemories" class="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded-lg text-sm">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                    </div>
                </div>

                <!-- Memory Grid -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div v-for="memory in memories" :key="memory.id"
                        class="memory-card bg-gray-800 rounded-xl p-4 border border-gray-700 border-l-4"
                        :class="'tier-' + memory.tier?.toLowerCase()">
                        <div class="flex items-start justify-between mb-2">
                            <span class="text-xs font-medium px-2 py-1 rounded-full"
                                :class="{
                                    'bg-amber-900 text-amber-300': memory.tier === 'SHORT_TERM',
                                    'bg-blue-900 text-blue-300': memory.tier === 'LONG_TERM',
                                    'bg-green-900 text-green-300': memory.tier === 'PERSISTENT'
                                }">
                                {{ memory.tier?.replace('_', ' ') }}
                            </span>
                            <div class="flex space-x-1">
                                <button @click="editMemory(memory)" class="text-gray-400 hover:text-white p-1">
                                    <i class="fas fa-edit text-xs"></i>
                                </button>
                                <button @click="deleteMemory(memory.id)" class="text-gray-400 hover:text-red-400 p-1">
                                    <i class="fas fa-trash text-xs"></i>
                                </button>
                            </div>
                        </div>
                        <p class="text-gray-200 text-sm line-clamp-3 mb-3">{{ memory.content }}</p>
                        <div class="flex items-center justify-between text-xs text-gray-500">
                            <span>{{ memory.memory_type }}</span>
                            <span>{{ (memory.confidence * 100).toFixed(0) }}% confidence</span>
                        </div>
                        <div v-if="memory.domains?.length" class="mt-2 flex flex-wrap gap-1">
                            <span v-for="domain in memory.domains.slice(0, 3)" :key="domain"
                                class="text-xs bg-gray-700 px-2 py-0.5 rounded">{{ domain }}</span>
                        </div>
                    </div>
                </div>

                <div v-if="!memories.length" class="text-center py-12 text-gray-500">
                    <i class="fas fa-inbox text-4xl mb-3"></i>
                    <p>No memories found</p>
                </div>

                <!-- Pagination -->
                <div v-if="memories.length" class="flex justify-center space-x-2">
                    <button @click="loadMemories(offset - limit)" :disabled="offset === 0"
                        class="px-4 py-2 bg-gray-700 rounded-lg disabled:opacity-50">Previous</button>
                    <button @click="loadMemories(offset + limit)"
                        class="px-4 py-2 bg-gray-700 rounded-lg">Next</button>
                </div>
            </div>

            <!-- Search View -->
            <div v-if="view === 'search'" class="space-y-6">
                <h2 class="text-2xl font-bold">Search Memories</h2>

                <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                    <div class="flex space-x-3">
                        <input v-model="searchQuery" @keyup.enter="performSearch"
                            type="text" placeholder="Search your memories..."
                            class="flex-1 bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 text-lg focus:border-purple-500 focus:ring-1 focus:ring-purple-500 outline-none">
                        <button @click="performSearch" :disabled="searching"
                            class="bg-purple-600 hover:bg-purple-500 px-6 py-3 rounded-lg font-medium transition disabled:opacity-50">
                            <i class="fas fa-search mr-2"></i>
                            {{ searching ? 'Searching...' : 'Search' }}
                        </button>
                    </div>
                    <div class="mt-3 flex items-center space-x-4">
                        <label class="flex items-center space-x-2 text-sm text-gray-400">
                            <input type="checkbox" v-model="semanticSearch" class="rounded">
                            <span>Semantic search</span>
                        </label>
                    </div>
                </div>

                <!-- Search Results -->
                <div v-if="searchResults.length" class="space-y-4">
                    <p class="text-gray-400">Found {{ searchResults.length }} results</p>
                    <div v-for="memory in searchResults" :key="memory.id"
                        class="bg-gray-800 rounded-xl p-4 border border-gray-700 border-l-4"
                        :class="'tier-' + memory.tier?.toLowerCase()">
                        <div class="flex items-start justify-between mb-2">
                            <div class="flex items-center space-x-2">
                                <span class="text-xs font-medium px-2 py-1 rounded-full bg-gray-700">
                                    {{ memory.memory_type }}
                                </span>
                                <span class="text-xs text-gray-500">{{ memory.tier?.replace('_', ' ') }}</span>
                            </div>
                            <span class="text-xs text-gray-500">{{ (memory.confidence * 100).toFixed(0) }}%</span>
                        </div>
                        <p class="text-gray-200">{{ memory.content }}</p>
                        <div v-if="memory.domains?.length" class="mt-2 flex flex-wrap gap-1">
                            <span v-for="domain in memory.domains" :key="domain"
                                class="text-xs bg-purple-900 text-purple-300 px-2 py-0.5 rounded">{{ domain }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Connections View -->
            <div v-if="view === 'connections'" class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Data Connections</h2>
                    <button @click="showConnectionModal = true" class="bg-green-600 hover:bg-green-500 px-4 py-2 rounded-lg text-sm font-medium transition">
                        <i class="fas fa-plus mr-1"></i> Add Connection
                    </button>
                </div>

                <!-- Existing Connections -->
                <div v-if="connections.length" class="space-y-4">
                    <div v-for="conn in connections" :key="conn.id"
                        class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                        <div class="flex items-start justify-between">
                            <div class="flex items-center space-x-4">
                                <div class="w-12 h-12 rounded-lg flex items-center justify-center"
                                    :class="{
                                        'bg-green-900': conn.status === 'connected',
                                        'bg-yellow-900': conn.status === 'pending',
                                        'bg-red-900': conn.status === 'error',
                                        'bg-gray-700': conn.status === 'disconnected'
                                    }">
                                    <i :class="getSourceIcon(getSourceCategory(conn.source_id))"
                                        class="text-xl"
                                        :class="{
                                            'text-green-400': conn.status === 'connected',
                                            'text-yellow-400': conn.status === 'pending',
                                            'text-red-400': conn.status === 'error',
                                            'text-gray-400': conn.status === 'disconnected'
                                        }"></i>
                                </div>
                                <div>
                                    <h3 class="font-semibold text-lg">{{ conn.name }}</h3>
                                    <p class="text-sm text-gray-400">{{ conn.source_id }}</p>
                                </div>
                            </div>
                            <div class="flex items-center space-x-2">
                                <span class="px-3 py-1 rounded-full text-xs font-medium"
                                    :class="{
                                        'bg-green-900 text-green-300': conn.status === 'connected',
                                        'bg-yellow-900 text-yellow-300': conn.status === 'pending',
                                        'bg-red-900 text-red-300': conn.status === 'error',
                                        'bg-gray-700 text-gray-300': conn.status === 'disconnected'
                                    }">
                                    {{ conn.status }}
                                </span>
                            </div>
                        </div>
                        <div class="mt-4 flex items-center justify-between">
                            <div class="text-sm text-gray-500">
                                <span v-if="conn.last_sync">Last sync: {{ formatDate(conn.last_sync) }}</span>
                                <span v-else>Never synced</span>
                            </div>
                            <div class="flex space-x-2">
                                <button @click="testConnection(conn.id)"
                                    class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition">
                                    <i class="fas fa-plug mr-1"></i> Test
                                </button>
                                <button @click="syncConnection(conn.id)"
                                    :disabled="conn.status !== 'connected'"
                                    class="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm transition disabled:opacity-50">
                                    <i class="fas fa-sync-alt mr-1"></i> Sync
                                </button>
                                <button @click="editConnection(conn)"
                                    class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition">
                                    <i class="fas fa-edit"></i>
                                </button>
                                <button @click="deleteConnection(conn.id)"
                                    class="px-3 py-1.5 bg-red-600 hover:bg-red-500 rounded-lg text-sm transition">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                        <div v-if="conn.last_error" class="mt-3 p-3 bg-red-900/30 border border-red-800 rounded-lg text-sm text-red-300">
                            <i class="fas fa-exclamation-triangle mr-1"></i> {{ conn.last_error }}
                        </div>
                    </div>
                </div>

                <div v-else class="text-center py-12 text-gray-500">
                    <i class="fas fa-plug text-4xl mb-3"></i>
                    <p>No connections configured</p>
                    <p class="text-sm mt-2">Add a connection to start importing your data</p>
                </div>

                <!-- Available Sources -->
                <div class="mt-8">
                    <h3 class="text-lg font-semibold mb-4">Available Data Sources</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                        <div v-for="source in sourcesConfig" :key="source.id"
                            @click="openAddConnection(source)"
                            class="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-purple-500 transition cursor-pointer text-center">
                            <i :class="getSourceIcon(source.category)" class="text-2xl text-purple-400 mb-2"></i>
                            <p class="text-sm font-medium">{{ source.name }}</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Ingest View -->
            <div v-if="view === 'ingest'" class="space-y-6">
                <h2 class="text-2xl font-bold">Data Ingestion</h2>

                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div v-for="source in sources" :key="source.id"
                        class="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-purple-500 transition cursor-pointer"
                        @click="selectSource(source)">
                        <div class="flex items-center space-x-3 mb-3">
                            <i :class="getSourceIcon(source.category)" class="text-2xl text-purple-400"></i>
                            <div>
                                <h3 class="font-semibold">{{ source.name }}</h3>
                                <p class="text-xs text-gray-500">{{ source.category }}</p>
                            </div>
                        </div>
                        <p class="text-sm text-gray-400">{{ source.description }}</p>
                    </div>
                </div>
            </div>

            <!-- Context View -->
            <div v-if="view === 'context'" class="space-y-6">
                <h2 class="text-2xl font-bold">Context Generator</h2>

                <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-2">Query</label>
                            <input v-model="contextQuery" type="text"
                                placeholder="What context do you need?"
                                class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 focus:border-purple-500 outline-none">
                        </div>
                        <div class="flex space-x-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-400 mb-2">Format</label>
                                <select v-model="contextFormat" class="bg-gray-900 border border-gray-600 rounded-lg px-4 py-2">
                                    <option value="claude">Claude XML</option>
                                    <option value="json">JSON</option>
                                    <option value="markdown">Markdown</option>
                                    <option value="system_prompt">System Prompt</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-400 mb-2">Max Memories</label>
                                <input v-model.number="contextMaxMemories" type="number" min="1" max="50"
                                    class="w-24 bg-gray-900 border border-gray-600 rounded-lg px-4 py-2">
                            </div>
                        </div>
                        <button @click="generateContext" :disabled="!contextQuery || generatingContext"
                            class="bg-purple-600 hover:bg-purple-500 px-6 py-3 rounded-lg font-medium transition disabled:opacity-50">
                            <i class="fas fa-magic mr-2"></i>
                            {{ generatingContext ? 'Generating...' : 'Generate Context' }}
                        </button>
                    </div>
                </div>

                <div v-if="generatedContext" class="bg-gray-800 rounded-xl border border-gray-700">
                    <div class="flex items-center justify-between p-4 border-b border-gray-700">
                        <h3 class="font-semibold">Generated Context</h3>
                        <button @click="copyContext" class="text-gray-400 hover:text-white">
                            <i class="fas fa-copy mr-1"></i> Copy
                        </button>
                    </div>
                    <pre class="p-4 text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">{{ generatedContext }}</pre>
                </div>
            </div>

            <!-- Chat View -->
            <div v-if="view === 'chat'" class="space-y-6">
                <h2 class="text-2xl font-bold">
                    <i class="fas fa-comments text-purple-400 mr-2"></i>Chat with Claude
                </h2>
                <p class="text-gray-400">Ask questions and get responses enhanced with your personal memory context.</p>

                <!-- Chat Container -->
                <div class="bg-gray-800 rounded-xl border border-gray-700 flex flex-col" style="height: 600px;">
                    <!-- Messages -->
                    <div class="flex-1 overflow-y-auto p-4 space-y-4" ref="chatMessages">
                        <div v-if="chatMessages.length === 0" class="text-center text-gray-500 py-8">
                            <i class="fas fa-robot text-4xl mb-4"></i>
                            <p>Start a conversation! Your memories will be used to provide personalized responses.</p>
                        </div>
                        <div v-for="(msg, idx) in chatMessages" :key="idx"
                            :class="msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'">
                            <div :class="msg.role === 'user'
                                ? 'bg-purple-600 text-white rounded-lg px-4 py-2 max-w-[80%]'
                                : 'bg-gray-700 text-gray-100 rounded-lg px-4 py-2 max-w-[80%]'">
                                <div class="whitespace-pre-wrap text-sm">{{ msg.content }}</div>
                                <div v-if="msg.memories_used" class="text-xs text-gray-400 mt-2">
                                    <i class="fas fa-brain mr-1"></i> {{ msg.memories_used }} memories used
                                </div>
                            </div>
                        </div>
                        <div v-if="chatLoading" class="flex justify-start">
                            <div class="bg-gray-700 text-gray-300 rounded-lg px-4 py-2">
                                <i class="fas fa-spinner fa-spin mr-2"></i> Thinking...
                            </div>
                        </div>
                    </div>

                    <!-- Input -->
                    <div class="border-t border-gray-700 p-4">
                        <div class="flex space-x-3">
                            <input v-model="chatInput" type="text"
                                @keyup.enter="sendMessage"
                                placeholder="Type your message..."
                                class="flex-1 bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 focus:border-purple-500 outline-none">
                            <button @click="sendMessage" :disabled="!chatInput || chatLoading"
                                class="bg-purple-600 hover:bg-purple-500 px-6 py-3 rounded-lg font-medium transition disabled:opacity-50">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                        <div class="flex items-center justify-between mt-2">
                            <label class="flex items-center text-sm text-gray-400">
                                <input type="checkbox" v-model="chatIncludeIdentity" class="mr-2">
                                Include identity context
                            </label>
                            <span class="text-sm text-gray-500">
                                Max memories:
                                <input type="number" v-model.number="chatMaxMemories" min="1" max="20"
                                    class="w-12 bg-gray-900 border border-gray-600 rounded px-2 py-1 text-center">
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Context Preview -->
                <div v-if="chatContextPreview" class="bg-gray-800 rounded-xl p-4 border border-gray-700">
                    <div class="flex items-center justify-between mb-2">
                        <h3 class="font-semibold text-sm text-gray-400">
                            <i class="fas fa-eye mr-1"></i> Last Context Preview
                        </h3>
                        <button @click="chatContextPreview = null" class="text-gray-500 hover:text-white">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <pre class="text-xs text-gray-400 overflow-x-auto">{{ chatContextPreview }}</pre>
                </div>
            </div>

            <!-- Compare View -->
            <div v-if="view === 'compare'" class="space-y-6">
                <div class="flex items-center justify-between">
                    <div>
                        <h2 class="text-2xl font-bold">
                            <i class="fas fa-balance-scale text-purple-400 mr-2"></i>LLM Comparison
                        </h2>
                        <p class="text-gray-400 mt-1">
                            Compare responses from multiple LLMs with and without your personal context.
                            Prove the value of your PLM!
                        </p>
                    </div>
                    <button @click="loadProviders" class="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded-lg text-sm">
                        <i class="fas fa-sync-alt mr-1"></i> Refresh Providers
                    </button>
                </div>

                <!-- Provider Status -->
                <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
                    <h3 class="font-semibold mb-3">Available Providers</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div v-for="provider in compareProviders" :key="provider.id"
                            :class="provider.available ? 'border-green-500 bg-green-900/20' : 'border-red-500 bg-red-900/20'"
                            class="border rounded-lg p-3">
                            <div class="flex items-center justify-between">
                                <span class="font-medium">{{ provider.name }}</span>
                                <i :class="provider.available ? 'fas fa-check-circle text-green-400' : 'fas fa-times-circle text-red-400'"></i>
                            </div>
                            <div class="text-xs text-gray-400 mt-1">{{ provider.model }}</div>
                            <div v-if="!provider.available" class="text-xs text-red-400 mt-1">
                                {{ provider.config_help }}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Query Input -->
                <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-2">
                                Test Query (ask about your personal data)
                            </label>
                            <textarea v-model="compareQuery" rows="2"
                                placeholder="e.g., What S3 buckets do I have? What GitHub repos am I working on?"
                                class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 focus:border-purple-500 outline-none"></textarea>
                        </div>
                        <div class="flex items-center justify-between">
                            <div class="flex items-center space-x-4">
                                <label class="text-sm text-gray-400">
                                    Max Memories:
                                    <input type="number" v-model.number="compareMaxMemories" min="1" max="20"
                                        class="w-16 ml-2 bg-gray-900 border border-gray-600 rounded px-2 py-1">
                                </label>
                            </div>
                            <button @click="runComparison" :disabled="!compareQuery || comparing"
                                class="bg-purple-600 hover:bg-purple-500 px-6 py-3 rounded-lg font-medium transition disabled:opacity-50">
                                <i class="fas fa-play mr-2"></i>
                                {{ comparing ? 'Running Comparison...' : 'Run Comparison' }}
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Results -->
                <div v-if="compareResults" class="space-y-6">
                    <!-- Context Preview -->
                    <div class="bg-gray-800 rounded-xl p-4 border border-gray-700">
                        <div class="flex items-center justify-between mb-2">
                            <h3 class="font-semibold">
                                <i class="fas fa-brain text-purple-400 mr-2"></i>
                                PLM Context Used ({{ compareResults.context_length }} chars)
                            </h3>
                            <button @click="showFullContext = !showFullContext" class="text-sm text-purple-400 hover:text-purple-300">
                                {{ showFullContext ? 'Collapse' : 'Expand' }}
                            </button>
                        </div>
                        <pre class="text-xs text-gray-400 overflow-x-auto whitespace-pre-wrap"
                            :class="showFullContext ? '' : 'max-h-24 overflow-y-hidden'">{{ compareResults.context_preview }}</pre>
                    </div>

                    <!-- Comparison Grid -->
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <!-- With Context Column -->
                        <div class="space-y-4">
                            <h3 class="text-lg font-bold text-green-400">
                                <i class="fas fa-check-circle mr-2"></i>WITH PLM Context
                            </h3>
                            <div v-for="(result, provider) in compareResults.with_context" :key="'with-' + provider"
                                class="bg-gray-800 rounded-xl border border-green-500/30 overflow-hidden">
                                <div class="bg-green-900/20 px-4 py-2 flex items-center justify-between border-b border-green-500/30">
                                    <span class="font-medium">{{ getProviderName(provider) }}</span>
                                    <span class="text-xs text-gray-400">{{ result.latency_ms }}ms</span>
                                </div>
                                <div class="p-4">
                                    <div v-if="result.error" class="text-red-400 text-sm">
                                        <i class="fas fa-exclamation-triangle mr-1"></i> {{ result.error }}
                                    </div>
                                    <div v-else class="text-sm text-gray-200 whitespace-pre-wrap max-h-64 overflow-y-auto">
                                        {{ result.response || 'No response' }}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Without Context Column -->
                        <div class="space-y-4">
                            <h3 class="text-lg font-bold text-amber-400">
                                <i class="fas fa-times-circle mr-2"></i>WITHOUT Context (Baseline)
                            </h3>
                            <div v-for="(result, provider) in compareResults.without_context" :key="'without-' + provider"
                                class="bg-gray-800 rounded-xl border border-amber-500/30 overflow-hidden">
                                <div class="bg-amber-900/20 px-4 py-2 flex items-center justify-between border-b border-amber-500/30">
                                    <span class="font-medium">{{ getProviderName(provider) }}</span>
                                    <span class="text-xs text-gray-400">{{ result.latency_ms }}ms</span>
                                </div>
                                <div class="p-4">
                                    <div v-if="result.error" class="text-red-400 text-sm">
                                        <i class="fas fa-exclamation-triangle mr-1"></i> {{ result.error }}
                                    </div>
                                    <div v-else class="text-sm text-gray-200 whitespace-pre-wrap max-h-64 overflow-y-auto">
                                        {{ result.response || 'No response' }}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Empty State -->
                <div v-if="!compareResults && !comparing" class="text-center py-12 text-gray-500">
                    <i class="fas fa-vial text-4xl mb-4"></i>
                    <p>Enter a query and click "Run Comparison" to see the difference PLM context makes.</p>
                    <p class="mt-2 text-sm">Try asking about your personal data like S3 buckets, GitHub repos, or browsing history.</p>
                </div>
            </div>
        </main>

        <!-- Add/Edit Memory Modal -->
        <div v-if="showAddModal || showEditModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-gray-800 rounded-xl p-6 w-full max-w-lg mx-4 border border-gray-700">
                <h3 class="text-xl font-bold mb-4">{{ showEditModal ? 'Edit Memory' : 'Add Memory' }}</h3>
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">Content</label>
                        <textarea v-model="memoryForm.content" rows="4"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-3 focus:border-purple-500 outline-none"
                            placeholder="What do you want to remember?"></textarea>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-2">Type</label>
                            <select v-model="memoryForm.memory_type" class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2">
                                <option value="fact">Fact</option>
                                <option value="belief">Belief</option>
                                <option value="preference">Preference</option>
                                <option value="skill">Skill</option>
                                <option value="event">Event</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-400 mb-2">Truth Category</label>
                            <select v-model="memoryForm.truth_category" class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2">
                                <option value="absolute">Absolute</option>
                                <option value="contextual">Contextual</option>
                                <option value="opinion">Opinion</option>
                                <option value="inferred">Inferred</option>
                            </select>
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">Confidence: {{ (memoryForm.confidence * 100).toFixed(0) }}%</label>
                        <input type="range" v-model.number="memoryForm.confidence" min="0" max="1" step="0.05" class="w-full">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">Domains (comma-separated)</label>
                        <input v-model="memoryForm.domainsStr" type="text"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2"
                            placeholder="e.g., python, web-development">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">Tags (comma-separated)</label>
                        <input v-model="memoryForm.tagsStr" type="text"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2"
                            placeholder="e.g., important, work">
                    </div>
                </div>
                <div class="flex justify-end space-x-3 mt-6">
                    <button @click="closeModal" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg">Cancel</button>
                    <button @click="saveMemory" :disabled="!memoryForm.content"
                        class="px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg disabled:opacity-50">
                        {{ showEditModal ? 'Update' : 'Save' }}
                    </button>
                </div>
            </div>
        </div>

        <!-- Add/Edit Connection Modal -->
        <div v-if="showConnectionModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-gray-800 rounded-xl p-6 w-full max-w-lg mx-4 border border-gray-700 max-h-[90vh] overflow-y-auto">
                <h3 class="text-xl font-bold mb-4">
                    {{ editingConnection ? 'Edit Connection' : 'Add Connection' }}
                </h3>

                <!-- Source Selection (only for new connections) -->
                <div v-if="!editingConnection && !selectedSourceConfig" class="space-y-4">
                    <p class="text-gray-400">Select a data source:</p>
                    <div class="grid grid-cols-2 gap-3 max-h-64 overflow-y-auto">
                        <div v-for="source in sourcesConfig" :key="source.id"
                            @click="selectedSourceConfig = source"
                            class="bg-gray-700 rounded-lg p-3 border border-gray-600 hover:border-purple-500 transition cursor-pointer">
                            <i :class="getSourceIcon(source.category)" class="text-xl text-purple-400 mr-2"></i>
                            <span class="text-sm">{{ source.name }}</span>
                        </div>
                    </div>
                </div>

                <!-- Connection Form -->
                <div v-else class="space-y-4">
                    <div class="flex items-center space-x-3 mb-4 pb-4 border-b border-gray-700">
                        <i :class="getSourceIcon(selectedSourceConfig?.category)" class="text-2xl text-purple-400"></i>
                        <div>
                            <h4 class="font-semibold">{{ selectedSourceConfig?.name }}</h4>
                            <p class="text-sm text-gray-400">{{ selectedSourceConfig?.auth_description }}</p>
                        </div>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">Connection Name</label>
                        <input v-model="connectionForm.name" type="text"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2 focus:border-purple-500 outline-none"
                            :placeholder="'My ' + (selectedSourceConfig?.name || 'Connection')">
                    </div>

                    <!-- Dynamic Credential Fields -->
                    <div v-for="field in (selectedSourceConfig?.auth_fields || [])" :key="field.name">
                        <label class="block text-sm font-medium text-gray-400 mb-2">{{ field.label }}</label>

                        <input v-if="field.type === 'text'"
                            v-model="connectionForm.credentials[field.name]"
                            type="text"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2 focus:border-purple-500 outline-none"
                            :placeholder="field.default || ''">

                        <input v-else-if="field.type === 'password'"
                            v-model="connectionForm.credentials[field.name]"
                            type="password"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2 focus:border-purple-500 outline-none">

                        <div v-else-if="field.type === 'file' || field.type === 'directory'" class="flex space-x-2">
                            <input v-model="connectionForm.credentials[field.name]"
                                type="text"
                                class="flex-1 bg-gray-900 border border-gray-600 rounded-lg px-4 py-2 focus:border-purple-500 outline-none"
                                :placeholder="field.default || 'Enter path...'">
                        </div>

                        <select v-else-if="field.type === 'select'"
                            v-model="connectionForm.credentials[field.name]"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2">
                            <option v-for="opt in field.options" :key="opt" :value="opt">{{ opt }}</option>
                        </select>
                    </div>

                    <!-- Sync Settings -->
                    <div class="pt-4 border-t border-gray-700">
                        <h4 class="font-medium mb-3">Sync Settings</h4>
                        <div class="flex items-center justify-between">
                            <label class="text-sm text-gray-400">Enable automatic sync</label>
                            <input type="checkbox" v-model="connectionForm.sync_enabled" class="rounded">
                        </div>
                        <div v-if="connectionForm.sync_enabled" class="mt-3">
                            <label class="block text-sm text-gray-400 mb-2">Sync interval (hours)</label>
                            <input v-model.number="connectionForm.sync_interval_hours" type="number" min="1" max="168"
                                class="w-24 bg-gray-900 border border-gray-600 rounded-lg px-3 py-1">
                        </div>
                    </div>
                </div>

                <div class="flex justify-end space-x-3 mt-6">
                    <button @click="closeConnectionModal" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg">Cancel</button>
                    <button v-if="!editingConnection && !selectedSourceConfig"
                        disabled
                        class="px-4 py-2 bg-purple-600 rounded-lg opacity-50">
                        Select a source
                    </button>
                    <button v-else @click="saveConnection"
                        :disabled="!connectionForm.name"
                        class="px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg disabled:opacity-50">
                        {{ editingConnection ? 'Update' : 'Create' }}
                    </button>
                </div>
            </div>
        </div>

        <!-- Ingest Modal -->
        <div v-if="showIngestModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div class="bg-gray-800 rounded-xl p-6 w-full max-w-lg mx-4 border border-gray-700">
                <div class="flex items-center space-x-3 mb-6">
                    <i :class="getSourceIcon(selectedIngestSource?.category)" class="text-3xl text-purple-400"></i>
                    <div>
                        <h3 class="text-xl font-bold">Ingest from {{ selectedIngestSource?.name }}</h3>
                        <p class="text-sm text-gray-400">{{ selectedIngestSource?.description }}</p>
                    </div>
                </div>

                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">Path (optional)</label>
                        <input v-model="ingestForm.path" type="text"
                            class="w-full bg-gray-900 border border-gray-600 rounded-lg px-4 py-2 focus:border-purple-500 outline-none"
                            placeholder="Enter path to data source...">
                        <p class="text-xs text-gray-500 mt-1">Leave empty for default location</p>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">Limit (max items)</label>
                        <input v-model.number="ingestForm.limit" type="number" min="1" max="10000"
                            class="w-32 bg-gray-900 border border-gray-600 rounded-lg px-4 py-2 focus:border-purple-500 outline-none">
                    </div>

                    <div class="bg-gray-700/50 rounded-lg p-3 text-sm text-gray-300">
                        <i class="fas fa-info-circle mr-2 text-blue-400"></i>
                        Data types: {{ selectedIngestSource?.data_types?.join(', ') || 'Various' }}
                    </div>
                </div>

                <div class="flex justify-end space-x-3 mt-6">
                    <button @click="closeIngestModal" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg">Cancel</button>
                    <button @click="triggerIngest"
                        :disabled="ingesting"
                        class="px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg disabled:opacity-50">
                        <span v-if="ingesting"><i class="fas fa-spinner fa-spin mr-2"></i>Ingesting...</span>
                        <span v-else><i class="fas fa-download mr-2"></i>Start Ingestion</span>
                    </button>
                </div>
            </div>
        </div>

        <!-- Toast Notifications -->
        <div class="fixed bottom-4 right-4 space-y-2">
            <transition-group name="fade">
                <div v-for="toast in toasts" :key="toast.id"
                    :class="{
                        'bg-green-600': toast.type === 'success',
                        'bg-red-600': toast.type === 'error',
                        'bg-blue-600': toast.type === 'info'
                    }"
                    class="px-4 py-3 rounded-lg shadow-lg text-white">
                    {{ toast.message }}
                </div>
            </transition-group>
        </div>
    </div>

    <script>
    const { createApp, ref, reactive, onMounted, watch } = Vue;

    createApp({
        setup() {
            // State
            const view = ref('dashboard');
            const stats = ref({});
            const memories = ref([]);
            const searchResults = ref([]);
            const sources = ref([]);
            const toasts = ref([]);

            // Filters
            const filterTier = ref('');
            const filterType = ref('');
            const offset = ref(0);
            const limit = ref(50);

            // Search
            const searchQuery = ref('');
            const semanticSearch = ref(true);
            const searching = ref(false);

            // Context
            const contextQuery = ref('');
            const contextFormat = ref('claude');
            const contextMaxMemories = ref(10);
            const generatedContext = ref('');
            const generatingContext = ref(false);

            // Modal
            const showAddModal = ref(false);
            const showEditModal = ref(false);
            const editingId = ref(null);
            const memoryForm = reactive({
                content: '',
                memory_type: 'fact',
                truth_category: 'contextual',
                confidence: 0.7,
                domainsStr: '',
                tagsStr: ''
            });

            // Connections
            const connections = ref([]);
            const sourcesConfig = ref([]);
            const showConnectionModal = ref(false);
            const editingConnection = ref(null);
            const selectedSourceConfig = ref(null);
            const connectionForm = reactive({
                name: '',
                credentials: {},
                sync_enabled: true,
                sync_interval_hours: 24
            });

            // Ingest
            const showIngestModal = ref(false);
            const selectedIngestSource = ref(null);
            const ingesting = ref(false);
            const ingestForm = reactive({
                path: '',
                limit: 100
            });

            // Chat
            const chatMessages = ref([]);
            const chatInput = ref('');
            const chatLoading = ref(false);
            const chatIncludeIdentity = ref(true);
            const chatMaxMemories = ref(10);
            const chatContextPreview = ref(null);

            // Compare
            const compareProviders = ref([]);
            const compareQuery = ref('');
            const compareMaxMemories = ref(10);
            const comparing = ref(false);
            const compareResults = ref(null);
            const showFullContext = ref(false);

            // Methods
            const showToast = (message, type = 'info') => {
                const id = Date.now();
                toasts.value.push({ id, message, type });
                setTimeout(() => {
                    toasts.value = toasts.value.filter(t => t.id !== id);
                }, 3000);
            };

            const loadStats = async () => {
                try {
                    const res = await fetch('/api/stats');
                    stats.value = await res.json();
                } catch (e) {
                    console.error('Failed to load stats:', e);
                }
            };

            const loadMemories = async (newOffset = 0) => {
                try {
                    offset.value = Math.max(0, newOffset);
                    let url = `/api/memories?limit=${limit.value}&offset=${offset.value}`;
                    if (filterTier.value) url += `&tier=${filterTier.value}`;
                    if (filterType.value) url += `&memory_type=${filterType.value}`;
                    const res = await fetch(url);
                    const data = await res.json();
                    memories.value = data.memories;
                } catch (e) {
                    showToast('Failed to load memories', 'error');
                }
            };

            const loadSources = async () => {
                try {
                    const res = await fetch('/api/sources');
                    const data = await res.json();
                    sources.value = data.sources;
                } catch (e) {
                    console.error('Failed to load sources:', e);
                }
            };

            const loadConnections = async () => {
                try {
                    const res = await fetch('/api/connections');
                    const data = await res.json();
                    connections.value = data.connections;
                } catch (e) {
                    console.error('Failed to load connections:', e);
                }
            };

            const loadSourcesConfig = async () => {
                try {
                    const res = await fetch('/api/sources/config');
                    const data = await res.json();
                    sourcesConfig.value = data.sources;
                } catch (e) {
                    console.error('Failed to load sources config:', e);
                }
            };

            const openAddConnection = (source) => {
                selectedSourceConfig.value = source;
                connectionForm.name = '';
                connectionForm.credentials = {};
                connectionForm.sync_enabled = true;
                connectionForm.sync_interval_hours = 24;

                // Pre-fill default values
                if (source?.auth_fields) {
                    source.auth_fields.forEach(field => {
                        if (field.default) {
                            connectionForm.credentials[field.name] = field.default;
                        }
                    });
                }

                showConnectionModal.value = true;
            };

            const editConnection = (conn) => {
                editingConnection.value = conn;
                selectedSourceConfig.value = sourcesConfig.value.find(s => s.id === conn.source_id);
                connectionForm.name = conn.name;
                connectionForm.credentials = { ...conn.credentials };
                connectionForm.sync_enabled = conn.sync_enabled;
                connectionForm.sync_interval_hours = conn.sync_interval_hours;
                showConnectionModal.value = true;
            };

            const closeConnectionModal = () => {
                showConnectionModal.value = false;
                editingConnection.value = null;
                selectedSourceConfig.value = null;
                connectionForm.name = '';
                connectionForm.credentials = {};
            };

            const saveConnection = async () => {
                try {
                    const payload = {
                        source_id: selectedSourceConfig.value.id,
                        name: connectionForm.name,
                        credentials: connectionForm.credentials,
                        settings: {
                            sync_enabled: connectionForm.sync_enabled,
                            sync_interval_hours: connectionForm.sync_interval_hours
                        }
                    };

                    let res;
                    if (editingConnection.value) {
                        res = await fetch(`/api/connections/${editingConnection.value.id}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                name: connectionForm.name,
                                credentials: connectionForm.credentials,
                                sync_enabled: connectionForm.sync_enabled,
                                sync_interval_hours: connectionForm.sync_interval_hours
                            })
                        });
                    } else {
                        res = await fetch('/api/connections', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                    }

                    if (res.ok) {
                        showToast(editingConnection.value ? 'Connection updated' : 'Connection created', 'success');
                        closeConnectionModal();
                        loadConnections();
                    } else {
                        const err = await res.json();
                        showToast(err.detail || 'Failed to save connection', 'error');
                    }
                } catch (e) {
                    showToast('Failed to save connection', 'error');
                }
            };

            const testConnection = async (connId) => {
                try {
                    showToast('Testing connection...', 'info');
                    const res = await fetch(`/api/connections/${connId}/test`, { method: 'POST' });
                    const data = await res.json();
                    if (data.success) {
                        showToast('Connection successful!', 'success');
                    } else {
                        showToast(`Connection failed: ${data.message}`, 'error');
                    }
                    loadConnections();
                } catch (e) {
                    showToast('Failed to test connection', 'error');
                }
            };

            const syncConnection = async (connId) => {
                try {
                    showToast('Syncing...', 'info');
                    const res = await fetch(`/api/connections/${connId}/sync`, { method: 'POST' });
                    if (res.ok) {
                        const data = await res.json();
                        showToast(`Sync complete! ${data.memories_created} memories created.`, 'success');
                        loadConnections();
                        loadStats();
                    } else {
                        const err = await res.json();
                        showToast(`Sync failed: ${err.detail}`, 'error');
                    }
                } catch (e) {
                    showToast('Sync failed', 'error');
                }
            };

            const deleteConnection = async (connId) => {
                if (!confirm('Delete this connection?')) return;
                try {
                    const res = await fetch(`/api/connections/${connId}`, { method: 'DELETE' });
                    if (res.ok) {
                        showToast('Connection deleted', 'success');
                        loadConnections();
                    }
                } catch (e) {
                    showToast('Failed to delete connection', 'error');
                }
            };

            const getSourceCategory = (sourceId) => {
                const source = sourcesConfig.value.find(s => s.id === sourceId);
                return source?.category || 'unknown';
            };

            const formatDate = (dateStr) => {
                if (!dateStr) return 'Never';
                const date = new Date(dateStr);
                return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            };

            const performSearch = async () => {
                if (!searchQuery.value.trim()) return;
                searching.value = true;
                try {
                    const res = await fetch('/api/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: searchQuery.value,
                            limit: 50,
                            semantic: semanticSearch.value
                        })
                    });
                    const data = await res.json();
                    searchResults.value = data.memories;
                } catch (e) {
                    showToast('Search failed', 'error');
                } finally {
                    searching.value = false;
                }
            };

            const generateContext = async () => {
                if (!contextQuery.value.trim()) return;
                generatingContext.value = true;
                try {
                    const res = await fetch(`/api/context?query=${encodeURIComponent(contextQuery.value)}&format=${contextFormat.value}&max_memories=${contextMaxMemories.value}`);
                    const data = await res.json();
                    generatedContext.value = data.context;
                } catch (e) {
                    showToast('Failed to generate context', 'error');
                } finally {
                    generatingContext.value = false;
                }
            };

            const copyContext = () => {
                navigator.clipboard.writeText(generatedContext.value);
                showToast('Copied to clipboard', 'success');
            };

            const sendMessage = async () => {
                if (!chatInput.value.trim() || chatLoading.value) return;

                const userMessage = chatInput.value.trim();
                chatInput.value = '';

                // Add user message to chat
                chatMessages.value.push({ role: 'user', content: userMessage });

                chatLoading.value = true;
                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: userMessage,
                            max_memories: chatMaxMemories.value,
                            include_identity: chatIncludeIdentity.value
                        })
                    });

                    const data = await res.json();

                    if (res.ok) {
                        chatMessages.value.push({
                            role: 'assistant',
                            content: data.response,
                            memories_used: data.memories_used
                        });
                        if (data.context_preview) {
                            chatContextPreview.value = data.context_preview;
                        }
                    } else {
                        chatMessages.value.push({
                            role: 'assistant',
                            content: 'Error: ' + (data.detail || 'Failed to get response')
                        });
                    }
                } catch (e) {
                    chatMessages.value.push({
                        role: 'assistant',
                        content: 'Error: Failed to connect to chat service'
                    });
                } finally {
                    chatLoading.value = false;
                }
            };

            // Compare Methods
            const loadProviders = async () => {
                try {
                    const res = await fetch('/api/compare/providers');
                    const data = await res.json();
                    compareProviders.value = data.providers;
                } catch (e) {
                    console.error('Failed to load providers:', e);
                    showToast('Failed to load LLM providers', 'error');
                }
            };

            const runComparison = async () => {
                if (!compareQuery.value.trim() || comparing.value) return;

                comparing.value = true;
                compareResults.value = null;

                try {
                    const res = await fetch('/api/compare', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: compareQuery.value,
                            max_memories: compareMaxMemories.value
                        })
                    });

                    if (res.ok) {
                        compareResults.value = await res.json();
                        showToast('Comparison complete!', 'success');
                    } else {
                        const err = await res.json();
                        showToast('Comparison failed: ' + (err.detail || 'Unknown error'), 'error');
                    }
                } catch (e) {
                    showToast('Failed to run comparison', 'error');
                    console.error('Comparison error:', e);
                } finally {
                    comparing.value = false;
                }
            };

            const getProviderName = (providerId) => {
                const names = {
                    'claude': 'Claude (Anthropic)',
                    'chatgpt': 'ChatGPT (OpenAI)',
                    'copilot': 'GitHub Copilot',
                    'amazonq': 'Amazon Q'
                };
                return names[providerId] || providerId;
            };

            const saveMemory = async () => {
                try {
                    const payload = {
                        content: memoryForm.content,
                        memory_type: memoryForm.memory_type,
                        truth_category: memoryForm.truth_category,
                        confidence: memoryForm.confidence,
                        domains: memoryForm.domainsStr ? memoryForm.domainsStr.split(',').map(d => d.trim()).filter(Boolean) : [],
                        tags: memoryForm.tagsStr ? memoryForm.tagsStr.split(',').map(t => t.trim()).filter(Boolean) : []
                    };

                    let res;
                    if (showEditModal.value && editingId.value) {
                        res = await fetch(`/api/memories/${editingId.value}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                    } else {
                        res = await fetch('/api/memories', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                    }

                    if (res.ok) {
                        showToast(showEditModal.value ? 'Memory updated' : 'Memory created', 'success');
                        closeModal();
                        loadMemories();
                        loadStats();
                    }
                } catch (e) {
                    showToast('Failed to save memory', 'error');
                }
            };

            const editMemory = (memory) => {
                editingId.value = memory.id;
                memoryForm.content = memory.content;
                memoryForm.memory_type = memory.memory_type?.toLowerCase() || 'fact';
                memoryForm.truth_category = memory.truth_category?.toLowerCase() || 'contextual';
                memoryForm.confidence = memory.confidence || 0.7;
                memoryForm.domainsStr = (memory.domains || []).join(', ');
                memoryForm.tagsStr = (memory.tags || []).join(', ');
                showEditModal.value = true;
            };

            const deleteMemory = async (id) => {
                if (!confirm('Delete this memory?')) return;
                try {
                    const res = await fetch(`/api/memories/${id}`, { method: 'DELETE' });
                    if (res.ok) {
                        showToast('Memory deleted', 'success');
                        loadMemories();
                        loadStats();
                    }
                } catch (e) {
                    showToast('Failed to delete memory', 'error');
                }
            };

            const closeModal = () => {
                showAddModal.value = false;
                showEditModal.value = false;
                editingId.value = null;
                Object.assign(memoryForm, {
                    content: '',
                    memory_type: 'fact',
                    truth_category: 'contextual',
                    confidence: 0.7,
                    domainsStr: '',
                    tagsStr: ''
                });
            };

            const selectSource = (source) => {
                selectedIngestSource.value = source;
                ingestForm.path = '';
                ingestForm.limit = 100;
                showIngestModal.value = true;
            };

            const closeIngestModal = () => {
                showIngestModal.value = false;
                selectedIngestSource.value = null;
            };

            const triggerIngest = async () => {
                if (!selectedIngestSource.value) return;

                ingesting.value = true;
                try {
                    const payload = {
                        source: selectedIngestSource.value.id,
                        path: ingestForm.path || null,
                        options: { limit: ingestForm.limit }
                    };

                    const res = await fetch('/api/ingest', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (res.ok) {
                        const data = await res.json();
                        showToast(`Ingestion complete! ${data.memories_created} memories created.`, 'success');
                        closeIngestModal();
                        loadStats();
                    } else {
                        const err = await res.json();
                        showToast(`Ingestion failed: ${err.detail}`, 'error');
                    }
                } catch (e) {
                    showToast('Ingestion failed', 'error');
                } finally {
                    ingesting.value = false;
                }
            };

            const getSourceIcon = (category) => {
                const icons = {
                    'social_media': 'fas fa-share-alt',
                    'communication': 'fas fa-comments',
                    'documents': 'fas fa-file-alt',
                    'media': 'fas fa-photo-video',
                    'code': 'fas fa-code',
                    'ai_history': 'fas fa-robot'
                };
                return icons[category] || 'fas fa-database';
            };

            // Watchers
            watch([filterTier, filterType], () => loadMemories(0));

            // Init
            onMounted(() => {
                loadStats();
                loadMemories();
                loadSources();
                loadConnections();
                loadSourcesConfig();
                loadProviders();
            });

            return {
                view, stats, memories, searchResults, sources, toasts,
                filterTier, filterType, offset, limit,
                searchQuery, semanticSearch, searching, performSearch,
                contextQuery, contextFormat, contextMaxMemories, generatedContext, generatingContext,
                generateContext, copyContext,
                showAddModal, showEditModal, memoryForm,
                saveMemory, editMemory, deleteMemory, closeModal,
                selectSource, getSourceIcon, loadMemories, showToast,
                // Connections
                connections, sourcesConfig, showConnectionModal, editingConnection, selectedSourceConfig, connectionForm,
                openAddConnection, editConnection, closeConnectionModal, saveConnection,
                testConnection, syncConnection, deleteConnection, getSourceCategory, formatDate,
                // Ingest
                showIngestModal, selectedIngestSource, ingesting, ingestForm,
                closeIngestModal, triggerIngest,
                // Chat
                chatMessages, chatInput, chatLoading, chatIncludeIdentity, chatMaxMemories, chatContextPreview,
                sendMessage,
                // Compare
                compareProviders, compareQuery, compareMaxMemories, comparing, compareResults, showFullContext,
                loadProviders, runComparison, getProviderName
            };
        }
    }).mount('#app');
    </script>
</body>
</html>"""


# Create module-level app instance for uvicorn import
app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the web server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
