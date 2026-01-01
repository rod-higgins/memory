"""
Identity profile schema - the 'who' of the memory system.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class IdentityProfile(BaseModel):
    """
    Core identity information representing the user.

    This is stored in the persistent tier and provides the foundation
    for all personalized memory interactions.
    """

    id: UUID = Field(default_factory=uuid4)

    # Personal
    name: str = ""
    email: str = ""
    emails: list[str] = Field(default_factory=list)  # All email addresses
    github_handles: list[str] = Field(default_factory=list)

    # Social Media
    social_handles: dict[str, str] = Field(default_factory=dict)
    # e.g., {"twitter": "@username", "facebook": "user.name", "linkedin": "user-name"}

    # Cloud Services
    cloud_services: dict[str, dict] = Field(default_factory=dict)
    # e.g., {"aws": {"profile": "default"}, "gcp": {"project": "my-project"}}

    # Local Data Paths
    local_data_paths: dict[str, str] = Field(default_factory=dict)
    # e.g., {"documents": "~/Documents", "downloads": "~/Downloads"}

    # Professional
    roles: list[str] = Field(default_factory=list)  # e.g., ["Developer", "Architect"]
    organizations: list[str] = Field(default_factory=list)  # e.g., ["Company", "Organization"]

    # Technical Profile
    primary_languages: list[str] = Field(default_factory=list)  # e.g., ["PHP", "Python"]
    frameworks: list[str] = Field(default_factory=list)  # e.g., ["Drupal", "NextJS"]
    tools: list[str] = Field(default_factory=list)  # e.g., ["VSCode", "Docker"]

    # Links to preference and belief memories (by UUID)
    preference_ids: list[UUID] = Field(default_factory=list)
    belief_ids: list[UUID] = Field(default_factory=list)

    # Active projects
    active_projects: list[str] = Field(default_factory=list)

    # Communication preferences
    communication_style: dict[str, str] = Field(default_factory=dict)
    # e.g., {"verbosity": "concise", "formality": "casual", "examples": "prefer"}

    # Temporal
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": False}

    def to_context_string(self) -> str:
        """Format identity for LLM context injection."""
        lines = [
            f"Name: {self.name}",
            f"Primary Languages: {', '.join(self.primary_languages[:5])}",
            f"Frameworks: {', '.join(self.frameworks[:5])}",
        ]

        if self.roles:
            lines.append(f"Roles: {', '.join(self.roles[:3])}")

        if self.active_projects:
            lines.append(f"Active Projects: {', '.join(self.active_projects[:5])}")

        return "\n".join(lines)

    @classmethod
    def create_default(cls) -> IdentityProfile:
        """Create an empty default identity profile."""
        return cls(
            name="",
            email="",
            emails=[],
            github_handles=[],
            social_handles={},
            cloud_services={},
            local_data_paths={},
            roles=[],
            organizations=[],
            primary_languages=[],
            frameworks=[],
            tools=[],
            active_projects=[],
            communication_style={},
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> IdentityProfile:
        """Load identity profile from a JSON config file."""
        import json
        from pathlib import Path

        path = Path(config_path).expanduser()
        if not path.exists():
            return cls.create_default()

        with open(path) as f:
            data = json.load(f)

        return cls(
            name=data.get("owner", ""),
            email=data.get("email", {}).get("gmail", [{}])[0].get("address", ""),
            emails=[e.get("address", "") for e in data.get("email", {}).get("gmail", [])],
            github_handles=[a.get("username", "") for a in data.get("code", {}).get("github", {}).get("accounts", [])],
            social_handles={k: v.get("handle", v.get("username", "")) for k, v in data.get("social_media", {}).items()},
            cloud_services=data.get("cloud_services", {}),
            local_data_paths={
                p.get("name", ""): p.get("path", "") for p in data.get("local_documents", {}).get("paths", [])
            },
            roles=[data.get("work", {}).get("role", "")] if data.get("work", {}).get("role") else [],
            organizations=[
                data.get("work", {}).get("employer", ""),
                data.get("work", {}).get("company", ""),
            ],
        )
