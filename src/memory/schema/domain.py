"""
Domain context schema - represents knowledge domains.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DomainContext(BaseModel):
    """
    Represents a knowledge domain with associated memories.

    Domains help organize memories and provide context for queries.
    They can be hierarchical (e.g., "programming" > "python" > "data-science").
    """

    id: UUID = Field(default_factory=uuid4)

    name: str = ""  # e.g., "Drupal Development"
    description: str | None = None

    # Hierarchy
    parent_domain: UUID | None = None
    child_domains: list[UUID] = Field(default_factory=list)

    # Associated memories (by UUID)
    memory_ids: list[UUID] = Field(default_factory=list)

    # Domain-specific vocabulary
    terminology: dict[str, str] = Field(default_factory=dict)
    # e.g., {"SDC": "Single Directory Components", "hook": "Drupal event handler"}

    # Expertise level (0.0 novice to 1.0 expert)
    expertise_level: float = 0.0

    # Statistics
    memory_count: int = 0
    last_activity: datetime | None = None

    # Temporal
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Flexible metadata
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": False}

    def add_memory(self, memory_id: UUID) -> None:
        """Add a memory to this domain."""
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)
            self.memory_count += 1
            self.last_activity = datetime.now()
            self.updated_at = datetime.now()

    def remove_memory(self, memory_id: UUID) -> None:
        """Remove a memory from this domain."""
        if memory_id in self.memory_ids:
            self.memory_ids.remove(memory_id)
            self.memory_count -= 1
            self.updated_at = datetime.now()

    def update_expertise(self, level: float) -> None:
        """Update expertise level (clamped to 0.0-1.0)."""
        self.expertise_level = max(0.0, min(1.0, level))
        self.updated_at = datetime.now()


# Predefined domains for initialization
DEFAULT_DOMAINS: list[dict[str, str | dict[str, str]]] = [
    {
        "name": "programming",
        "description": "General software development",
        "terminology": {
            "refactor": "Restructure code without changing behavior",
            "technical debt": "Accumulated shortcuts that need fixing",
        },
    },
    {
        "name": "drupal",
        "description": "Drupal CMS development",
        "terminology": {
            "SDC": "Single Directory Components",
            "hook": "Drupal event handler",
            "module": "Drupal extension package",
            "theme": "Drupal presentation layer",
        },
    },
    {
        "name": "python",
        "description": "Python programming",
        "terminology": {
            "pip": "Python package manager",
            "venv": "Virtual environment",
            "pydantic": "Data validation library",
        },
    },
    {
        "name": "typescript",
        "description": "TypeScript/JavaScript development",
        "terminology": {
            "npm": "Node package manager",
            "ESM": "ECMAScript modules",
            "type guard": "Runtime type checking function",
        },
    },
    {
        "name": "ai_development",
        "description": "AI and machine learning development",
        "terminology": {
            "LLM": "Large Language Model",
            "RAG": "Retrieval Augmented Generation",
            "embedding": "Vector representation of text",
            "fine-tuning": "Adapting a model to specific data",
        },
    },
    {
        "name": "government",
        "description": "Government and public sector projects",
        "terminology": {
            "GovCMS": "Australian Government CMS platform",
            "WCAG": "Web Content Accessibility Guidelines",
        },
    },
    {
        "name": "personal",
        "description": "Personal projects and interests",
        "terminology": {},
    },
]
