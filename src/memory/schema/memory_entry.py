"""
Core memory entry schema - the fundamental building block of the memory system.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class TruthCategory(str, Enum):
    """Distinguishes between types of knowledge."""

    ABSOLUTE = "absolute"  # Empirically verifiable: 1+1=2, spelling, dates
    CONTEXTUAL = "contextual"  # True in specific contexts: "Vue > React for X"
    OPINION = "opinion"  # Personal beliefs, preferences, cannot be proven
    INFERRED = "inferred"  # Derived from patterns, not explicitly stated


class MemoryTier(str, Enum):
    """Three-tier memory model."""

    SHORT_TERM = "short_term"  # Recent interactions, unvalidated
    LONG_TERM = "long_term"  # Validated across multiple interactions
    PERSISTENT = "persistent"  # Core truths, identity, verified facts


class MemoryType(str, Enum):
    """Classification of memory content."""

    FACT = "fact"  # Verifiable information
    BELIEF = "belief"  # Held convictions
    PREFERENCE = "preference"  # Likes/dislikes
    SKILL = "skill"  # Known capabilities
    RELATIONSHIP = "relationship"  # Connections between entities
    EVENT = "event"  # Timestamped occurrences
    CONTEXT = "context"  # Domain/situational information


class SourceType(str, Enum):
    """Origin of the memory."""

    CLAUDE_HISTORY = "claude_history"
    GIT_COMMIT = "git_commit"
    DOCUMENT = "document"
    GITHUB_REPO = "github_repo"
    IMESSAGE = "imessage"
    EMAIL = "email"
    NOTES = "notes"
    MANUAL = "manual"
    INFERRED = "inferred"


class MemorySource(BaseModel):
    """Provenance tracking for memories."""

    source_type: SourceType
    source_path: str | None = None
    source_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    raw_content: str | None = None
    extraction_method: str = "unknown"

    model_config = {"frozen": False}


class ConfidenceScore(BaseModel):
    """Multi-dimensional confidence assessment."""

    overall: float = 0.5  # 0.0 - 1.0 (computed)
    source_reliability: float = 0.5  # Based on source type
    recency: float = 0.5  # Decays over time
    corroboration_count: int = 0  # Number of supporting sources
    contradiction_count: int = 0  # Number of conflicting sources
    last_validated: datetime | None = None

    model_config = {"frozen": False}

    def compute_overall(self) -> float:
        """
        Compute weighted confidence score.

        Weights:
        - source_reliability: 30%
        - recency: 20%
        - corroboration: 30% (capped at 5 corroborations)
        - contradiction penalty: 20% (capped at 3)
        """
        base = self.source_reliability * 0.3
        recency_weight = self.recency * 0.2
        corroboration_weight = min(self.corroboration_count / 5, 1.0) * 0.3
        contradiction_penalty = min(self.contradiction_count / 3, 1.0) * 0.2

        self.overall = max(
            0.0, min(1.0, base + recency_weight + corroboration_weight - contradiction_penalty)
        )
        return self.overall

    @classmethod
    def from_source_type(cls, source_type: SourceType, created_at: datetime) -> ConfidenceScore:
        """Create initial confidence score based on source type."""
        source_reliability = {
            SourceType.MANUAL: 0.9,
            SourceType.GIT_COMMIT: 0.85,
            SourceType.DOCUMENT: 0.7,
            SourceType.CLAUDE_HISTORY: 0.6,
            SourceType.GITHUB_REPO: 0.6,
            SourceType.EMAIL: 0.7,
            SourceType.IMESSAGE: 0.6,
            SourceType.NOTES: 0.65,
            SourceType.INFERRED: 0.4,
        }.get(source_type, 0.5)

        # Recency: 1.0 for today, decay over time
        days_old = (datetime.now() - created_at).days
        recency = max(0.1, 1.0 - (days_old / 365))

        score = cls(source_reliability=source_reliability, recency=recency)
        score.compute_overall()
        return score


class MemoryEntry(BaseModel):
    """
    Core memory unit - the fundamental building block of the memory system.

    Each memory represents a discrete piece of knowledge, belief, preference,
    or fact about the user, with full provenance tracking and confidence scoring.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4)

    # Content
    content: str = ""  # The memory itself (natural language)
    summary: str | None = None  # AI-generated summary

    # Classification
    tier: MemoryTier = MemoryTier.SHORT_TERM
    truth_category: TruthCategory = TruthCategory.INFERRED
    memory_type: MemoryType = MemoryType.FACT

    # Provenance
    sources: list[MemorySource] = Field(default_factory=list)

    # Confidence
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)

    # Semantic
    embedding: list[float] | None = None  # Vector embedding
    embedding_model: str = ""  # Model used for embedding

    # Relationships
    related_memories: list[UUID] = Field(default_factory=list)
    contradicts: list[UUID] = Field(default_factory=list)
    supports: list[UUID] = Field(default_factory=list)
    supersedes: UUID | None = None  # If this updates an older memory

    # Organization
    tags: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)  # e.g., ["programming", "drupal"]
    entities: list[str] = Field(default_factory=list)  # Named entities mentioned

    # Temporal
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0

    # Lifecycle
    is_active: bool = True
    promotion_history: list[dict[str, Any]] = Field(default_factory=list)

    # Flexible metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    @computed_field
    @property
    def content_hash(self) -> str:
        """Generate content hash for deduplication."""
        normalized = " ".join(self.content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def record_access(self) -> None:
        """Record that this memory was accessed."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def add_source(self, source: MemorySource) -> None:
        """Add a new source and update confidence."""
        self.sources.append(source)
        self.confidence.corroboration_count += 1
        self.confidence.compute_overall()
        self.updated_at = datetime.now()

    def add_contradiction(self, memory_id: UUID) -> None:
        """Record a contradicting memory."""
        if memory_id not in self.contradicts:
            self.contradicts.append(memory_id)
            self.confidence.contradiction_count += 1
            self.confidence.compute_overall()
            self.updated_at = datetime.now()

    def add_support(self, memory_id: UUID) -> None:
        """Record a supporting memory."""
        if memory_id not in self.supports:
            self.supports.append(memory_id)
            self.confidence.corroboration_count += 1
            self.confidence.compute_overall()
            self.updated_at = datetime.now()

    def promote_to(self, new_tier: MemoryTier) -> None:
        """Promote this memory to a new tier."""
        old_tier = self.tier
        self.tier = new_tier
        self.promotion_history.append(
            {
                "from": old_tier.value,
                "to": new_tier.value,
                "timestamp": datetime.now().isoformat(),
                "confidence": self.confidence.overall,
            }
        )
        self.updated_at = datetime.now()

    def to_context_string(self, include_metadata: bool = False) -> str:
        """Format memory for LLM context injection."""
        parts = [f"[{self.truth_category.value}] {self.summary or self.content[:200]}"]

        if include_metadata:
            parts.append(f"  Domains: {', '.join(self.domains)}")
            parts.append(f"  Confidence: {self.confidence.overall:.2f}")
            parts.append(f"  Type: {self.memory_type.value}")

        return "\n".join(parts)
