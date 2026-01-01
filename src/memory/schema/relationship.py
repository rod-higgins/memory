"""
Memory relationship schema - explicit relationships between memories.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of relationships between memories."""

    SUPPORTS = "supports"  # Memory A provides evidence for Memory B
    CONTRADICTS = "contradicts"  # Memory A conflicts with Memory B
    REFINES = "refines"  # Memory A adds detail to Memory B
    SUPERSEDES = "supersedes"  # Memory A replaces Memory B
    RELATED_TO = "related_to"  # General semantic relationship
    CAUSES = "causes"  # Memory A is a cause of Memory B
    PART_OF = "part_of"  # Memory A is a component of Memory B
    EXAMPLE_OF = "example_of"  # Memory A is an instance of Memory B


class MemoryRelationship(BaseModel):
    """
    Explicit relationship between two memories.

    Relationships are directional (source -> target) but can be
    marked as bidirectional for symmetric relationships.
    """

    id: UUID = Field(default_factory=uuid4)

    source_id: UUID  # The memory this relationship originates from
    target_id: UUID  # The memory this relationship points to
    relationship_type: RelationshipType = RelationshipType.RELATED_TO

    # Relationship strength (0.0 to 1.0)
    strength: float = 0.5

    # If true, relationship applies both directions
    bidirectional: bool = False

    # Provenance
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = "system"  # "system" or "manual"

    # Optional context explaining the relationship
    context: str | None = None

    # Flexible metadata
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": False}

    def invert(self) -> MemoryRelationship:
        """
        Create the inverse relationship (swap source and target).

        Note: Some relationship types have natural inverses:
        - SUPPORTS <-> SUPPORTED_BY (not defined, just swap direction)
        - CAUSES <-> CAUSED_BY (not defined, just swap direction)
        - SUPERSEDES has no natural inverse
        """
        return MemoryRelationship(
            source_id=self.target_id,
            target_id=self.source_id,
            relationship_type=self.relationship_type,
            strength=self.strength,
            bidirectional=self.bidirectional,
            created_by=self.created_by,
            context=self.context,
            metadata=self.metadata.copy(),
        )

    @classmethod
    def create_support(
        cls,
        source_id: UUID,
        target_id: UUID,
        strength: float = 0.7,
        context: str | None = None,
    ) -> MemoryRelationship:
        """Create a support relationship."""
        return cls(
            source_id=source_id,
            target_id=target_id,
            relationship_type=RelationshipType.SUPPORTS,
            strength=strength,
            context=context,
        )

    @classmethod
    def create_contradiction(
        cls,
        source_id: UUID,
        target_id: UUID,
        strength: float = 0.8,
        context: str | None = None,
    ) -> MemoryRelationship:
        """Create a contradiction relationship (always bidirectional)."""
        return cls(
            source_id=source_id,
            target_id=target_id,
            relationship_type=RelationshipType.CONTRADICTS,
            strength=strength,
            bidirectional=True,
            context=context,
        )

    @classmethod
    def create_supersedes(
        cls,
        new_memory_id: UUID,
        old_memory_id: UUID,
        context: str | None = None,
    ) -> MemoryRelationship:
        """Create a supersedes relationship (new replaces old)."""
        return cls(
            source_id=new_memory_id,
            target_id=old_memory_id,
            relationship_type=RelationshipType.SUPERSEDES,
            strength=1.0,
            bidirectional=False,
            context=context,
        )
