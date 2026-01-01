"""Tests for memory schema module."""

from datetime import datetime
from uuid import UUID


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_create_basic_entry(self):
        """Test creating a basic memory entry."""
        from memory.schema import MemoryEntry

        entry = MemoryEntry(content="Test memory content")

        assert entry.content == "Test memory content"
        assert isinstance(entry.id, UUID)
        assert entry.tier.value == "short_term"

    def test_entry_with_all_fields(self):
        """Test creating entry with all fields."""
        from memory.schema import (
            ConfidenceScore,
            MemoryEntry,
            MemoryTier,
            MemoryType,
            TruthCategory,
        )

        confidence = ConfidenceScore(overall=0.75, source_reliability=0.8)
        entry = MemoryEntry(
            content="I prefer Python for data processing",
            summary="Python preference for data",
            tier=MemoryTier.LONG_TERM,
            truth_category=TruthCategory.OPINION,
            memory_type=MemoryType.PREFERENCE,
            confidence=confidence,
            domains=["programming", "python"],
            tags=["preference", "language"],
        )

        assert entry.content == "I prefer Python for data processing"
        assert entry.tier == MemoryTier.LONG_TERM
        assert entry.truth_category == TruthCategory.OPINION
        assert entry.memory_type == MemoryType.PREFERENCE
        assert entry.confidence.overall == 0.75
        assert "programming" in entry.domains

    def test_entry_defaults(self):
        """Test default values for entry."""
        from memory.schema import MemoryEntry, MemoryTier, MemoryType, TruthCategory

        entry = MemoryEntry(content="Minimal entry")

        assert entry.tier == MemoryTier.SHORT_TERM
        assert entry.truth_category == TruthCategory.INFERRED
        assert entry.memory_type == MemoryType.FACT
        assert entry.confidence.overall == 0.5
        assert entry.domains == []
        assert entry.tags == []

    def test_entry_timestamps(self):
        """Test that timestamps are set correctly."""
        from memory.schema import MemoryEntry

        before = datetime.now()
        entry = MemoryEntry(content="Test")
        after = datetime.now()

        assert before <= entry.created_at <= after
        assert before <= entry.updated_at <= after

    def test_entry_to_dict(self):
        """Test converting entry to dictionary."""
        from memory.schema import ConfidenceScore, MemoryEntry

        confidence = ConfidenceScore(overall=0.9)
        entry = MemoryEntry(
            content="Test content",
            confidence=confidence,
            domains=["test"],
        )

        data = entry.model_dump()

        assert "id" in data
        assert data["content"] == "Test content"
        assert "confidence" in data
        assert data["domains"] == ["test"]

    def test_entry_from_dict(self):
        """Test creating entry from dictionary."""
        from memory.schema import MemoryEntry

        data = {
            "content": "Test content",
            "domains": ["test"],
            "tags": ["tag1"],
        }

        entry = MemoryEntry.model_validate(data)

        assert entry.content == "Test content"
        assert entry.domains == ["test"]


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_all_memory_types(self):
        """Test all memory types exist."""
        from memory.schema import MemoryType

        assert MemoryType.FACT.value == "fact"
        assert MemoryType.BELIEF.value == "belief"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.SKILL.value == "skill"
        assert MemoryType.EVENT.value == "event"
        assert MemoryType.CONTEXT.value == "context"

    def test_memory_type_from_string(self):
        """Test creating memory type from string."""
        from memory.schema import MemoryType

        assert MemoryType("preference") == MemoryType.PREFERENCE


class TestTruthCategory:
    """Tests for TruthCategory enum."""

    def test_all_truth_categories(self):
        """Test all truth categories exist."""
        from memory.schema import TruthCategory

        assert TruthCategory.ABSOLUTE.value == "absolute"
        assert TruthCategory.CONTEXTUAL.value == "contextual"
        assert TruthCategory.OPINION.value == "opinion"
        assert TruthCategory.INFERRED.value == "inferred"


class TestMemoryTier:
    """Tests for MemoryTier enum."""

    def test_all_tiers(self):
        """Test all memory tiers exist."""
        from memory.schema import MemoryTier

        assert MemoryTier.SHORT_TERM.value == "short_term"
        assert MemoryTier.LONG_TERM.value == "long_term"
        assert MemoryTier.PERSISTENT.value == "persistent"


class TestConfidenceScore:
    """Tests for ConfidenceScore model."""

    def test_confidence_defaults(self):
        """Test default confidence values."""
        from memory.schema import ConfidenceScore

        score = ConfidenceScore()

        assert score.overall == 0.5
        assert score.source_reliability == 0.5
        assert score.recency == 0.5
        assert score.corroboration_count == 0
        assert score.contradiction_count == 0

    def test_confidence_custom_values(self):
        """Test custom confidence values."""
        from memory.schema import ConfidenceScore

        score = ConfidenceScore(
            overall=0.8,
            source_reliability=0.9,
            recency=0.7,
            corroboration_count=3,
        )

        assert score.overall == 0.8
        assert score.source_reliability == 0.9
        assert score.corroboration_count == 3

    def test_compute_overall(self):
        """Test computing overall confidence score."""
        from memory.schema import ConfidenceScore

        score = ConfidenceScore(
            source_reliability=0.8,
            recency=0.9,
            corroboration_count=2,
            contradiction_count=0,
        )
        computed = score.compute_overall()

        assert 0.0 <= computed <= 1.0
        assert computed == score.overall


class TestIdentityProfile:
    """Tests for IdentityProfile model."""

    def test_create_identity(self):
        """Test creating an identity profile."""
        from memory.schema import IdentityProfile

        profile = IdentityProfile(
            name="Test User",
            email="test@example.com",
        )

        assert profile.name == "Test User"
        assert profile.email == "test@example.com"

    def test_identity_with_skills(self):
        """Test identity with technical profile."""
        from memory.schema import IdentityProfile

        profile = IdentityProfile(
            name="Developer",
            primary_languages=["Python", "JavaScript"],
            frameworks=["Django", "React"],
        )

        assert "Python" in profile.primary_languages
        assert "React" in profile.frameworks

    def test_identity_to_context_string(self):
        """Test converting identity to context string."""
        from memory.schema import IdentityProfile

        profile = IdentityProfile(
            name="Test User",
            primary_languages=["Python"],
            frameworks=["Django"],
        )

        context = profile.to_context_string()

        assert "Test User" in context
        assert "Python" in context

    def test_create_default_identity(self):
        """Test creating empty default identity profile."""
        from memory.schema import IdentityProfile

        profile = IdentityProfile.create_default()

        # Default profile should be empty (personal info loaded from config)
        assert profile.name == ""
        assert profile.primary_languages == []


class TestMemorySource:
    """Tests for MemorySource model."""

    def test_create_source(self):
        """Test creating a memory source."""
        from memory.schema import MemorySource, SourceType

        source = MemorySource(
            source_type=SourceType.GIT_COMMIT,
            source_path="/path/to/repo",
            source_id="abc123",
        )

        assert source.source_type == SourceType.GIT_COMMIT
        assert source.source_path == "/path/to/repo"
        assert source.source_id == "abc123"

    def test_source_types(self):
        """Test all source types exist."""
        from memory.schema import SourceType

        assert SourceType.CLAUDE_HISTORY.value == "claude_history"
        assert SourceType.GIT_COMMIT.value == "git_commit"
        assert SourceType.DOCUMENT.value == "document"
        assert SourceType.GITHUB_REPO.value == "github_repo"
        assert SourceType.MANUAL.value == "manual"

    def test_source_defaults(self):
        """Test source default values."""
        from memory.schema import MemorySource, SourceType

        source = MemorySource(source_type=SourceType.MANUAL)

        assert source.source_path is None
        assert source.source_id is None
        assert source.extraction_method == "unknown"
