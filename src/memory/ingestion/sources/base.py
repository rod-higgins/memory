"""
Base classes for universal data source ingestion.

The Personal SLM can learn from ANY digital artifact:
- Social media posts, comments, likes
- Photos (metadata, captions, OCR text)
- Videos (transcripts, descriptions)
- Documents (essays, reports, presentations)
- Communications (email, messages, chats)
- Code and projects
- Academic work
- Professional history
- Creative works
- And anything else...

Each source adapter converts raw data into standardized DataPoints
that can be used to train the personal model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class DataCategory(str, Enum):
    """Categories of personal data."""

    # Communications
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    MESSAGE = "message"
    CHAT = "chat"
    COMMUNICATION = "communication"

    # Documents
    DOCUMENT = "document"
    DOCUMENTS = "documents"  # Alias
    ESSAY = "essay"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    NOTE = "note"
    NOTES = "notes"  # Alias

    # Media
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    MUSIC = "music"

    # Code & Technical
    CODE = "code"
    COMMIT = "commit"
    PROJECT = "project"
    ISSUE = "issue"

    # Professional
    RESUME = "resume"
    WORK_HISTORY = "work_history"
    LINKEDIN = "linkedin"
    PORTFOLIO = "portfolio"

    # Academic
    COURSEWORK = "coursework"
    RESEARCH = "research"
    THESIS = "thesis"
    TRANSCRIPT = "transcript"

    # Personal
    JOURNAL = "journal"
    DIARY = "diary"
    CALENDAR = "calendar"
    BOOKMARK = "bookmark"
    SEARCH = "search"
    BROWSING = "browsing"

    # Financial
    TRANSACTION = "transaction"
    INVESTMENT = "investment"

    # Health
    HEALTH_RECORD = "health_record"
    FITNESS = "fitness"

    # Other
    OTHER = "other"


@dataclass
class DataPoint:
    """
    A single data point from any source.

    This is the universal format that all sources convert to.
    Each DataPoint represents one piece of information that
    can contribute to the personal model.
    """

    id: str = field(default_factory=lambda: str(uuid4()))

    # Core content
    content: str = ""  # The main text/content
    summary: str | None = None  # AI-generated or extracted summary

    # Classification
    category: DataCategory = DataCategory.OTHER
    subcategory: str | None = None  # More specific type

    # Source tracking
    source_type: str = ""  # e.g., "twitter", "gmail", "iphone_photos"
    source_path: str | None = None  # Original file/URL
    source_id: str | None = None  # ID in original system

    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    original_date: datetime | None = None  # When the original was created

    # Context
    author: str | None = None  # Who created this (usually the user)
    recipients: list[str] = field(default_factory=list)  # For messages/emails
    location: str | None = None  # Geographic context
    device: str | None = None  # Device that created it

    # Semantic
    topics: list[str] = field(default_factory=list)  # Extracted topics
    entities: list[str] = field(default_factory=list)  # Named entities
    sentiment: str | None = None  # positive/negative/neutral
    language: str = "en"

    # Media (for photos/videos)
    media_type: str | None = None  # image/jpeg, video/mp4, etc.
    media_path: str | None = None  # Path to media file
    ocr_text: str | None = None  # Text extracted from images
    transcript: str | None = None  # For audio/video

    # Relationships
    reply_to: str | None = None  # If this is a reply
    thread_id: str | None = None  # Conversation thread
    related_ids: list[str] = field(default_factory=list)

    # Signals
    engagement: dict[str, int] = field(default_factory=dict)  # likes, shares, etc.
    importance: float = 0.5  # 0-1 estimated importance
    is_public: bool = False  # Was this public or private

    # Raw data (for reprocessing)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_training_text(self) -> str:
        """Convert to text suitable for model training."""
        parts = []

        if self.category != DataCategory.OTHER:
            parts.append(f"[{self.category.value}]")

        if self.original_date:
            parts.append(f"({self.original_date.strftime('%Y-%m-%d')})")

        parts.append(self.content)

        if self.ocr_text:
            parts.append(f"\n[Image text: {self.ocr_text}]")

        if self.transcript:
            parts.append(f"\n[Transcript: {self.transcript}]")

        return " ".join(parts)


class DataSource(ABC):
    """
    Abstract base class for data source adapters.

    Each adapter knows how to:
    1. Detect if the source is available
    2. Estimate how much data is available
    3. Iterate through all data points
    4. Convert raw data to DataPoints
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this source."""
        pass

    @property
    @abstractmethod
    def category(self) -> DataCategory:
        """Primary category of data from this source."""
        pass

    @property
    def description(self) -> str:
        """Description of what this source provides."""
        return f"Data from {self.name}"

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this data source is accessible."""
        pass

    @abstractmethod
    async def estimate_count(self) -> int:
        """Estimate how many data points are available."""
        pass

    @abstractmethod
    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate through all data points from this source."""
        pass

    async def get_metadata(self) -> dict[str, Any]:
        """Get metadata about this source."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "available": await self.is_available(),
            "estimated_count": await self.estimate_count() if await self.is_available() else 0,
        }


class FileBasedSource(DataSource):
    """Base class for file-based data sources."""

    def __init__(self, base_path: str | Path):
        self._base_path = Path(base_path).expanduser()

    async def is_available(self) -> bool:
        return self._base_path.exists()

    @property
    def base_path(self) -> Path:
        return self._base_path


class APIBasedSource(DataSource):
    """Base class for API-based data sources."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    async def is_available(self) -> bool:
        # Subclasses should check API connectivity
        return self._api_key is not None


class ExportBasedSource(DataSource):
    """Base class for sources that require data export (GDPR downloads, etc.)."""

    def __init__(self, export_path: str | Path):
        self._export_path = Path(export_path).expanduser()

    async def is_available(self) -> bool:
        return self._export_path.exists()

    @property
    def export_path(self) -> Path:
        return self._export_path
