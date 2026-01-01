"""
Registry of available data sources.

Provides discovery and management of all data source adapters.
"""

from __future__ import annotations

from typing import Any

from memory.ingestion.sources.base import DataCategory, DataSource


class SourceRegistry:
    """
    Registry for data source adapters.

    Allows dynamic registration and discovery of data sources
    that can contribute to a personal model.
    """

    _sources: dict[str, type[DataSource]] = {}
    _instances: dict[str, DataSource] = {}

    @classmethod
    def register(cls, source_class: type[DataSource]) -> type[DataSource]:
        """Register a data source class."""
        # Use class name as key
        key = source_class.__name__
        cls._sources[key] = source_class
        return source_class

    @classmethod
    def get_source(cls, name: str, **kwargs: Any) -> DataSource | None:
        """Get an instance of a registered source."""
        if name in cls._instances:
            return cls._instances[name]

        if name in cls._sources:
            instance = cls._sources[name](**kwargs)
            cls._instances[name] = instance
            return instance

        return None

    @classmethod
    def list_sources(cls) -> list[dict[str, Any]]:
        """List all available sources with their info."""
        sources = []

        # Add built-in sources
        for source_id, info in BUILTIN_SOURCES.items():
            sources.append(
                {
                    "id": source_id,
                    "name": info.get("name", source_id),
                    "category": info.get("category", DataCategory.OTHER).value
                    if hasattr(info.get("category", DataCategory.OTHER), "value")
                    else str(info.get("category", "other")),
                    "description": info.get("export_format") or info.get("access") or "",
                    "data_types": info.get("data_types", []),
                }
            )

        return sources

    @classmethod
    def get_sources_by_category(cls, category: DataCategory) -> list[str]:
        """Get sources that provide data for a category."""
        matching = []
        for name, source_class in cls._sources.items():
            # Create temporary instance to check category
            try:
                if hasattr(source_class, "category"):
                    # Check if it's a property
                    if source_class.category == category:
                        matching.append(name)
            except Exception:
                pass
        return matching


async def get_available_sources() -> list[dict[str, Any]]:
    """
    Get metadata for all available data sources.

    Returns information about each source and whether it's accessible.
    """
    available = []

    for name in SourceRegistry.list_sources():
        source = SourceRegistry.get_source(name)
        if source:
            try:
                metadata = await source.get_metadata()
                available.append(metadata)
            except Exception as e:
                available.append(
                    {
                        "name": name,
                        "error": str(e),
                        "available": False,
                    }
                )

    return available


# Built-in source definitions
BUILTIN_SOURCES = {
    # Social Media
    "twitter": {
        "name": "Twitter/X",
        "category": DataCategory.SOCIAL_MEDIA,
        "export_format": "GDPR data download (JSON)",
        "data_types": ["tweets", "likes", "replies", "DMs", "bookmarks"],
    },
    "facebook": {
        "name": "Facebook",
        "category": DataCategory.SOCIAL_MEDIA,
        "export_format": "Download Your Information (JSON)",
        "data_types": ["posts", "comments", "messages", "photos", "events"],
    },
    "instagram": {
        "name": "Instagram",
        "category": DataCategory.SOCIAL_MEDIA,
        "export_format": "Download Data (JSON)",
        "data_types": ["posts", "stories", "comments", "messages", "saved"],
    },
    "linkedin": {
        "name": "LinkedIn",
        "category": DataCategory.LINKEDIN,
        "export_format": "Get a copy of your data",
        "data_types": ["profile", "connections", "messages", "posts", "endorsements"],
    },
    "reddit": {
        "name": "Reddit",
        "category": DataCategory.SOCIAL_MEDIA,
        "export_format": "GDPR data request",
        "data_types": ["posts", "comments", "saved", "upvotes"],
    },
    # Communications
    "gmail": {
        "name": "Gmail",
        "category": DataCategory.EMAIL,
        "export_format": "Google Takeout (MBOX)",
        "data_types": ["emails", "drafts", "sent", "labels"],
    },
    "apple_mail": {
        "name": "Apple Mail",
        "category": DataCategory.EMAIL,
        "access": "~/Library/Mail",
        "data_types": ["emails"],
    },
    "imessage": {
        "name": "iMessage",
        "category": DataCategory.MESSAGE,
        "access": "~/Library/Messages/chat.db",
        "data_types": ["messages", "attachments"],
    },
    "whatsapp": {
        "name": "WhatsApp",
        "category": DataCategory.MESSAGE,
        "export_format": "Export chat",
        "data_types": ["messages", "media"],
    },
    "slack": {
        "name": "Slack",
        "category": DataCategory.CHAT,
        "export_format": "Workspace export (JSON)",
        "data_types": ["messages", "files", "reactions"],
    },
    "discord": {
        "name": "Discord",
        "category": DataCategory.CHAT,
        "export_format": "GDPR data request",
        "data_types": ["messages", "servers"],
    },
    # Documents
    "google_docs": {
        "name": "Google Docs",
        "category": DataCategory.DOCUMENT,
        "export_format": "Google Takeout",
        "data_types": ["documents", "sheets", "slides"],
    },
    "notion": {
        "name": "Notion",
        "category": DataCategory.NOTE,
        "export_format": "Export workspace (Markdown/HTML)",
        "data_types": ["pages", "databases", "comments"],
    },
    "obsidian": {
        "name": "Obsidian",
        "category": DataCategory.NOTE,
        "access": "Vault folder (Markdown)",
        "data_types": ["notes", "links"],
    },
    "apple_notes": {
        "name": "Apple Notes",
        "category": DataCategory.NOTE,
        "access": "~/Library/Group Containers/group.com.apple.notes",
        "data_types": ["notes", "folders"],
    },
    "evernote": {
        "name": "Evernote",
        "category": DataCategory.NOTE,
        "export_format": "Export notes (ENEX)",
        "data_types": ["notes", "notebooks", "tags"],
    },
    # Media
    "apple_photos": {
        "name": "Apple Photos",
        "category": DataCategory.PHOTO,
        "access": "~/Pictures/Photos Library.photoslibrary",
        "data_types": ["photos", "albums", "memories", "metadata"],
    },
    "google_photos": {
        "name": "Google Photos",
        "category": DataCategory.PHOTO,
        "export_format": "Google Takeout",
        "data_types": ["photos", "albums", "metadata"],
    },
    "youtube": {
        "name": "YouTube",
        "category": DataCategory.VIDEO,
        "export_format": "Google Takeout",
        "data_types": ["watch_history", "likes", "playlists", "comments"],
    },
    "spotify": {
        "name": "Spotify",
        "category": DataCategory.MUSIC,
        "export_format": "Download your data",
        "data_types": ["streaming_history", "playlists", "library"],
    },
    # Code & Technical
    "github": {
        "name": "GitHub",
        "category": DataCategory.CODE,
        "access": "GitHub API / gh CLI",
        "data_types": ["repos", "commits", "issues", "PRs", "stars", "gists"],
    },
    "gitlab": {
        "name": "GitLab",
        "category": DataCategory.CODE,
        "access": "GitLab API",
        "data_types": ["projects", "commits", "issues", "MRs"],
    },
    "local_git": {
        "name": "Local Git Repos",
        "category": DataCategory.CODE,
        "access": "Local filesystem",
        "data_types": ["commits", "diffs", "branches"],
    },
    "stackoverflow": {
        "name": "Stack Overflow",
        "category": DataCategory.CODE,
        "export_format": "GDPR request / API",
        "data_types": ["questions", "answers", "comments", "votes"],
    },
    # Professional
    "resume": {
        "name": "Resume/CV",
        "category": DataCategory.RESUME,
        "access": "PDF/DOCX files",
        "data_types": ["experience", "education", "skills"],
    },
    # Academic
    "google_scholar": {
        "name": "Google Scholar",
        "category": DataCategory.RESEARCH,
        "access": "Export library",
        "data_types": ["papers", "citations"],
    },
    "zotero": {
        "name": "Zotero",
        "category": DataCategory.RESEARCH,
        "access": "Zotero database",
        "data_types": ["references", "notes", "annotations"],
    },
    # Browser
    "chrome_history": {
        "name": "Chrome History",
        "category": DataCategory.SEARCH,
        "access": "~/Library/Application Support/Google/Chrome/Default/History",
        "data_types": ["browsing_history", "bookmarks", "searches"],
    },
    "safari_history": {
        "name": "Safari History",
        "category": DataCategory.SEARCH,
        "access": "~/Library/Safari/History.db",
        "data_types": ["browsing_history", "bookmarks"],
    },
    # Calendar
    "google_calendar": {
        "name": "Google Calendar",
        "category": DataCategory.CALENDAR,
        "export_format": "Google Takeout / iCal",
        "data_types": ["events", "reminders"],
    },
    "apple_calendar": {
        "name": "Apple Calendar",
        "category": DataCategory.CALENDAR,
        "access": "~/Library/Calendars",
        "data_types": ["events", "reminders"],
    },
    # Health & Fitness
    "apple_health": {
        "name": "Apple Health",
        "category": DataCategory.HEALTH_RECORD,
        "export_format": "Export Health Data",
        "data_types": ["workouts", "vitals", "sleep", "nutrition"],
    },
    "strava": {
        "name": "Strava",
        "category": DataCategory.FITNESS,
        "export_format": "Download your data",
        "data_types": ["activities", "routes"],
    },
    # AI Interactions
    "claude_history": {
        "name": "Claude Code History",
        "category": DataCategory.CHAT,
        "access": "~/.claude/history.jsonl",
        "data_types": ["conversations", "code_changes"],
    },
    "chatgpt_history": {
        "name": "ChatGPT History",
        "category": DataCategory.CHAT,
        "export_format": "Export data",
        "data_types": ["conversations"],
    },
    # Financial
    "bank_statements": {
        "name": "Bank Statements",
        "category": DataCategory.TRANSACTION,
        "access": "CSV/PDF exports",
        "data_types": ["transactions", "categories"],
    },
    # Location
    "google_maps": {
        "name": "Google Maps Timeline",
        "category": DataCategory.OTHER,
        "export_format": "Google Takeout",
        "data_types": ["location_history", "places"],
    },
}


def get_source_info(source_name: str) -> dict[str, Any] | None:
    """Get information about a built-in source."""
    return BUILTIN_SOURCES.get(source_name)


def list_all_sources() -> dict[str, dict[str, Any]]:
    """List all built-in source definitions."""
    return BUILTIN_SOURCES.copy()


def get_sources_by_category(category: DataCategory) -> list[str]:
    """Get all sources for a given category."""
    return [name for name, info in BUILTIN_SOURCES.items() if info.get("category") == category]
