"""
Universal data source adapters for PLM.

Any digital artifact can become a memory. This module provides
adapters for common data sources that anyone might have.
"""

from .base import DataCategory, DataPoint, DataSource

# Browser
from .browser import (
    ArcHistorySource,
    ChromeBookmarksSource,
    ChromeHistorySource,
    FirefoxHistorySource,
    SafariBookmarksSource,
    SafariHistorySource,
)

# Cloud Services
from .cloud import AWSSource

# Code
from .code import (
    GitHubSource,
    GitLabSource,
    LocalGitSource,
)

# Communication
from .communication import (
    AppleMailSource,
    DiscordExportSource,
    GmailAPISource,
    GmailExportSource,
    SlackExportSource,
    WhatsAppExportSource,
    iMessageSource,
)

# Documents
from .documents import (
    AppleNotesSource,
    BearExportSource,
    EvernoteExportSource,
    GoogleDocsExportSource,
    LocalDocumentsSource,
    NotionExportSource,
    ObsidianVaultSource,
)

# Media
from .media import (
    ApplePhotosSource,
    GooglePhotosExportSource,
    SpotifyExportSource,
    YouTubeHistorySource,
)
from .registry import SourceRegistry, get_available_sources

# Social Media
from .social import (
    FacebookExportSource,
    InstagramExportSource,
    LinkedInExportSource,
    TwitterExportSource,
)

__all__ = [
    # Base
    "DataCategory",
    "DataPoint",
    "DataSource",
    "SourceRegistry",
    "get_available_sources",
    # Social
    "TwitterExportSource",
    "FacebookExportSource",
    "InstagramExportSource",
    "LinkedInExportSource",
    # Media
    "ApplePhotosSource",
    "GooglePhotosExportSource",
    "YouTubeHistorySource",
    "SpotifyExportSource",
    # Communication
    "iMessageSource",
    "SlackExportSource",
    "DiscordExportSource",
    "WhatsAppExportSource",
    "GmailExportSource",
    "GmailAPISource",
    "AppleMailSource",
    # Documents
    "ObsidianVaultSource",
    "NotionExportSource",
    "AppleNotesSource",
    "EvernoteExportSource",
    "BearExportSource",
    "LocalDocumentsSource",
    "GoogleDocsExportSource",
    # Code
    "LocalGitSource",
    "GitHubSource",
    "GitLabSource",
    # Browser
    "ChromeHistorySource",
    "SafariHistorySource",
    "FirefoxHistorySource",
    "ChromeBookmarksSource",
    "SafariBookmarksSource",
    "ArcHistorySource",
    # Cloud
    "AWSSource",
]
