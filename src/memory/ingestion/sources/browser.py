"""Browser data source adapters (Chrome, Safari, Firefox history and bookmarks)."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from .base import DataCategory, DataPoint, DataSource


class ChromeHistorySource(DataSource):
    """Adapter for Chrome browser history - scans ALL Chrome profiles."""

    def __init__(
        self,
        chrome_path: str = "~/Library/Application Support/Google/Chrome",
        max_entries: int = 10000,
    ):
        """
        Initialize Chrome history source.

        Args:
            chrome_path: Path to Chrome data directory
            max_entries: Maximum history entries to process per profile
        """
        self.chrome_path = Path(chrome_path).expanduser()
        self.max_entries = max_entries

    @property
    def name(self) -> str:
        return "Chrome History"

    @property
    def category(self) -> DataCategory:
        return DataCategory.BROWSING

    def _find_history_files(self) -> list[Path]:
        """Find all Chrome history files across all profiles."""
        history_files = []
        if not self.chrome_path.exists():
            return history_files

        # Check Default profile
        default_history = self.chrome_path / "Default" / "History"
        if default_history.exists():
            history_files.append(default_history)

        # Check numbered profiles (Profile 1, Profile 2, etc.)
        for profile_dir in self.chrome_path.iterdir():
            if profile_dir.is_dir() and profile_dir.name.startswith("Profile"):
                history_file = profile_dir / "History"
                if history_file.exists():
                    history_files.append(history_file)

        return history_files

    async def is_available(self) -> bool:
        return len(self._find_history_files()) > 0

    async def estimate_count(self) -> int:
        history_files = self._find_history_files()
        return len(history_files) * self.max_entries  # Rough estimate

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Chrome history from all profiles."""
        history_files = self._find_history_files()
        if not history_files:
            return

        for history_db in history_files:
            async for dp in self._iterate_profile(history_db):
                yield dp

    async def _iterate_profile(self, history_db: Path) -> AsyncIterator[DataPoint]:
        """Iterate over a single profile's history."""
        if not history_db.exists():
            return

        # Copy database to temp file (Chrome locks the database)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shutil.copy2(history_db, tmp.name)
            tmp_path = tmp.name

        try:
            conn = sqlite3.connect(tmp_path)
            conn.row_factory = sqlite3.Row

            # Check if tables exist
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='urls'").fetchone()
            if not tables:
                conn.close()
                return

            query = """
                SELECT
                    u.url,
                    u.title,
                    u.visit_count,
                    v.visit_time
                FROM urls u
                JOIN visits v ON u.id = v.url
                ORDER BY v.visit_time DESC
                LIMIT ?
            """

            cursor = conn.execute(query, (self.max_entries,))

            for row in cursor:
                url = row["url"]
                title = row["title"] or url

                # Convert Chrome timestamp (microseconds since 1601-01-01)
                timestamp = None
                if row["visit_time"]:
                    # Chrome epoch is 1601-01-01
                    chrome_epoch = datetime(1601, 1, 1)
                    delta = __import__("datetime").timedelta(microseconds=row["visit_time"])
                    timestamp = chrome_epoch + delta

                yield DataPoint(
                    content=f"{title}\n{url}",
                    category=self.category,
                    source_type="chrome_history",
                    source_id=url,
                    original_date=timestamp,
                    raw_data={
                        "url": url,
                        "title": title,
                        "visit_count": row["visit_count"],
                    },
                    topics=["chrome", "history", self._get_domain(url)],
                )

            conn.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""


class SafariHistorySource(DataSource):
    """Adapter for Safari browser history."""

    source_type = "safari_history"
    category = DataCategory.BROWSING

    def __init__(
        self,
        db_path: str = "~/Library/Safari/History.db",
        max_entries: int = 10000,
        **kwargs,
    ):
        """
        Initialize Safari history source.

        Args:
            db_path: Path to Safari history database
            max_entries: Maximum entries to process
        """
        super().__init__(**kwargs)
        self.db_path = Path(db_path).expanduser()
        self.max_entries = max_entries

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Safari history."""
        if not self.db_path.exists():
            return

        # Copy database to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shutil.copy2(self.db_path, tmp.name)
            tmp_path = tmp.name

        try:
            conn = sqlite3.connect(tmp_path)
            conn.row_factory = sqlite3.Row

            query = """
                SELECT
                    hi.url,
                    hv.title,
                    hv.visit_time
                FROM history_items hi
                JOIN history_visits hv ON hi.id = hv.history_item
                ORDER BY hv.visit_time DESC
                LIMIT ?
            """

            cursor = conn.execute(query, (self.max_entries,))

            for row in cursor:
                url = row["url"]
                title = row["title"] or url

                # Convert Safari timestamp (seconds since 2001-01-01)
                timestamp = None
                if row["visit_time"]:
                    safari_epoch = datetime(2001, 1, 1)
                    delta = __import__("datetime").timedelta(seconds=row["visit_time"])
                    timestamp = safari_epoch + delta

                yield DataPoint(
                    content=f"{title}\n{url}",
                    category=self.category,
                    source_type=self.source_type,
                    source_identifier=url,
                    timestamp=timestamp,
                    metadata={
                        "url": url,
                        "title": title,
                    },
                    tags=["safari", "history", self._get_domain(url)],
                )

            conn.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""


class FirefoxHistorySource(DataSource):
    """Adapter for Firefox browser history."""

    source_type = "firefox_history"
    category = DataCategory.BROWSING

    def __init__(
        self,
        profile_path: str | None = None,
        max_entries: int = 10000,
        **kwargs,
    ):
        """
        Initialize Firefox history source.

        Args:
            profile_path: Path to Firefox profile (auto-detected if None)
            max_entries: Maximum entries to process
        """
        super().__init__(**kwargs)
        self.profile_path = self._find_profile(profile_path)
        self.max_entries = max_entries

    def _find_profile(self, profile_path: str | None) -> Path | None:
        """Find Firefox profile path."""
        if profile_path:
            return Path(profile_path).expanduser()

        # Auto-detect on macOS
        firefox_dir = Path("~/Library/Application Support/Firefox/Profiles").expanduser()
        if firefox_dir.exists():
            for profile in firefox_dir.iterdir():
                if profile.is_dir() and (profile / "places.sqlite").exists():
                    return profile

        return None

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Firefox history."""
        if not self.profile_path:
            return

        places_db = self.profile_path / "places.sqlite"
        if not places_db.exists():
            return

        # Copy database to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shutil.copy2(places_db, tmp.name)
            tmp_path = tmp.name

        try:
            conn = sqlite3.connect(tmp_path)
            conn.row_factory = sqlite3.Row

            query = """
                SELECT
                    p.url,
                    p.title,
                    p.visit_count,
                    h.visit_date
                FROM moz_places p
                JOIN moz_historyvisits h ON p.id = h.place_id
                WHERE p.hidden = 0
                ORDER BY h.visit_date DESC
                LIMIT ?
            """

            cursor = conn.execute(query, (self.max_entries,))

            for row in cursor:
                url = row["url"]
                title = row["title"] or url

                # Convert Firefox timestamp (microseconds since 1970-01-01)
                timestamp = None
                if row["visit_date"]:
                    timestamp = datetime.fromtimestamp(row["visit_date"] / 1_000_000)

                yield DataPoint(
                    content=f"{title}\n{url}",
                    category=self.category,
                    source_type=self.source_type,
                    source_identifier=url,
                    timestamp=timestamp,
                    metadata={
                        "url": url,
                        "title": title,
                        "visit_count": row["visit_count"],
                    },
                    tags=["firefox", "history", self._get_domain(url)],
                )

            conn.close()

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""


class ChromeBookmarksSource(DataSource):
    """Adapter for Chrome bookmarks."""

    source_type = "chrome_bookmarks"
    category = DataCategory.BROWSING

    def __init__(
        self,
        profile_path: str = "~/Library/Application Support/Google/Chrome/Default",
        **kwargs,
    ):
        """
        Initialize Chrome bookmarks source.

        Args:
            profile_path: Path to Chrome profile
        """
        super().__init__(**kwargs)
        self.profile_path = Path(profile_path).expanduser()

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Chrome bookmarks."""
        bookmarks_file = self.profile_path / "Bookmarks"
        if not bookmarks_file.exists():
            return

        with open(bookmarks_file, encoding="utf-8") as f:
            data = json.load(f)

        async for dp in self._iterate_folder(data.get("roots", {}), []):
            yield dp

    async def _iterate_folder(self, node: dict, path: list[str]) -> AsyncIterator[DataPoint]:
        """Recursively iterate over bookmark folders."""
        for key, value in node.items():
            if isinstance(value, dict):
                if value.get("type") == "url":
                    url = value.get("url", "")
                    name = value.get("name", "")

                    # Parse date
                    timestamp = None
                    date_added = value.get("date_added")
                    if date_added:
                        try:
                            # Chrome timestamp
                            chrome_epoch = datetime(1601, 1, 1)
                            delta = __import__("datetime").timedelta(microseconds=int(date_added))
                            timestamp = chrome_epoch + delta
                        except (ValueError, OSError):
                            pass

                    yield DataPoint(
                        content=f"{name}\n{url}",
                        category=self.category,
                        source_type=self.source_type,
                        source_identifier=url,
                        timestamp=timestamp,
                        metadata={
                            "url": url,
                            "name": name,
                            "folder": "/".join(path),
                        },
                        tags=["chrome", "bookmark"] + path,
                    )

                elif value.get("type") == "folder":
                    folder_name = value.get("name", key)
                    children = value.get("children", [])
                    new_path = path + [folder_name]

                    for child in children:
                        async for dp in self._iterate_folder({child.get("name", ""): child}, new_path):
                            yield dp


class SafariBookmarksSource(DataSource):
    """Adapter for Safari bookmarks."""

    source_type = "safari_bookmarks"
    category = DataCategory.BROWSING

    def __init__(
        self,
        plist_path: str = "~/Library/Safari/Bookmarks.plist",
        **kwargs,
    ):
        """
        Initialize Safari bookmarks source.

        Args:
            plist_path: Path to Safari bookmarks plist
        """
        super().__init__(**kwargs)
        self.plist_path = Path(plist_path).expanduser()

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Safari bookmarks."""
        if not self.plist_path.exists():
            return

        try:
            import plistlib

            with open(self.plist_path, "rb") as f:
                data = plistlib.load(f)
        except Exception:
            return

        async for dp in self._iterate_folder(data, []):
            yield dp

    async def _iterate_folder(self, node: dict, path: list[str]) -> AsyncIterator[DataPoint]:
        """Recursively iterate over bookmark folders."""
        children = node.get("Children", [])

        for child in children:
            web_bookmark_type = child.get("WebBookmarkType", "")

            if web_bookmark_type == "WebBookmarkTypeLeaf":
                url = child.get("URLString", "")
                title = child.get("URIDictionary", {}).get("title", "") or url

                yield DataPoint(
                    content=f"{title}\n{url}",
                    category=self.category,
                    source_type=self.source_type,
                    source_identifier=url,
                    timestamp=None,
                    metadata={
                        "url": url,
                        "title": title,
                        "folder": "/".join(path),
                    },
                    tags=["safari", "bookmark"] + path,
                )

            elif web_bookmark_type == "WebBookmarkTypeList":
                folder_title = child.get("Title", "")
                new_path = path + [folder_title] if folder_title else path

                async for dp in self._iterate_folder(child, new_path):
                    yield dp


class ArcHistorySource(DataSource):
    """Adapter for Arc browser history."""

    source_type = "arc_history"
    category = DataCategory.BROWSING

    def __init__(
        self,
        profile_path: str = "~/Library/Application Support/Arc/User Data/Default",
        max_entries: int = 10000,
        **kwargs,
    ):
        """
        Initialize Arc history source.

        Arc is Chromium-based, so uses same history format as Chrome.
        """
        super().__init__(**kwargs)
        self.profile_path = Path(profile_path).expanduser()
        self.max_entries = max_entries

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate over Arc history."""
        # Arc uses same format as Chrome
        chrome_source = ChromeHistorySource(
            profile_path=str(self.profile_path),
            max_entries=self.max_entries,
        )

        async for dp in chrome_source.iterate():
            # Update source type and tags
            dp.source_type = self.source_type
            dp.tags = ["arc", "history", chrome_source._get_domain(dp.metadata.get("url", ""))]
            yield dp
