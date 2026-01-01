"""
Media data source adapters.

Photos, videos, and audio with metadata and content extraction.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from memory.ingestion.sources.base import (
    DataCategory,
    DataPoint,
    ExportBasedSource,
    FileBasedSource,
)


class ApplePhotosSource(FileBasedSource):
    """
    Apple Photos library.

    Extracts photos with metadata, location, captions, and optionally OCR.
    """

    def __init__(self, library_path: str | Path | None = None):
        path = library_path or "~/Pictures/Photos Library.photoslibrary"
        super().__init__(path)

    @property
    def name(self) -> str:
        return "Apple Photos"

    @property
    def category(self) -> DataCategory:
        return DataCategory.PHOTO

    @property
    def description(self) -> str:
        return "Photos with metadata, location, captions from Apple Photos"

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        # Check database for count
        db_path = self._base_path / "database" / "Photos.sqlite"
        if db_path.exists():
            try:
                import sqlite3

                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute("SELECT COUNT(*) FROM ZASSET")
                count = cursor.fetchone()[0]
                conn.close()
                return count
            except Exception:
                pass

        return 0

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        db_path = self._base_path / "database" / "Photos.sqlite"
        if not db_path.exists():
            return

        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))

            # Query photos with metadata
            query = """
                SELECT
                    ZASSET.ZUUID,
                    ZASSET.ZFILENAME,
                    ZASSET.ZDATECREATED,
                    ZASSET.ZLATITUDE,
                    ZASSET.ZLONGITUDE,
                    ZADDITIONALASSETATTRIBUTES.ZTITLE,
                    ZADDITIONALASSETATTRIBUTES.ZCAPTION
                FROM ZASSET
                LEFT JOIN ZADDITIONALASSETATTRIBUTES
                    ON ZASSET.Z_PK = ZADDITIONALASSETATTRIBUTES.ZASSET
                LIMIT 10000
            """

            cursor = conn.execute(query)

            for row in cursor:
                uuid, filename, date_created, lat, lon, title, caption = row

                content_parts = []
                if title:
                    content_parts.append(f"Title: {title}")
                if caption:
                    content_parts.append(f"Caption: {caption}")
                if not content_parts:
                    content_parts.append(f"Photo: {filename}")

                location = None
                if lat and lon:
                    location = f"{lat}, {lon}"

                yield DataPoint(
                    content=" | ".join(content_parts),
                    category=DataCategory.PHOTO,
                    source_type="apple_photos",
                    source_id=uuid,
                    source_path=filename,
                    original_date=self._cocoa_to_datetime(date_created) if date_created else None,
                    location=location,
                    importance=0.5,
                )

            conn.close()

        except Exception:
            pass

    def _cocoa_to_datetime(self, cocoa_timestamp: float) -> datetime:
        """Convert Cocoa timestamp to datetime."""
        # Cocoa epoch is 2001-01-01
        import datetime as dt

        cocoa_epoch = dt.datetime(2001, 1, 1)
        return cocoa_epoch + dt.timedelta(seconds=cocoa_timestamp)


class GooglePhotosExportSource(ExportBasedSource):
    """
    Google Photos from Takeout export.

    Export from: takeout.google.com > Google Photos
    """

    @property
    def name(self) -> str:
        return "Google Photos"

    @property
    def category(self) -> DataCategory:
        return DataCategory.PHOTO

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        count = 0
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.heic", "*.mp4", "*.mov"]:
            count += len(list(self._export_path.rglob(ext)))

        return count

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        # Find all media files with their JSON metadata
        for json_file in self._export_path.rglob("*.json"):
            try:
                data = json.loads(json_file.read_text())

                # Skip non-photo metadata
                if "title" not in data:
                    continue

                content_parts = [f"Photo: {data.get('title', '')}"]

                if data.get("description"):
                    content_parts.append(f"Description: {data['description']}")

                location = None
                geo = data.get("geoData", {})
                if geo.get("latitude") and geo.get("longitude"):
                    location = f"{geo['latitude']}, {geo['longitude']}"

                timestamp = data.get("photoTakenTime", {}).get("timestamp")
                original_date = datetime.fromtimestamp(int(timestamp)) if timestamp else None

                yield DataPoint(
                    content=" | ".join(content_parts),
                    category=DataCategory.PHOTO,
                    source_type="google_photos",
                    source_path=str(json_file.with_suffix("")),
                    original_date=original_date,
                    location=location,
                    raw_data=data,
                )

            except (json.JSONDecodeError, KeyError):
                continue


class YouTubeHistorySource(ExportBasedSource):
    """
    YouTube watch history and interactions from Takeout.

    Export from: takeout.google.com > YouTube
    """

    @property
    def name(self) -> str:
        return "YouTube"

    @property
    def category(self) -> DataCategory:
        return DataCategory.VIDEO

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        watch_history = self._export_path / "history" / "watch-history.json"
        if watch_history.exists():
            data = json.loads(watch_history.read_text())
            return len(data)

        return 0

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        # Watch history
        watch_history = self._export_path / "history" / "watch-history.json"
        if watch_history.exists():
            data = json.loads(watch_history.read_text())

            for item in data:
                yield DataPoint(
                    content=f"Watched: {item.get('title', '')}",
                    category=DataCategory.VIDEO,
                    subcategory="watch_history",
                    source_type="youtube",
                    source_path=item.get("titleUrl"),
                    original_date=self._parse_time(item.get("time")),
                    importance=0.3,
                    raw_data=item,
                )

        # Liked videos
        likes_file = self._export_path / "playlists" / "Liked videos.json"
        if likes_file.exists():
            data = json.loads(likes_file.read_text())

            for item in data:
                yield DataPoint(
                    content=f"Liked video: {item.get('snippet', {}).get('title', '')}",
                    category=DataCategory.VIDEO,
                    subcategory="like",
                    source_type="youtube",
                    importance=0.5,
                    raw_data=item,
                )

    def _parse_time(self, time_str: str | None) -> datetime | None:
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except ValueError:
            return None


class SpotifyExportSource(ExportBasedSource):
    """
    Spotify listening history from data export.

    Export from: Privacy settings > Download your data
    """

    @property
    def name(self) -> str:
        return "Spotify"

    @property
    def category(self) -> DataCategory:
        return DataCategory.MUSIC

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        count = 0
        for f in self._export_path.glob("StreamingHistory*.json"):
            data = json.loads(f.read_text())
            count += len(data)

        return count

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        for f in self._export_path.glob("StreamingHistory*.json"):
            data = json.loads(f.read_text())

            for item in data:
                yield DataPoint(
                    content=f"Listened to: {item.get('trackName', '')} by {item.get('artistName', '')}",
                    category=DataCategory.MUSIC,
                    subcategory="stream",
                    source_type="spotify",
                    original_date=self._parse_time(item.get("endTime")),
                    importance=0.2,  # Individual plays are low importance
                    raw_data=item,
                )

    def _parse_time(self, time_str: str | None) -> datetime | None:
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(time_str)
        except ValueError:
            return None
