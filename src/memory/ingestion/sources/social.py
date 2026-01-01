"""
Social media data source adapters.

Supports GDPR data exports from major platforms.
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
)


class TwitterExportSource(ExportBasedSource):
    """
    Twitter/X data from GDPR export.

    Export from: Settings > Your Account > Download an archive of your data
    """

    @property
    def name(self) -> str:
        return "Twitter/X"

    @property
    def category(self) -> DataCategory:
        return DataCategory.SOCIAL_MEDIA

    @property
    def description(self) -> str:
        return "Tweets, likes, replies, and DMs from Twitter/X GDPR export"

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        count = 0
        tweets_file = self._export_path / "data" / "tweets.js"
        if tweets_file.exists():
            content = tweets_file.read_text()
            count += content.count('"tweet"')

        likes_file = self._export_path / "data" / "like.js"
        if likes_file.exists():
            content = likes_file.read_text()
            count += content.count('"like"')

        return count

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        # Parse tweets
        tweets_file = self._export_path / "data" / "tweets.js"
        if tweets_file.exists():
            async for dp in self._parse_tweets(tweets_file):
                yield dp

        # Parse likes
        likes_file = self._export_path / "data" / "like.js"
        if likes_file.exists():
            async for dp in self._parse_likes(likes_file):
                yield dp

    async def _parse_tweets(self, path: Path) -> AsyncIterator[DataPoint]:
        content = path.read_text()
        # Twitter exports start with "window.YTD.tweets.part0 = "
        json_start = content.find("[")
        if json_start == -1:
            return

        data = json.loads(content[json_start:])

        for item in data:
            tweet = item.get("tweet", {})
            yield DataPoint(
                content=tweet.get("full_text", ""),
                category=DataCategory.SOCIAL_MEDIA,
                subcategory="tweet",
                source_type="twitter",
                source_id=tweet.get("id_str"),
                original_date=self._parse_date(tweet.get("created_at")),
                engagement={
                    "retweets": int(tweet.get("retweet_count", 0)),
                    "likes": int(tweet.get("favorite_count", 0)),
                },
                is_public=True,
                raw_data=tweet,
            )

    async def _parse_likes(self, path: Path) -> AsyncIterator[DataPoint]:
        content = path.read_text()
        json_start = content.find("[")
        if json_start == -1:
            return

        data = json.loads(content[json_start:])

        for item in data:
            like = item.get("like", {})
            yield DataPoint(
                content=f"Liked: {like.get('fullText', '')}",
                category=DataCategory.SOCIAL_MEDIA,
                subcategory="like",
                source_type="twitter",
                source_id=like.get("tweetId"),
                importance=0.3,  # Likes are lower importance
                raw_data=like,
            )

    def _parse_date(self, date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            # Twitter format: "Wed Oct 10 20:19:24 +0000 2018"
            return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            return None


class FacebookExportSource(ExportBasedSource):
    """
    Facebook data from Download Your Information.

    Export from: Settings > Your Facebook Information > Download Your Information
    """

    @property
    def name(self) -> str:
        return "Facebook"

    @property
    def category(self) -> DataCategory:
        return DataCategory.SOCIAL_MEDIA

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        count = 0
        posts_dir = self._export_path / "posts"
        if posts_dir.exists():
            for f in posts_dir.glob("*.json"):
                data = json.loads(f.read_text())
                count += len(data) if isinstance(data, list) else 1

        return count

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        # Parse posts
        posts_dir = self._export_path / "posts"
        if posts_dir.exists():
            for f in posts_dir.glob("*.json"):
                data = json.loads(f.read_text())
                posts = data if isinstance(data, list) else [data]
                for post in posts:
                    if "data" in post:
                        for item in post["data"]:
                            if "post" in item:
                                yield DataPoint(
                                    content=item["post"],
                                    category=DataCategory.SOCIAL_MEDIA,
                                    subcategory="post",
                                    source_type="facebook",
                                    original_date=datetime.fromtimestamp(post.get("timestamp", 0)),
                                    is_public=True,
                                )


class LinkedInExportSource(ExportBasedSource):
    """
    LinkedIn data from "Get a copy of your data".

    Export from: Settings > Data privacy > Get a copy of your data
    """

    @property
    def name(self) -> str:
        return "LinkedIn"

    @property
    def category(self) -> DataCategory:
        return DataCategory.LINKEDIN

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        count = 0
        for csv_file in self._export_path.glob("*.csv"):
            count += sum(1 for _ in open(csv_file)) - 1  # Subtract header

        return count

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        import csv

        # Profile
        profile_file = self._export_path / "Profile.csv"
        if profile_file.exists():
            with open(profile_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield DataPoint(
                        content=f"LinkedIn Profile: {row.get('First Name', '')} {row.get('Last Name', '')} - {row.get('Headline', '')}",
                        category=DataCategory.LINKEDIN,
                        subcategory="profile",
                        source_type="linkedin",
                        importance=0.9,
                    )

        # Positions
        positions_file = self._export_path / "Positions.csv"
        if positions_file.exists():
            with open(positions_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield DataPoint(
                        content=f"Position: {row.get('Title', '')} at {row.get('Company Name', '')} ({row.get('Started On', '')} - {row.get('Finished On', 'Present')})",
                        category=DataCategory.WORK_HISTORY,
                        subcategory="position",
                        source_type="linkedin",
                        importance=0.8,
                    )

        # Skills
        skills_file = self._export_path / "Skills.csv"
        if skills_file.exists():
            with open(skills_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield DataPoint(
                        content=f"Skill: {row.get('Name', '')}",
                        category=DataCategory.LINKEDIN,
                        subcategory="skill",
                        source_type="linkedin",
                        importance=0.6,
                    )


class InstagramExportSource(ExportBasedSource):
    """
    Instagram data from Download Data.

    Export from: Settings > Your activity > Download your information
    """

    @property
    def name(self) -> str:
        return "Instagram"

    @property
    def category(self) -> DataCategory:
        return DataCategory.SOCIAL_MEDIA

    async def estimate_count(self) -> int:
        if not await self.is_available():
            return 0

        count = 0
        posts_file = self._export_path / "content" / "posts_1.json"
        if posts_file.exists():
            data = json.loads(posts_file.read_text())
            count += len(data) if isinstance(data, list) else 1

        return count

    async def iterate(self) -> AsyncIterator[DataPoint]:
        if not await self.is_available():
            return

        # Posts
        posts_file = self._export_path / "content" / "posts_1.json"
        if posts_file.exists():
            data = json.loads(posts_file.read_text())
            posts = data if isinstance(data, list) else [data]

            for post in posts:
                media = post.get("media", [{}])[0] if post.get("media") else {}
                yield DataPoint(
                    content=media.get("title", "") or post.get("title", ""),
                    category=DataCategory.SOCIAL_MEDIA,
                    subcategory="post",
                    source_type="instagram",
                    media_path=media.get("uri"),
                    original_date=datetime.fromtimestamp(media.get("creation_timestamp", 0)),
                    is_public=True,
                )
