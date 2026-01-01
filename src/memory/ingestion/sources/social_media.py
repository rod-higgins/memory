"""
Social Media Data Source Adapters.

Parses GDPR data exports from:
- Twitter/X
- Facebook
- LinkedIn

These platforms provide data exports that users can request.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.schema import MemoryEntry, MemoryType


@dataclass
class SocialPost:
    """A social media post."""

    platform: str
    content: str
    timestamp: datetime
    post_type: str  # tweet, post, comment, message, etc.
    metadata: dict[str, Any] | None = None


class TwitterExportParser:
    """Parse Twitter/X data export."""

    def __init__(self, export_path: str):
        self.export_path = Path(export_path).expanduser()

    async def parse(self, limit: int = 500) -> list[MemoryEntry]:
        """Parse Twitter export and return memories."""
        memories = []

        if not self.export_path.exists():
            return []

        # Handle both zip and extracted folder
        if self.export_path.suffix == ".zip":
            memories = await self._parse_zip(limit)
        else:
            memories = await self._parse_folder(limit)

        return memories

    async def _parse_zip(self, limit: int) -> list[MemoryEntry]:
        """Parse a Twitter zip export."""
        memories = []

        with zipfile.ZipFile(self.export_path) as zf:
            # Look for tweets.js or tweet.js
            tweet_files = [n for n in zf.namelist() if "tweet" in n.lower() and n.endswith(".js")]

            for tweet_file in tweet_files:
                content = zf.read(tweet_file).decode("utf-8")
                memories.extend(self._parse_tweets_js(content, limit))
                if len(memories) >= limit:
                    break

        return memories[:limit]

    async def _parse_folder(self, limit: int) -> list[MemoryEntry]:
        """Parse an extracted Twitter export folder."""
        memories = []

        # Look for data/tweets.js or similar
        for js_file in self.export_path.rglob("*.js"):
            if "tweet" in js_file.name.lower():
                content = js_file.read_text()
                memories.extend(self._parse_tweets_js(content, limit))
                if len(memories) >= limit:
                    break

        return memories[:limit]

    def _parse_tweets_js(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse tweets.js content."""
        memories = []

        # Twitter exports have format: window.YTD.tweet.part0 = [...]
        try:
            # Find the JSON array
            start = content.find("[")
            if start == -1:
                return []

            json_str = content[start:]
            tweets = json.loads(json_str)

            for tweet_data in tweets[:limit]:
                tweet = tweet_data.get("tweet", tweet_data)

                text = tweet.get("full_text", tweet.get("text", ""))
                if not text:
                    continue

                # Parse timestamp
                created_at = tweet.get("created_at", "")
                try:
                    timestamp = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                except (ValueError, TypeError):
                    timestamp = datetime.now()

                # Determine if it's a reply or retweet
                is_reply = tweet.get("in_reply_to_user_id") is not None
                is_retweet = text.startswith("RT @")

                if is_retweet:
                    post_type = "retweet"
                elif is_reply:
                    post_type = "reply"
                else:
                    post_type = "tweet"

                memory = MemoryEntry(
                    content=text,
                    summary=f"Twitter {post_type}: {text[:100]}...",
                    memory_type=MemoryType.INTERACTION,
                    domains=["social_media", "twitter"],
                    tags=["twitter", post_type, "social"],
                    source_id=f"twitter:{tweet.get('id_str', '')}",
                    created_at=timestamp,
                )
                memories.append(memory)

        except json.JSONDecodeError:
            pass

        return memories


class FacebookExportParser:
    """Parse Facebook data export."""

    def __init__(self, export_path: str):
        self.export_path = Path(export_path).expanduser()

    async def parse(self, limit: int = 500) -> list[MemoryEntry]:
        """Parse Facebook export and return memories."""
        memories = []

        if not self.export_path.exists():
            return []

        # Handle zip or folder
        if self.export_path.suffix == ".zip":
            memories = await self._parse_zip(limit)
        else:
            memories = await self._parse_folder(limit)

        return memories

    async def _parse_zip(self, limit: int) -> list[MemoryEntry]:
        """Parse a Facebook zip export."""
        memories = []

        with zipfile.ZipFile(self.export_path) as zf:
            # Look for posts, comments, messages
            for name in zf.namelist():
                if len(memories) >= limit:
                    break

                if name.endswith(".json"):
                    if "posts" in name.lower():
                        content = zf.read(name).decode("utf-8")
                        memories.extend(self._parse_posts_json(content, limit - len(memories)))
                    elif "comments" in name.lower():
                        content = zf.read(name).decode("utf-8")
                        memories.extend(self._parse_comments_json(content, limit - len(memories)))
                    elif "messages" in name.lower():
                        content = zf.read(name).decode("utf-8")
                        memories.extend(self._parse_messages_json(content, limit - len(memories)))

        return memories[:limit]

    async def _parse_folder(self, limit: int) -> list[MemoryEntry]:
        """Parse an extracted Facebook export folder."""
        memories = []

        for json_file in self.export_path.rglob("*.json"):
            if len(memories) >= limit:
                break

            content = json_file.read_text(encoding="utf-8")
            name = json_file.name.lower()

            if "posts" in name or "your_posts" in str(json_file).lower():
                memories.extend(self._parse_posts_json(content, limit - len(memories)))
            elif "comments" in name:
                memories.extend(self._parse_comments_json(content, limit - len(memories)))
            elif "messages" in str(json_file).lower():
                memories.extend(self._parse_messages_json(content, limit - len(memories)))

        return memories[:limit]

    def _parse_posts_json(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse Facebook posts JSON."""
        memories = []

        try:
            data = json.loads(content)
            posts = data if isinstance(data, list) else data.get("posts", data.get("data", []))

            for post in posts[:limit]:
                # Facebook encodes text in a specific way
                text = ""
                if "data" in post:
                    for item in post.get("data", []):
                        if "post" in item:
                            text = item["post"]
                            break

                if not text and "message" in post:
                    text = post["message"]

                if not text:
                    continue

                # Decode Facebook's encoding
                text = self._decode_facebook_text(text)

                timestamp = datetime.fromtimestamp(post.get("timestamp", 0))

                memory = MemoryEntry(
                    content=text,
                    summary=f"Facebook post: {text[:100]}...",
                    memory_type=MemoryType.INTERACTION,
                    domains=["social_media", "facebook"],
                    tags=["facebook", "post", "social"],
                    source_id=f"facebook:post:{post.get('timestamp', '')}",
                    created_at=timestamp,
                )
                memories.append(memory)

        except (json.JSONDecodeError, KeyError):
            pass

        return memories

    def _parse_comments_json(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse Facebook comments JSON."""
        memories = []

        try:
            data = json.loads(content)
            comments = data.get("comments_v2", data.get("comments", data if isinstance(data, list) else []))

            for comment in comments[:limit]:
                text = comment.get("comment", {}).get("comment", "")
                if not text:
                    text = comment.get("data", [{}])[0].get("comment", {}).get("comment", "")

                if not text:
                    continue

                text = self._decode_facebook_text(text)
                timestamp = datetime.fromtimestamp(comment.get("timestamp", 0))

                memory = MemoryEntry(
                    content=text,
                    summary=f"Facebook comment: {text[:100]}...",
                    memory_type=MemoryType.INTERACTION,
                    domains=["social_media", "facebook"],
                    tags=["facebook", "comment", "social"],
                    source_id=f"facebook:comment:{comment.get('timestamp', '')}",
                    created_at=timestamp,
                )
                memories.append(memory)

        except (json.JSONDecodeError, KeyError):
            pass

        return memories

    def _parse_messages_json(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse Facebook messages JSON."""
        memories = []

        try:
            data = json.loads(content)
            messages = data.get("messages", [])

            # Only get messages you sent
            for msg in messages[: limit * 2]:
                if msg.get("sender_name", "").lower() == "you":
                    continue  # Skip, we want our own messages

                text = msg.get("content", "")
                if not text:
                    continue

                text = self._decode_facebook_text(text)
                timestamp = datetime.fromtimestamp(msg.get("timestamp_ms", 0) / 1000)

                memory = MemoryEntry(
                    content=text,
                    summary=f"Facebook message: {text[:100]}...",
                    memory_type=MemoryType.INTERACTION,
                    domains=["social_media", "facebook", "messaging"],
                    tags=["facebook", "message", "social"],
                    source_id=f"facebook:msg:{msg.get('timestamp_ms', '')}",
                    created_at=timestamp,
                )
                memories.append(memory)

                if len(memories) >= limit:
                    break

        except (json.JSONDecodeError, KeyError):
            pass

        return memories

    def _decode_facebook_text(self, text: str) -> str:
        """Decode Facebook's text encoding."""
        try:
            # Facebook exports use latin-1 encoded as UTF-8
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text


class LinkedInExportParser:
    """Parse LinkedIn data export."""

    def __init__(self, export_path: str):
        self.export_path = Path(export_path).expanduser()

    async def parse(self, limit: int = 500) -> list[MemoryEntry]:
        """Parse LinkedIn export and return memories."""
        memories = []

        if not self.export_path.exists():
            return []

        # Handle zip or folder
        if self.export_path.suffix == ".zip":
            memories = await self._parse_zip(limit)
        else:
            memories = await self._parse_folder(limit)

        return memories

    async def _parse_zip(self, limit: int) -> list[MemoryEntry]:
        """Parse a LinkedIn zip export."""
        memories = []

        with zipfile.ZipFile(self.export_path) as zf:
            for name in zf.namelist():
                if len(memories) >= limit:
                    break

                if name.endswith(".csv"):
                    content = zf.read(name).decode("utf-8")
                    if "profile" in name.lower():
                        memories.extend(self._parse_profile_csv(content))
                    elif "positions" in name.lower() or "experience" in name.lower():
                        memories.extend(self._parse_positions_csv(content, limit - len(memories)))
                    elif "skills" in name.lower():
                        memories.extend(self._parse_skills_csv(content, limit - len(memories)))
                    elif "messages" in name.lower():
                        memories.extend(self._parse_messages_csv(content, limit - len(memories)))
                    elif "connections" in name.lower():
                        memories.extend(self._parse_connections_csv(content, limit - len(memories)))

        return memories[:limit]

    async def _parse_folder(self, limit: int) -> list[MemoryEntry]:
        """Parse an extracted LinkedIn export folder."""
        memories = []

        for csv_file in self.export_path.rglob("*.csv"):
            if len(memories) >= limit:
                break

            content = csv_file.read_text(encoding="utf-8")
            name = csv_file.name.lower()

            if "profile" in name:
                memories.extend(self._parse_profile_csv(content))
            elif "positions" in name or "experience" in name:
                memories.extend(self._parse_positions_csv(content, limit - len(memories)))
            elif "skills" in name:
                memories.extend(self._parse_skills_csv(content, limit - len(memories)))
            elif "messages" in name:
                memories.extend(self._parse_messages_csv(content, limit - len(memories)))
            elif "connections" in name:
                memories.extend(self._parse_connections_csv(content, limit - len(memories)))

        return memories[:limit]

    def _parse_csv_rows(self, content: str) -> list[dict[str, str]]:
        """Parse CSV content into rows."""
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(content))
        return list(reader)

    def _parse_profile_csv(self, content: str) -> list[MemoryEntry]:
        """Parse LinkedIn profile CSV."""
        memories = []

        try:
            rows = self._parse_csv_rows(content)
            if rows:
                row = rows[0]
                name = f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip()
                headline = row.get("Headline", "")
                summary = row.get("Summary", "")

                if headline or summary:
                    memory = MemoryEntry(
                        content=f"LinkedIn Profile: {name}\nHeadline: {headline}\nSummary: {summary}",
                        summary=f"My LinkedIn profile: {headline[:100]}",
                        memory_type=MemoryType.FACT,
                        domains=["professional", "linkedin", "identity"],
                        tags=["linkedin", "profile", "professional"],
                        source_id="linkedin:profile",
                    )
                    memories.append(memory)

        except Exception:
            pass

        return memories

    def _parse_positions_csv(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse LinkedIn positions/experience CSV."""
        memories = []

        try:
            rows = self._parse_csv_rows(content)

            for row in rows[:limit]:
                company = row.get("Company Name", "")
                title = row.get("Title", "")
                description = row.get("Description", "")
                start_date = row.get("Started On", "")
                end_date = row.get("Finished On", "Present")

                if company and title:
                    text = f"Position: {title} at {company} ({start_date} - {end_date})"
                    if description:
                        text += f"\n{description}"

                    memory = MemoryEntry(
                        content=text,
                        summary=f"Worked as {title} at {company}",
                        memory_type=MemoryType.EXPERIENCE,
                        domains=["professional", "linkedin", "career"],
                        tags=["linkedin", "experience", "job", company.lower().replace(" ", "_")],
                        source_id=f"linkedin:position:{company}:{title}",
                    )
                    memories.append(memory)

        except Exception:
            pass

        return memories

    def _parse_skills_csv(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse LinkedIn skills CSV."""
        memories = []

        try:
            rows = self._parse_csv_rows(content)
            skills = [row.get("Name", "") for row in rows if row.get("Name")]

            if skills:
                memory = MemoryEntry(
                    content=f"LinkedIn Skills: {', '.join(skills[:50])}",
                    summary=f"Professional skills: {', '.join(skills[:10])}...",
                    memory_type=MemoryType.SKILL,
                    domains=["professional", "linkedin", "skills"],
                    tags=["linkedin", "skills"] + [s.lower().replace(" ", "_") for s in skills[:20]],
                    source_id="linkedin:skills",
                )
                memories.append(memory)

        except Exception:
            pass

        return memories

    def _parse_messages_csv(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse LinkedIn messages CSV."""
        memories = []

        try:
            rows = self._parse_csv_rows(content)

            for row in rows[:limit]:
                sender = row.get("FROM", "")
                msg_content = row.get("CONTENT", "")
                date = row.get("DATE", "")

                if msg_content and len(msg_content) > 20:
                    memory = MemoryEntry(
                        content=f"LinkedIn message from {sender}: {msg_content}",
                        summary=f"LinkedIn message: {msg_content[:100]}...",
                        memory_type=MemoryType.INTERACTION,
                        domains=["professional", "linkedin", "messaging"],
                        tags=["linkedin", "message", "professional"],
                        source_id=f"linkedin:msg:{date}",
                    )
                    memories.append(memory)

        except Exception:
            pass

        return memories

    def _parse_connections_csv(self, content: str, limit: int) -> list[MemoryEntry]:
        """Parse LinkedIn connections CSV."""
        memories = []

        try:
            rows = self._parse_csv_rows(content)

            # Create a summary of connections by company
            companies: dict[str, int] = {}
            for row in rows:
                company = row.get("Company", "")
                if company:
                    companies[company] = companies.get(company, 0) + 1

            top_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)[:20]

            if top_companies:
                memory = MemoryEntry(
                    content=f"LinkedIn network: {len(rows)} connections. Top companies: {', '.join(f'{c} ({n})' for c, n in top_companies)}",
                    summary=f"LinkedIn network of {len(rows)} professional connections",
                    memory_type=MemoryType.FACT,
                    domains=["professional", "linkedin", "network"],
                    tags=["linkedin", "connections", "network"],
                    source_id="linkedin:connections",
                )
                memories.append(memory)

        except Exception:
            pass

        return memories


# Export adapter functions
async def ingest_twitter(export_path: str, limit: int = 500) -> list[MemoryEntry]:
    """Ingest Twitter export."""
    parser = TwitterExportParser(export_path)
    return await parser.parse(limit)


async def ingest_facebook(export_path: str, limit: int = 500) -> list[MemoryEntry]:
    """Ingest Facebook export."""
    parser = FacebookExportParser(export_path)
    return await parser.parse(limit)


async def ingest_linkedin(export_path: str, limit: int = 500) -> list[MemoryEntry]:
    """Ingest LinkedIn export."""
    parser = LinkedInExportParser(export_path)
    return await parser.parse(limit)
