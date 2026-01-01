"""
Content Processing Pipeline for PLM.

This module handles:
1. Fetching actual webpage content from URLs
2. Extracting and cleaning text content
3. Generating summaries using Ollama
4. Creating embeddings for semantic search
5. Categorizing and tagging content
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura

from memory.schema.memory_entry import (
    ConfidenceScore,
    MemoryEntry,
    MemoryTier,
    MemoryType,
    TruthCategory,
)


@dataclass
class ProcessedContent:
    """Result of processing content."""

    original_url: str | None
    title: str
    content: str
    summary: str
    domains: list[str]
    tags: list[str]
    content_type: str  # article, video, email, document
    word_count: int
    embedding: list[float] | None = None


class OllamaProcessor:
    """Uses Ollama for content processing tasks."""

    def __init__(
        self,
        model: str = "tinyllama",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text[:2000]},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def summarize(self, content: str, max_length: int = 200) -> str:
        """Generate a concise summary of content."""
        prompt = f"""Summarize this content in 2-3 sentences (max {max_length} chars). Focus on the key facts and insights:

{content[:3000]}

Summary:"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 150, "temperature": 0.3},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["response"].strip()[:max_length]

    async def extract_domains(self, content: str) -> list[str]:
        """Extract knowledge domains from content."""
        prompt = f"""What knowledge domains does this content belong to? Choose from:
programming, web_development, cloud, aws, devops, ai, machine_learning,
business, consulting, government, legal, finance, health, education,
drupal, php, python, javascript, database, security, personal, news

Content: {content[:1500]}

Return only the domain names as a comma-separated list (max 5):"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 50, "temperature": 0.1},
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()["response"].strip().lower()
            # Parse comma-separated domains
            domains = [d.strip() for d in result.split(",") if d.strip()]
            return domains[:5]

    async def extract_key_insights(self, content: str, user_name: str = "the user") -> str:
        """Extract key insights relevant to the user from content."""
        prompt = f"""Extract the key insights from this content that would be useful for {user_name}'s personal knowledge base.

Content:
{content[:2500]}

Focus on:
- Factual information
- Technical details
- Actionable insights
- Relevant context

Key insights (be specific and concise):"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 300, "temperature": 0.3},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["response"].strip()


class WebContentFetcher:
    """Fetches and extracts content from web pages."""

    def __init__(self):
        self.processor = OllamaProcessor()

    async def fetch_url(self, url: str) -> ProcessedContent | None:
        """Fetch and process content from a URL."""
        try:
            # Use trafilatura to fetch and extract content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None

            # Extract main content
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )

            if not content or len(content) < 100:
                return None

            # Get metadata
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata else urlparse(url).netloc

            # Generate summary
            summary = await self.processor.summarize(content)

            # Extract domains
            domains = await self.processor.extract_domains(content)

            # Generate tags from URL and content
            tags = self._generate_tags(url, title, content)

            # Generate embedding
            embedding = await self.processor.get_embedding(f"{title}\n{summary}")

            return ProcessedContent(
                original_url=url,
                title=title or "Untitled",
                content=content,
                summary=summary,
                domains=domains,
                tags=tags,
                content_type="article",
                word_count=len(content.split()),
                embedding=embedding,
            )

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _generate_tags(self, url: str, title: str, content: str) -> list[str]:
        """Generate tags from URL and content."""
        tags = ["web_content"]

        # Add domain-based tags
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if "github" in domain:
            tags.append("github")
        elif "stackoverflow" in domain:
            tags.append("stackoverflow")
        elif "aws.amazon" in domain:
            tags.append("aws")
        elif "drupal" in domain:
            tags.append("drupal")
        elif "youtube" in domain or "youtu.be" in domain:
            tags.append("video")

        # Add content-based tags
        content_lower = content.lower()
        tech_keywords = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
            "drupal": "drupal",
            "php": "php",
            "aws": "aws",
            "docker": "docker",
            "kubernetes": "kubernetes",
            "api": "api",
            "database": "database",
        }

        for keyword, tag in tech_keywords.items():
            if keyword in content_lower:
                tags.append(tag)

        return list(set(tags))[:10]


class ContentIngestionPipeline:
    """
    Main pipeline for ingesting and processing content into the PLM.

    Handles:
    - Browser history URL processing
    - Email content enrichment
    - Document summarization
    - Video transcription (YouTube)
    """

    def __init__(self, storage_manager=None):
        self.fetcher = WebContentFetcher()
        self.processor = OllamaProcessor()
        self.storage = storage_manager

    async def process_browser_history(
        self,
        history_items: list[dict[str, Any]],
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """
        Process browser history items by fetching actual page content.

        Args:
            history_items: List of dicts with 'url', 'title', 'visit_time'
            limit: Maximum items to process
        """
        memories = []
        processed = 0

        for item in history_items[:limit]:
            url = item.get("url", "")

            # Skip non-content URLs
            if not self._is_content_url(url):
                continue

            print(f"Processing: {url[:60]}...")

            content = await self.fetcher.fetch_url(url)
            if not content:
                continue

            # Create memory entry
            memory = MemoryEntry(
                content=content.content[:5000],
                summary=content.summary,
                memory_type=MemoryType.FACT,
                tier=MemoryTier.LONG_TERM,  # Browser content goes to long-term
                truth_category=TruthCategory.CONTEXTUAL,
                domains=content.domains,
                tags=content.tags + ["browser_history"],
                confidence=ConfidenceScore(
                    overall=0.7,
                    source_reliability=0.8,
                    recency=0.9,
                ),
            )

            if self.storage:
                await self.storage.store(memory)

            memories.append(memory)
            processed += 1

            if processed % 10 == 0:
                print(f"  Processed {processed} pages...")

            # Rate limiting
            await asyncio.sleep(0.5)

        return memories

    async def enrich_email_memories(
        self,
        email_memories: list[MemoryEntry],
    ) -> list[MemoryEntry]:
        """
        Enrich existing email memories with summaries and insights.
        """
        enriched = []

        for memory in email_memories:
            if memory.summary and len(memory.summary) > 50:
                # Already has good summary
                enriched.append(memory)
                continue

            # Generate summary
            summary = await self.processor.summarize(memory.content)
            memory.summary = summary

            # Extract domains if missing
            if not memory.domains:
                memory.domains = await self.processor.extract_domains(memory.content)

            # Generate embedding
            embedding_text = f"{summary}\n{memory.content[:500]}"
            memory.embedding = await self.processor.get_embedding(embedding_text)

            enriched.append(memory)

        return enriched

    async def process_youtube_video(self, url: str) -> MemoryEntry | None:
        """
        Process a YouTube video - extract metadata and potentially transcript.
        """
        # Extract video ID
        video_id = self._extract_youtube_id(url)
        if not video_id:
            return None

        try:
            # Fetch video page for metadata
            content = await self.fetcher.fetch_url(url)
            if not content:
                return None

            # Mark as video content
            content.content_type = "video"
            content.tags.append("youtube")

            memory = MemoryEntry(
                content=f"Video: {content.title}\n\n{content.summary}",
                summary=f"YouTube video: {content.title}",
                memory_type=MemoryType.FACT,
                tier=MemoryTier.LONG_TERM,
                truth_category=TruthCategory.CONTEXTUAL,
                domains=content.domains,
                tags=content.tags,
                confidence=ConfidenceScore(overall=0.7),
            )

            return memory

        except Exception as e:
            print(f"Error processing video {url}: {e}")
            return None

    def _is_content_url(self, url: str) -> bool:
        """Check if URL is likely to have meaningful content."""
        if not url.startswith(("http://", "https://")):
            return False

        # Skip common non-content URLs
        skip_patterns = [
            "google.com/search",
            "google.com/maps",
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "linkedin.com/feed",
            "mail.google.com",
            "calendar.google.com",
            "drive.google.com",
            "docs.google.com",
            "localhost",
            "127.0.0.1",
            ".pdf",  # Handle separately
            ".jpg",
            ".png",
            ".gif",  # Images
        ]

        url_lower = url.lower()
        return not any(pattern in url_lower for pattern in skip_patterns)

    def _extract_youtube_id(self, url: str) -> str | None:
        """Extract YouTube video ID from URL."""
        patterns = [
            r"youtube\.com/watch\?v=([^&]+)",
            r"youtu\.be/([^?]+)",
            r"youtube\.com/embed/([^?]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None


async def run_content_enrichment(base_path: str = "~/memory/data"):
    """
    Run content enrichment on existing memories.

    This processes existing browser history and email memories
    to add summaries, embeddings, and better categorization.
    """
    from memory.storage.manager import StorageManager

    storage = StorageManager(base_path=base_path)
    await storage.initialize()

    pipeline = ContentIngestionPipeline(storage)

    # Get browser history URLs from existing memories
    print("Fetching browser history memories...")

    # Query for browser history that hasn't been enriched
    memories, _ = await storage.text_search("browser_history", limit=100)

    urls_to_process = []
    for mem in memories:
        # Extract URL from content if present
        if mem.content.startswith("http"):
            urls_to_process.append({"url": mem.content.split("\n")[0]})

    if urls_to_process:
        print(f"Processing {len(urls_to_process)} browser history URLs...")
        await pipeline.process_browser_history(urls_to_process, limit=50)

    # Enrich email memories without summaries
    print("\nEnriching email memories...")
    email_memories, _ = await storage.text_search("gmail", limit=100)
    no_summary = [m for m in email_memories if not m.summary or len(m.summary) < 30]

    if no_summary:
        print(f"Enriching {len(no_summary)} emails without summaries...")
        await pipeline.enrich_email_memories(no_summary[:50])

    await storage.close()
    print("\nContent enrichment complete!")
