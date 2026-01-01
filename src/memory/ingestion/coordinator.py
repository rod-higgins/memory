"""
Ingestion coordinator - orchestrates data ingestion from all sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.ingestion.enrichment import EnrichmentPipeline, SimpleEnrichmentPipeline
from memory.ingestion.parsers.claude_history import ClaudeHistoryParser
from memory.schema.memory_entry import MemoryEntry, MemoryTier
from memory.storage.manager import StorageManager


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""

    source: str
    total_processed: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "total_processed": self.total_processed,
            "duplicates_skipped": self.duplicates_skipped,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
        }


class IngestionCoordinator:
    """
    Coordinates ingestion from all sources.

    Handles:
    - Source parsing
    - Enrichment
    - Deduplication
    - Storage
    """

    def __init__(
        self,
        storage: StorageManager,
        enrichment: EnrichmentPipeline | SimpleEnrichmentPipeline | None = None,
    ):
        self._storage = storage
        self._enrichment = enrichment or SimpleEnrichmentPipeline()
        self._stats: dict[str, IngestionStats] = {}

    async def ingest_claude_history(
        self,
        history_path: str | Path = "~/.claude/history.jsonl",
        max_entries: int | None = None,
        batch_size: int = 32,
        skip_enrichment: bool = False,
    ) -> IngestionStats:
        """
        Ingest memories from Claude Code history.

        Args:
            history_path: Path to history.jsonl
            max_entries: Maximum entries to process (None = all)
            batch_size: Batch size for enrichment
            skip_enrichment: Skip embedding generation for faster testing

        Returns:
            Ingestion statistics
        """
        stats = IngestionStats(source="claude_history")
        parser = ClaudeHistoryParser(history_path)

        batch: list[MemoryEntry] = []

        async for memory in parser.parse(max_entries=max_entries):
            try:
                # Check for duplicates
                existing = await self._storage.find_by_hash(memory.content_hash)
                if existing:
                    stats.duplicates_skipped += 1
                    continue

                batch.append(memory)

                # Process batch when full
                if len(batch) >= batch_size:
                    await self._process_batch(batch, skip_enrichment)
                    stats.total_processed += len(batch)
                    batch = []

            except Exception:
                stats.errors += 1
                continue

        # Process remaining batch
        if batch:
            await self._process_batch(batch, skip_enrichment)
            stats.total_processed += len(batch)

        stats.end_time = datetime.now()
        self._stats["claude_history"] = stats
        return stats

    async def _process_batch(
        self,
        memories: list[MemoryEntry],
        skip_enrichment: bool = False,
    ) -> None:
        """Process a batch of memories."""
        # Enrich if not skipped
        if not skip_enrichment:
            if isinstance(self._enrichment, SimpleEnrichmentPipeline):
                memories = await self._enrichment.enrich_batch(memories)
            else:
                memories = await self._enrichment.enrich_batch(
                    memories,
                    use_slm_categorization=False,  # Skip SLM for speed
                )

        # Store in short-term tier
        for memory in memories:
            memory.tier = MemoryTier.SHORT_TERM
            await self._storage.store(memory)

    def get_stats(self, source: str | None = None) -> dict[str, Any]:
        """Get ingestion statistics."""
        if source:
            stats = self._stats.get(source)
            return stats.to_dict() if stats else {}

        return {name: s.to_dict() for name, s in self._stats.items()}

    async def get_source_info(self) -> dict[str, Any]:
        """Get information about available sources."""
        claude_parser = ClaudeHistoryParser()

        info: dict[str, Any] = {
            "claude_history": {
                "path": str(claude_parser._history_path),
                "exists": claude_parser._history_path.exists(),
            }
        }

        if claude_parser._history_path.exists():
            count = await claude_parser.count_entries()
            first, last = await claude_parser.get_date_range()

            info["claude_history"].update(
                {
                    "entry_count": count,
                    "first_entry": first.isoformat() if first else None,
                    "last_entry": last.isoformat() if last else None,
                }
            )

        return info
