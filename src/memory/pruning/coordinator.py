"""
Pruning coordinator for memory quality management.

Orchestrates the analysis, filtering, and pruning of memories
to maintain a high-quality memory store.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.pruning.analyzer import AnalysisResult, ContentAnalyzer


@dataclass
class PruningStats:
    """Statistics from a pruning run."""

    total_analyzed: int = 0
    total_pruned: int = 0
    total_archived: int = 0
    total_kept: int = 0
    by_filter: dict[str, int] | None = None
    duration_seconds: float = 0.0
    timestamp: str = ""


class PruningCoordinator:
    """
    Coordinates memory pruning operations.

    - Analyzes memories for quality
    - Applies pruning rules
    - Maintains audit log
    - Supports dry-run mode
    """

    def __init__(
        self,
        db_path: str | Path,
        analyzer: ContentAnalyzer | None = None,
        batch_size: int = 100,
    ):
        self.db_path = Path(db_path).expanduser()
        self.analyzer = analyzer or ContentAnalyzer()
        self.batch_size = batch_size
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def analyze_all(
        self,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[AnalysisResult]:
        """Analyze all memories and return results."""
        if not self._conn:
            raise RuntimeError("Coordinator not initialized")

        query = "SELECT * FROM memories WHERE is_active = 1"
        params: list[Any] = []

        if filters:
            if "domain" in filters:
                query += " AND domains_json LIKE ?"
                params.append(f'%{filters["domain"]}%')

        query += " ORDER BY created_at DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            result = self.analyzer.analyze(
                content=row["content"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                memory_id=row["id"],
                tags=json.loads(row["tags_json"] or "[]"),
                domains=json.loads(row["domains_json"] or "[]"),
            )
            results.append(result)

        return results

    async def prune(
        self,
        dry_run: bool = True,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> PruningStats:
        """
        Run pruning operation.

        Args:
            dry_run: If True, only analyze without making changes
            limit: Maximum number of memories to process
            filters: Additional filters for memory selection
        """
        start_time = datetime.now()
        stats = PruningStats(
            by_filter={},
            timestamp=start_time.isoformat(),
        )

        # Analyze memories
        results = await self.analyze_all(limit=limit, filters=filters)
        stats.total_analyzed = len(results)

        for result in results:
            if result.should_prune:
                if result.suggested_action == "prune":
                    stats.total_pruned += 1
                    if not dry_run:
                        await self._mark_inactive(result.memory_id)
                else:
                    stats.total_archived += 1
                    if not dry_run:
                        await self._archive(result.memory_id)

                # Track by filter
                for fr in result.filter_results:
                    if fr.should_prune:
                        stats.by_filter[fr.filter_name] = (
                            stats.by_filter.get(fr.filter_name, 0) + 1
                        )
            else:
                stats.total_kept += 1

        if not dry_run and self._conn:
            self._conn.commit()

        stats.duration_seconds = (datetime.now() - start_time).total_seconds()
        return stats

    async def _mark_inactive(self, memory_id: str) -> None:
        """Mark a memory as inactive (soft delete)."""
        if self._conn:
            self._conn.execute(
                "UPDATE memories SET is_active = 0, updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), memory_id)
            )

    async def _archive(self, memory_id: str) -> None:
        """Archive a memory (keep but reduce confidence)."""
        if self._conn:
            self._conn.execute(
                """UPDATE memories
                   SET confidence_overall = confidence_overall * 0.5,
                       metadata_json = json_set(COALESCE(metadata_json, '{}'), '$.archived', true),
                       updated_at = ?
                   WHERE id = ?""",
                (datetime.now().isoformat(), memory_id)
            )

    async def get_prune_preview(
        self,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get a preview of what would be pruned.

        Returns list of memories with their analysis results.
        """
        results = await self.analyze_all(limit=limit, filters=filters)

        preview = []
        for r in results:
            if r.should_prune:
                if not self._conn:
                    continue

                cursor = self._conn.execute(
                    "SELECT id, summary, content, domains_json, tags_json FROM memories WHERE id = ?",
                    (r.memory_id,)
                )
                row = cursor.fetchone()
                if row:
                    preview.append({
                        "id": row["id"],
                        "summary": row["summary"],
                        "content_preview": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                        "domains": json.loads(row["domains_json"] or "[]"),
                        "tags": json.loads(row["tags_json"] or "[]"),
                        "quality_score": r.quality_score,
                        "relevance_score": r.relevance_score,
                        "prune_reasons": r.prune_reasons,
                        "positive_signals": r.positive_signals,
                        "suggested_action": r.suggested_action,
                        "confidence": r.confidence,
                    })

        return preview

    async def get_quality_report(
        self,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Generate a quality report for the memory store."""
        results = await self.analyze_all(limit=limit)

        if not results:
            return {"error": "No memories to analyze"}

        # Calculate distributions
        quality_bins = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        action_counts = {"keep": 0, "archive": 0, "prune": 0}
        filter_triggers = {}
        positive_signal_counts = {}

        for r in results:
            # Quality distribution
            if r.quality_score >= 0.8:
                quality_bins["excellent"] += 1
            elif r.quality_score >= 0.6:
                quality_bins["good"] += 1
            elif r.quality_score >= 0.4:
                quality_bins["fair"] += 1
            else:
                quality_bins["poor"] += 1

            # Action distribution
            action_counts[r.suggested_action] = action_counts.get(r.suggested_action, 0) + 1

            # Filter triggers
            for fr in r.filter_results:
                if fr.should_prune:
                    filter_triggers[fr.filter_name] = filter_triggers.get(fr.filter_name, 0) + 1

            # Positive signals
            for sig in r.positive_signals:
                positive_signal_counts[sig] = positive_signal_counts.get(sig, 0) + 1

        avg_quality = sum(r.quality_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)

        return {
            "total_memories": len(results),
            "average_quality_score": round(avg_quality, 3),
            "average_relevance_score": round(avg_relevance, 3),
            "quality_distribution": quality_bins,
            "suggested_actions": action_counts,
            "filter_triggers": filter_triggers,
            "positive_signals": positive_signal_counts,
            "prune_candidates": action_counts.get("prune", 0) + action_counts.get("archive", 0),
        }
