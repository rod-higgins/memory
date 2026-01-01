"""
Content analyzer for memory quality assessment.

Analyzes memories to determine:
- Quality score (0-1)
- Relevance to user's domains
- Whether content should be pruned
- Suggestions for improvement
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory.pruning.filters import (
    ContentFilter,
    FilterResult,
    LowValueFilter,
    NewsletterFilter,
    NotificationFilter,
    PromotionalFilter,
    SocialMediaFilter,
    SpamFilter,
)


@dataclass
class AnalysisResult:
    """Complete analysis of a memory's quality and relevance."""

    memory_id: str
    quality_score: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    should_prune: bool
    prune_reasons: list[str] = field(default_factory=list)
    filter_results: list[FilterResult] = field(default_factory=list)
    positive_signals: list[str] = field(default_factory=list)
    suggested_action: str = "keep"  # keep, prune, archive, merge
    confidence: float = 0.0


class ContentAnalyzer:
    """
    Analyzes memory content for quality and relevance.

    Combines multiple filters and signals to produce a comprehensive
    quality assessment for each memory.
    """

    # User's known domains of interest (could be loaded from identity)
    USER_DOMAINS = [
        "software", "programming", "development", "code",
        "drupal", "php", "python", "javascript", "typescript",
        "aws", "cloud", "infrastructure", "devops",
        "ai", "machine learning", "llm", "artificial intelligence",
        "business", "consulting", "project management",
        "senua", "senuamedia", "higgins",
    ]

    def __init__(
        self,
        filters: list[ContentFilter] | None = None,
        prune_threshold: float = 0.6,
    ):
        self.prune_threshold = prune_threshold
        self.filters = filters or [
            SpamFilter(),
            PromotionalFilter(),
            NewsletterFilter(),
            NotificationFilter(),
            SocialMediaFilter(),
            LowValueFilter(),
        ]

    def analyze(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        memory_id: str = "",
        tags: list[str] | None = None,
        domains: list[str] | None = None,
    ) -> AnalysisResult:
        """
        Analyze content and return comprehensive quality assessment.
        """
        metadata = metadata or {}
        tags = tags or []
        domains = domains or []

        filter_results = []
        prune_reasons = []
        positive_signals = []

        # Run all filters
        for f in self.filters:
            result = f.evaluate(content, metadata)
            filter_results.append(result)
            if result.should_prune:
                prune_reasons.append(result.reason)

        # Check for positive signals
        positive_signals.extend(self._detect_positive_signals(content, tags, metadata))

        # Calculate quality score
        quality_score = self._calculate_quality_score(filter_results, positive_signals)

        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(content, domains, tags)

        # Determine if should prune
        prune_score = sum(
            r.confidence for r in filter_results if r.should_prune
        ) / max(len(self.filters), 1)

        # Positive signals reduce prune likelihood
        positive_boost = len(positive_signals) * 0.15
        adjusted_prune_score = max(0, prune_score - positive_boost)

        should_prune = adjusted_prune_score >= self.prune_threshold

        # Determine suggested action
        if should_prune and adjusted_prune_score >= 0.8:
            suggested_action = "prune"
        elif should_prune:
            suggested_action = "archive"
        elif relevance_score < 0.3:
            suggested_action = "archive"
        else:
            suggested_action = "keep"

        return AnalysisResult(
            memory_id=memory_id,
            quality_score=quality_score,
            relevance_score=relevance_score,
            should_prune=should_prune,
            prune_reasons=prune_reasons,
            filter_results=filter_results,
            positive_signals=positive_signals,
            suggested_action=suggested_action,
            confidence=adjusted_prune_score if should_prune else (1 - adjusted_prune_score),
        )

    def _detect_positive_signals(
        self,
        content: str,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> list[str]:
        """Detect positive quality signals."""
        signals = []

        # User-authored content
        if "sent" in tags or "sent_email" in tags:
            signals.append("user_authored")

        # Starred/important
        if "starred" in tags or "important" in tags:
            signals.append("marked_important")

        # Has replies/engagement
        if "replied" in tags:
            signals.append("has_engagement")

        # Business email
        if any(d in ["business", "work", "senuamedia"] for d in (metadata.get("domains") or [])):
            signals.append("business_context")

        # Contains code
        if re.search(r'```|def |function |class |import |from |const |let |var ', content):
            signals.append("contains_code")

        # Contains project references
        if re.search(r'(?i)(project|ticket|jira|github|pull request|pr|issue)', content):
            signals.append("project_reference")

        # Technical discussion
        if re.search(r'(?i)(api|database|server|deploy|bug|feature|implement)', content):
            signals.append("technical_content")

        return signals

    def _calculate_quality_score(
        self,
        filter_results: list[FilterResult],
        positive_signals: list[str],
    ) -> float:
        """Calculate overall quality score (0-1)."""
        # Start with base score
        score = 0.5

        # Negative adjustments from filters
        for result in filter_results:
            if result.should_prune:
                score -= result.confidence * 0.15

        # Positive adjustments from signals
        signal_weights = {
            "user_authored": 0.25,
            "marked_important": 0.20,
            "has_engagement": 0.15,
            "business_context": 0.15,
            "contains_code": 0.10,
            "project_reference": 0.10,
            "technical_content": 0.10,
        }

        for signal in positive_signals:
            score += signal_weights.get(signal, 0.05)

        return max(0.0, min(1.0, score))

    def _calculate_relevance_score(
        self,
        content: str,
        domains: list[str],
        tags: list[str],
    ) -> float:
        """Calculate relevance to user's interests (0-1)."""
        score = 0.0
        content_lower = content.lower()

        # Check domain keywords
        matches = sum(1 for d in self.USER_DOMAINS if d in content_lower)
        score += min(matches * 0.1, 0.5)

        # Check if memory domains match user domains
        matching_domains = set(d.lower() for d in domains) & set(self.USER_DOMAINS)
        score += len(matching_domains) * 0.1

        # Boost for business/work content
        if any(t in ["sent", "business", "senuamedia", "work"] for t in tags):
            score += 0.2

        return max(0.0, min(1.0, score))

    def batch_analyze(
        self,
        memories: list[dict[str, Any]],
    ) -> list[AnalysisResult]:
        """Analyze multiple memories efficiently."""
        return [
            self.analyze(
                content=m.get("content", ""),
                metadata=m.get("metadata"),
                memory_id=str(m.get("id", "")),
                tags=m.get("tags", []),
                domains=m.get("domains", []),
            )
            for m in memories
        ]
