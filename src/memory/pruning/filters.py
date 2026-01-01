"""
Content filters for identifying low-quality memories.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class FilterResult:
    """Result of applying a filter to content."""

    should_prune: bool
    reason: str
    confidence: float  # 0.0 to 1.0
    filter_name: str
    metadata: dict[str, Any] | None = None


class ContentFilter(ABC):
    """Base class for content filters."""

    name: str = "base_filter"

    @abstractmethod
    def evaluate(self, content: str, metadata: dict[str, Any] | None = None) -> FilterResult:
        """Evaluate content and return filter result."""
        pass


class SpamFilter(ContentFilter):
    """Filter for spam and junk content."""

    name = "spam"

    SPAM_PATTERNS = [
        r"(?i)unsubscribe",
        r"(?i)click here to stop receiving",
        r"(?i)you are receiving this (email|message) because",
        r"(?i)this (email|message) was sent to",
        r"(?i)to unsubscribe from this list",
        r"(?i)manage your (email )?preferences",
        r"(?i)update your preferences",
        r"(?i)privacy policy",
        r"(?i)terms of service",
        r"(?i)view in browser",
        r"(?i)view this email in your browser",
        r"(?i)if you (can.?t see|cannot view) this",
        r"(?i)add us to your (safe sender|address book)",
        r"(?i)this is an automated (message|email|notification)",
        r"(?i)do not reply to this (message|email)",
        r"(?i)noreply@",
        r"(?i)no-reply@",
        r"(?i)mailer-daemon",
    ]

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._patterns = [re.compile(p) for p in self.SPAM_PATTERNS]

    def evaluate(self, content: str, metadata: dict[str, Any] | None = None) -> FilterResult:
        matches = sum(1 for p in self._patterns if p.search(content))
        score = min(matches / 5.0, 1.0)  # 5+ matches = 100% spam confidence

        return FilterResult(
            should_prune=score >= self.threshold,
            reason=f"Spam patterns detected: {matches}",
            confidence=score,
            filter_name=self.name,
            metadata={"pattern_matches": matches},
        )


class PromotionalFilter(ContentFilter):
    """Filter for promotional/marketing content."""

    name = "promotional"

    PROMO_PATTERNS = [
        r"(?i)limited time offer",
        r"(?i)act now",
        r"(?i)don.?t miss (out|this)",
        r"(?i)exclusive (deal|offer|discount)",
        r"(?i)save \d+%",
        r"(?i)\$\d+ off",
        r"(?i)free shipping",
        r"(?i)buy now",
        r"(?i)shop now",
        r"(?i)order now",
        r"(?i)sale ends",
        r"(?i)flash sale",
        r"(?i)black friday",
        r"(?i)cyber monday",
        r"(?i)holiday (sale|deal|special)",
        r"(?i)promo code",
        r"(?i)coupon code",
        r"(?i)discount code",
        r"(?i)use code",
        r"(?i)claim your",
        r"(?i)redeem your",
        r"(?i)last chance",
        r"(?i)hurry",
        r"(?i)while supplies last",
        r"(?i)lowest price",
        r"(?i)best price",
        r"(?i)price drop",
        r"(?i)new arrivals",
        r"(?i)new collection",
        r"(?i)trending now",
    ]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._patterns = [re.compile(p) for p in self.PROMO_PATTERNS]

    def evaluate(self, content: str, metadata: dict[str, Any] | None = None) -> FilterResult:
        matches = sum(1 for p in self._patterns if p.search(content))
        score = min(matches / 4.0, 1.0)  # 4+ matches = 100% promo confidence

        return FilterResult(
            should_prune=score >= self.threshold,
            reason=f"Promotional patterns detected: {matches}",
            confidence=score,
            filter_name=self.name,
            metadata={"pattern_matches": matches},
        )


class NewsletterFilter(ContentFilter):
    """Filter for generic newsletters with low personal relevance."""

    name = "newsletter"

    NEWSLETTER_PATTERNS = [
        r"(?i)newsletter",
        r"(?i)weekly digest",
        r"(?i)daily digest",
        r"(?i)monthly update",
        r"(?i)what.?s new",
        r"(?i)in this (issue|edition)",
        r"(?i)this week in",
        r"(?i)your weekly",
        r"(?i)your daily",
        r"(?i)top stories",
        r"(?i)trending stories",
        r"(?i)editor.?s pick",
        r"(?i)curated for you",
        r"(?i)recommended for you",
    ]

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._patterns = [re.compile(p) for p in self.NEWSLETTER_PATTERNS]

    def evaluate(self, content: str, metadata: dict[str, Any] | None = None) -> FilterResult:
        matches = sum(1 for p in self._patterns if p.search(content))
        score = min(matches / 3.0, 1.0)

        return FilterResult(
            should_prune=score >= self.threshold,
            reason=f"Newsletter patterns detected: {matches}",
            confidence=score,
            filter_name=self.name,
            metadata={"pattern_matches": matches},
        )


class NotificationFilter(ContentFilter):
    """Filter for automated notifications and alerts."""

    name = "notification"

    NOTIFICATION_PATTERNS = [
        r"(?i)your order (has been|was) (shipped|delivered|confirmed)",
        r"(?i)order confirmation",
        r"(?i)shipping confirmation",
        r"(?i)delivery notification",
        r"(?i)password reset",
        r"(?i)verify your (email|account)",
        r"(?i)confirm your (email|account)",
        r"(?i)your (password|account) (has been|was) (changed|updated)",
        r"(?i)security alert",
        r"(?i)login attempt",
        r"(?i)new sign.?in",
        r"(?i)payment (received|confirmed|processed)",
        r"(?i)invoice #",
        r"(?i)receipt for",
        r"(?i)subscription (renewed|cancelled|expiring)",
        r"(?i)your subscription",
        r"(?i)calendar (reminder|invitation|event)",
        r"(?i)reminder:",
        r"(?i)notification from",
        r"(?i)alert from",
        r"(?i)\[automated\]",
        r"(?i)\[auto-generated\]",
    ]

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._patterns = [re.compile(p) for p in self.NOTIFICATION_PATTERNS]

    def evaluate(self, content: str, metadata: dict[str, Any] | None = None) -> FilterResult:
        matches = sum(1 for p in self._patterns if p.search(content))
        score = min(matches / 3.0, 1.0)

        return FilterResult(
            should_prune=score >= self.threshold,
            reason=f"Notification patterns detected: {matches}",
            confidence=score,
            filter_name=self.name,
            metadata={"pattern_matches": matches},
        )


class SocialMediaFilter(ContentFilter):
    """Filter for social media notification emails."""

    name = "social_media"

    SOCIAL_PATTERNS = [
        r"(?i)someone (liked|commented|mentioned|tagged|followed|shared)",
        r"(?i)new (follower|like|comment|mention|connection)",
        r"(?i)you have \d+ new (notification|message|connection)",
        r"(?i)(facebook|twitter|linkedin|instagram|tiktok) notification",
        r"(?i)posted on your",
        r"(?i)replied to your",
        r"(?i)reacted to your",
        r"(?i)sent you a (message|friend request|connection request)",
        r"(?i)wants to connect",
        r"(?i)endorsed you",
        r"(?i)viewed your profile",
        r"(?i)people you may know",
        r"(?i)suggested (for you|connection|friend)",
        r"(?i)trending (on|in) your network",
    ]

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._patterns = [re.compile(p) for p in self.SOCIAL_PATTERNS]

    def evaluate(self, content: str, metadata: dict[str, Any] | None = None) -> FilterResult:
        matches = sum(1 for p in self._patterns if p.search(content))
        score = min(matches / 2.0, 1.0)

        return FilterResult(
            should_prune=score >= self.threshold,
            reason=f"Social media notification patterns: {matches}",
            confidence=score,
            filter_name=self.name,
            metadata={"pattern_matches": matches},
        )


class LowValueFilter(ContentFilter):
    """Filter for low-value, short, or empty content."""

    name = "low_value"

    def __init__(self, min_content_length: int = 50, min_word_count: int = 10):
        self.min_content_length = min_content_length
        self.min_word_count = min_word_count

    def evaluate(self, content: str, metadata: dict[str, Any] | None = None) -> FilterResult:
        # Remove common email headers
        body = re.sub(r"^(Subject|From|To|Date|Cc|Bcc):.*$", "", content, flags=re.MULTILINE)
        body = body.strip()

        content_length = len(body)
        word_count = len(body.split())

        if content_length < self.min_content_length or word_count < self.min_word_count:
            return FilterResult(
                should_prune=True,
                reason=f"Low content value: {word_count} words, {content_length} chars",
                confidence=0.8,
                filter_name=self.name,
                metadata={"word_count": word_count, "content_length": content_length},
            )

        return FilterResult(
            should_prune=False,
            reason="Sufficient content",
            confidence=0.0,
            filter_name=self.name,
            metadata={"word_count": word_count, "content_length": content_length},
        )
