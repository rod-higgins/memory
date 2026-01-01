"""
Content pruning and filtering for memory quality management.

Filters out:
- Promotional emails and newsletters
- Automated notifications
- Low-value transactional content
- Duplicate/redundant information

Prioritizes:
- User-authored content (sent emails, created documents)
- High-engagement content (starred, replied, important)
- Domain-relevant information
"""

from memory.pruning.analyzer import ContentAnalyzer
from memory.pruning.coordinator import PruningCoordinator
from memory.pruning.filters import ContentFilter, PromotionalFilter, SpamFilter

__all__ = [
    "ContentFilter",
    "SpamFilter",
    "PromotionalFilter",
    "ContentAnalyzer",
    "PruningCoordinator",
]
