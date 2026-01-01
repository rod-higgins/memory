"""
Knowledge Management for PLM.

Stores distilled knowledge, not raw content.
"""

from memory.knowledge.extractor import KnowledgeExtractor
from memory.knowledge.store import KnowledgeEntry, KnowledgeStore, KnowledgeType

__all__ = [
    "KnowledgeEntry",
    "KnowledgeStore",
    "KnowledgeType",
    "KnowledgeExtractor",
]
