"""
Export formats for memory injection into LLMs.

Converts memories into formats that can be injected into any LLM's
context window - Claude, GPT, Ollama, or local models.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from memory.schema.memory_entry import MemoryEntry


class ExportFormat(str, Enum):
    """Supported export formats."""

    XML = "xml"
    JSON = "json"
    MARKDOWN = "markdown"
    SYSTEM_PROMPT = "system_prompt"
    CLAUDE_FORMAT = "claude"  # Optimized for Claude
    COMPACT = "compact"  # Minimal token usage


@dataclass
class ExportConfig:
    """Configuration for memory export."""

    max_memories: int = 20
    max_tokens: int = 4000
    include_metadata: bool = True
    include_confidence: bool = True
    include_sources: bool = False
    group_by_type: bool = True
    sort_by: str = "relevance"  # relevance, recency, confidence


class BaseFormatter(ABC):
    """Base class for memory formatters."""

    @abstractmethod
    def format(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        """Format memories for export."""
        pass


class XMLFormatter(BaseFormatter):
    """Format memories as XML for LLM injection."""

    def format(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        config = config or ExportConfig()
        scores = scores or [1.0] * len(memories)

        lines = ['<user_memory_context>']

        if config.group_by_type:
            # Group by memory type
            by_type: dict[str, list[tuple[MemoryEntry, float]]] = {}
            for mem, score in zip(memories, scores):
                type_name = mem.memory_type.value
                if type_name not in by_type:
                    by_type[type_name] = []
                by_type[type_name].append((mem, score))

            for type_name, items in by_type.items():
                lines.append(f'  <{type_name}s>')
                for mem, score in items:
                    lines.append(self._format_memory(mem, score, config))
                lines.append(f'  </{type_name}s>')
        else:
            for mem, score in zip(memories, scores):
                lines.append(self._format_memory(mem, score, config))

        lines.append('</user_memory_context>')
        return '\n'.join(lines)

    def _format_memory(
        self,
        mem: MemoryEntry,
        score: float,
        config: ExportConfig,
    ) -> str:
        attrs = [f'type="{mem.memory_type.value}"']

        if config.include_confidence:
            attrs.append(f'confidence="{mem.confidence.overall:.2f}"')
            attrs.append(f'truth="{mem.truth_category.value}"')

        if config.include_metadata:
            if mem.domains:
                attrs.append(f'domains="{",".join(mem.domains[:3])}"')

        attr_str = " ".join(attrs)
        content = mem.summary or mem.content[:200]

        return f'    <memory {attr_str}>{content}</memory>'


class JSONFormatter(BaseFormatter):
    """Format memories as JSON."""

    def format(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        config = config or ExportConfig()
        scores = scores or [1.0] * len(memories)

        result = {
            "user_context": {
                "generated_at": datetime.now().isoformat(),
                "memory_count": len(memories),
                "memories": [],
            }
        }

        for mem, score in zip(memories, scores):
            entry: dict[str, Any] = {
                "content": mem.summary or mem.content[:300],
                "type": mem.memory_type.value,
            }

            if config.include_confidence:
                entry["confidence"] = round(mem.confidence.overall, 2)
                entry["truth_category"] = mem.truth_category.value

            if config.include_metadata:
                if mem.domains:
                    entry["domains"] = mem.domains[:3]
                if mem.tags:
                    entry["tags"] = mem.tags[:5]

            result["user_context"]["memories"].append(entry)

        return json.dumps(result, indent=2)


class MarkdownFormatter(BaseFormatter):
    """Format memories as Markdown."""

    def format(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        config = config or ExportConfig()
        scores = scores or [1.0] * len(memories)

        lines = ["# User Context", ""]

        if config.group_by_type:
            by_type: dict[str, list[tuple[MemoryEntry, float]]] = {}
            for mem, score in zip(memories, scores):
                type_name = mem.memory_type.value.title()
                if type_name not in by_type:
                    by_type[type_name] = []
                by_type[type_name].append((mem, score))

            for type_name, items in by_type.items():
                lines.append(f"## {type_name}s")
                lines.append("")
                for mem, score in items:
                    content = mem.summary or mem.content[:200]
                    confidence = f" ({mem.confidence.overall:.0%})" if config.include_confidence else ""
                    lines.append(f"- {content}{confidence}")
                lines.append("")
        else:
            for mem, score in zip(memories, scores):
                content = mem.summary or mem.content[:200]
                lines.append(f"- **{mem.memory_type.value}**: {content}")

        return '\n'.join(lines)


class SystemPromptFormatter(BaseFormatter):
    """Format memories as a system prompt section."""

    def format(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        config = config or ExportConfig()
        scores = scores or [1.0] * len(memories)

        lines = [
            "You have access to the following context about the user. Use this to personalize your responses:",
            ""
        ]

        # Group by type for clarity
        by_type: dict[str, list[MemoryEntry]] = {}
        for mem in memories:
            type_name = mem.memory_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(mem)

        # Preferences first
        if "preference" in by_type:
            lines.append("USER PREFERENCES:")
            for mem in by_type["preference"]:
                lines.append(f"  - {mem.summary or mem.content[:150]}")
            lines.append("")

        # Beliefs
        if "belief" in by_type:
            lines.append("USER BELIEFS:")
            for mem in by_type["belief"]:
                lines.append(f"  - {mem.summary or mem.content[:150]}")
            lines.append("")

        # Skills
        if "skill" in by_type:
            lines.append("USER EXPERTISE:")
            for mem in by_type["skill"]:
                lines.append(f"  - {mem.summary or mem.content[:150]}")
            lines.append("")

        # Facts
        if "fact" in by_type:
            lines.append("KNOWN FACTS:")
            for mem in by_type["fact"]:
                lines.append(f"  - {mem.summary or mem.content[:150]}")
            lines.append("")

        # Context
        if "context" in by_type:
            lines.append("RELEVANT CONTEXT:")
            for mem in by_type["context"][:5]:  # Limit context
                lines.append(f"  - {mem.summary or mem.content[:150]}")
            lines.append("")

        return '\n'.join(lines)


class ClaudeFormatter(BaseFormatter):
    """Format optimized for Claude models using XML tags."""

    def format(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        config = config or ExportConfig()
        scores = scores or [1.0] * len(memories)

        if not memories:
            return "<user_profile>\nNo relevant memories found for this query.\n</user_profile>"

        lines = [
            "<user_profile>",
            "The following represents known information about the user.",
            "Use this context to provide personalized, relevant responses.",
            ""
        ]

        # Group by memory type for better organization
        by_type: dict[str, list[MemoryEntry]] = {}
        for mem in memories:
            type_name = mem.memory_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(mem)

        # Preferences and beliefs are most important for personalization
        for type_name in ["preference", "belief", "skill"]:
            if type_name in by_type:
                tag_name = type_name + "s"
                lines.append(f"<user_{tag_name}>")
                for mem in by_type[type_name][:5]:
                    content = self._get_display_content(mem)
                    lines.append(f"  - {content}")
                lines.append(f"</user_{tag_name}>")
                lines.append("")

        # Facts and other content
        remaining = []
        for type_name, mems in by_type.items():
            if type_name not in ["preference", "belief", "skill"]:
                remaining.extend(mems)

        if remaining:
            lines.append("<relevant_context>")
            for mem in remaining[:10]:
                content = self._get_display_content(mem)
                domains = ", ".join(mem.domains[:2]) if mem.domains else ""
                domain_str = f" [{domains}]" if domains else ""
                lines.append(f"  - {content}{domain_str}")
            lines.append("</relevant_context>")

        lines.append("</user_profile>")

        return '\n'.join(lines)

    def _get_display_content(self, mem: MemoryEntry) -> str:
        """Get the best display content for a memory."""
        # Prefer summary
        if mem.summary and len(mem.summary) > 10:
            return mem.summary[:200]

        # For emails, try to extract subject line
        content = mem.content
        if content.startswith("Subject:"):
            lines = content.split("\n")
            subject = lines[0].replace("Subject:", "").strip()
            # Get first meaningful line of body
            for line in lines[4:]:
                line = line.strip()
                if line and len(line) > 20:
                    return f"{subject}: {line[:100]}"
            return subject[:150]

        # For other content, just truncate
        return content[:200].replace("\n", " ").strip()


class CompactFormatter(BaseFormatter):
    """Minimal token usage format."""

    def format(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        config = config or ExportConfig()

        # Super compact: just facts
        lines = ["[User context:]"]
        for mem in memories:
            # Very short format: type:content
            type_abbrev = mem.memory_type.value[0].upper()  # F, B, P, S, E, C
            content = (mem.summary or mem.content)[:80]
            lines.append(f"{type_abbrev}: {content}")

        return '\n'.join(lines)


class MemoryExporter:
    """
    Main exporter class for converting memories to LLM-ready formats.

    Usage:
        exporter = MemoryExporter()
        xml_output = exporter.export(memories, format=ExportFormat.XML)
        claude_output = exporter.export(memories, format=ExportFormat.CLAUDE_FORMAT)
    """

    def __init__(self):
        self._formatters: dict[ExportFormat, BaseFormatter] = {
            ExportFormat.XML: XMLFormatter(),
            ExportFormat.JSON: JSONFormatter(),
            ExportFormat.MARKDOWN: MarkdownFormatter(),
            ExportFormat.SYSTEM_PROMPT: SystemPromptFormatter(),
            ExportFormat.CLAUDE_FORMAT: ClaudeFormatter(),
            ExportFormat.COMPACT: CompactFormatter(),
        }

    def export(
        self,
        memories: list[MemoryEntry],
        format: ExportFormat = ExportFormat.CLAUDE_FORMAT,
        scores: list[float] | None = None,
        config: ExportConfig | None = None,
    ) -> str:
        """
        Export memories to the specified format.

        Args:
            memories: List of memories to export
            format: Target format
            scores: Optional relevance scores
            config: Export configuration

        Returns:
            Formatted string ready for LLM injection
        """
        formatter = self._formatters.get(format)
        if not formatter:
            raise ValueError(f"Unknown format: {format}")

        return formatter.format(memories, scores, config)

    def export_for_claude(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
    ) -> str:
        """Convenience method for Claude format."""
        return self.export(memories, ExportFormat.CLAUDE_FORMAT, scores)

    def export_for_system_prompt(
        self,
        memories: list[MemoryEntry],
        scores: list[float] | None = None,
    ) -> str:
        """Convenience method for generic system prompt format."""
        return self.export(memories, ExportFormat.SYSTEM_PROMPT, scores)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4


def export_memories(
    memories: list[MemoryEntry],
    format: str = "claude",
    scores: list[float] | None = None,
) -> str:
    """
    Convenience function to export memories.

    Args:
        memories: Memories to export
        format: Format name (xml, json, markdown, system_prompt, claude, compact)
        scores: Optional relevance scores

    Returns:
        Formatted string
    """
    exporter = MemoryExporter()
    fmt = ExportFormat(format)
    return exporter.export(memories, fmt, scores)
