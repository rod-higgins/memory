"""
Parser for Claude Code interaction history.

Extracts meaningful memories from ~/.claude/history.jsonl
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from memory.schema.memory_entry import (
    MemoryEntry,
    MemorySource,
    MemoryType,
    SourceType,
    TruthCategory,
)


class ClaudeHistoryParser:
    """
    Parser for Claude Code history.jsonl files.

    Extracts user inputs and meaningful interactions.
    """

    def __init__(self, history_path: str | Path = "~/.claude/history.jsonl"):
        self._history_path = Path(history_path).expanduser()

    async def parse(
        self,
        min_content_length: int = 20,
        max_entries: int | None = None,
    ) -> AsyncIterator[MemoryEntry]:
        """
        Parse Claude history and yield memory entries.

        Args:
            min_content_length: Minimum content length to include
            max_entries: Maximum number of entries to parse (None = all)

        Yields:
            MemoryEntry objects for each meaningful interaction
        """
        if not self._history_path.exists():
            return

        count = 0
        with open(self._history_path, encoding="utf-8") as f:
            for line in f:
                if max_entries and count >= max_entries:
                    break

                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Extract content from display field
                content = entry.get("display", "")
                if len(content) < min_content_length:
                    continue

                # Skip system/tool messages
                if self._is_system_message(content):
                    continue

                # Extract metadata
                project = entry.get("project", "")
                timestamp = entry.get("timestamp", 0)
                session_id = entry.get("sessionId", "")

                # Create memory entry
                memory = MemoryEntry(
                    content=content,
                    memory_type=self._classify_content(content),
                    truth_category=TruthCategory.INFERRED,
                    sources=[
                        MemorySource(
                            source_type=SourceType.CLAUDE_HISTORY,
                            source_path=str(self._history_path),
                            source_id=session_id,
                            timestamp=datetime.fromtimestamp(timestamp / 1000)
                            if timestamp
                            else datetime.now(),
                            raw_content=content,
                            extraction_method="claude_history_parser",
                        )
                    ],
                    domains=self._extract_domains(project, content),
                    tags=self._extract_tags(content),
                    metadata={
                        "project": project,
                        "session_id": session_id,
                        "original_timestamp": timestamp,
                    },
                )

                count += 1
                yield memory

    def _is_system_message(self, content: str) -> bool:
        """Check if content is a system/tool message to skip."""
        skip_patterns = [
            "Tool result:",
            "Function call:",
            "```tool",
            "<tool_",
            "Successfully",
            "Error:",
            "Warning:",
        ]
        content_start = content[:100].lower()
        return any(p.lower() in content_start for p in skip_patterns)

    def _classify_content(self, content: str) -> MemoryType:
        """Classify content type based on heuristics."""
        content_lower = content.lower()

        # Preference indicators
        if any(
            phrase in content_lower
            for phrase in [
                "i prefer",
                "i like",
                "i want",
                "i'd rather",
                "i always use",
                "my favorite",
            ]
        ):
            return MemoryType.PREFERENCE

        # Belief indicators
        if any(
            phrase in content_lower
            for phrase in [
                "i believe",
                "i think",
                "in my opinion",
                "i feel that",
                "my view is",
            ]
        ):
            return MemoryType.BELIEF

        # Skill indicators
        if any(
            phrase in content_lower
            for phrase in [
                "i know how",
                "i can",
                "i'm experienced",
                "i've worked with",
                "my expertise",
            ]
        ):
            return MemoryType.SKILL

        # Event indicators (past tense actions)
        if any(
            phrase in content_lower
            for phrase in [
                "i did",
                "i created",
                "i built",
                "i fixed",
                "yesterday",
                "last week",
            ]
        ):
            return MemoryType.EVENT

        # Default to context
        return MemoryType.CONTEXT

    def _extract_domains(self, project: str, content: str) -> list[str]:
        """Extract knowledge domains from project path and content."""
        domains: set[str] = set()
        combined = f"{project} {content}".lower()

        # Technology domains
        domain_keywords = {
            "drupal": ["drupal", "twig", "hook_", "module", "theme"],
            "python": ["python", "pip", "pytest", "django", "flask"],
            "typescript": ["typescript", ".ts", "tsx", "interface"],
            "javascript": ["javascript", ".js", "node", "npm", "react", "vue"],
            "php": ["php", "composer", "laravel", "symfony"],
            "database": ["sql", "postgres", "mysql", "sqlite", "mongodb"],
            "devops": ["docker", "kubernetes", "terraform", "ci/cd", "deploy"],
            "ai": ["llm", "gpt", "claude", "embedding", "model", "ai"],
            "git": ["git", "commit", "branch", "merge", "pull request"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in combined for kw in keywords):
                domains.add(domain)

        # Project-specific domains from path
        if "document" in project.lower():
            domains.add("document_management")
        if "fwc" in project.lower():
            domains.add("government")
        if "simplex" in project.lower():
            domains.add("programming_language")
        if "aether" in project.lower():
            domains.add("cms")

        return list(domains)[:5]  # Limit to 5 domains

    def _extract_tags(self, content: str) -> list[str]:
        """Extract relevant tags from content."""
        tags: set[str] = set()
        content_lower = content.lower()

        # Common action tags
        action_tags = {
            "debug": ["debug", "error", "fix", "bug", "issue"],
            "create": ["create", "new", "add", "implement"],
            "update": ["update", "modify", "change", "refactor"],
            "delete": ["delete", "remove", "clean"],
            "question": ["how", "what", "why", "can you", "help me"],
            "review": ["review", "check", "look at", "examine"],
        }

        for tag, keywords in action_tags.items():
            if any(kw in content_lower for kw in keywords):
                tags.add(tag)

        # Add source tag
        tags.add("claude_interaction")

        return list(tags)[:10]  # Limit to 10 tags

    async def count_entries(self) -> int:
        """Count total entries in the history file."""
        if not self._history_path.exists():
            return 0

        count = 0
        with open(self._history_path, encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count

    async def get_date_range(self) -> tuple[datetime | None, datetime | None]:
        """Get the date range of entries in the history."""
        if not self._history_path.exists():
            return None, None

        first_ts = None
        last_ts = None

        with open(self._history_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    ts = entry.get("timestamp", 0)
                    if ts:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                except json.JSONDecodeError:
                    continue

        first_dt = datetime.fromtimestamp(first_ts / 1000) if first_ts else None
        last_dt = datetime.fromtimestamp(last_ts / 1000) if last_ts else None

        return first_dt, last_dt
