"""
LLM Interaction Tracker.

Captures and stores interactions with various LLMs:
- Claude (via API and Claude Code)
- ChatGPT (via export)
- Local LLMs (Ollama)

This creates memories from your AI conversations.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.schema import MemoryEntry, MemoryType


@dataclass
class LLMInteraction:
    """A single LLM interaction."""

    timestamp: datetime
    provider: str  # claude, chatgpt, ollama, etc.
    model: str
    user_message: str
    assistant_response: str
    tokens_used: int = 0
    metadata: dict[str, Any] | None = None


class ClaudeCodeTracker:
    """Track Claude Code sessions and extract insights."""

    def __init__(self, projects_path: str = "~/.claude/projects"):
        self.projects_path = Path(projects_path).expanduser()

    async def get_recent_sessions(self, limit: int = 50) -> list[LLMInteraction]:
        """Get recent Claude Code sessions."""
        interactions = []

        if not self.projects_path.exists():
            return []

        # Find all project directories
        for project_dir in self.projects_path.iterdir():
            if not project_dir.is_dir():
                continue

            # Look for conversation files
            for conv_file in project_dir.glob("*.jsonl"):
                try:
                    interactions.extend(
                        self._parse_conversation_file(conv_file, limit=limit // 5)
                    )
                except Exception:
                    continue

            # Also check for session database
            db_path = project_dir / "sessions.db"
            if db_path.exists():
                try:
                    interactions.extend(self._parse_sessions_db(db_path, limit=limit // 5))
                except Exception:
                    continue

        # Sort by timestamp and limit
        interactions.sort(key=lambda x: x.timestamp, reverse=True)
        return interactions[:limit]

    def _parse_conversation_file(
        self, file_path: Path, limit: int = 10
    ) -> list[LLMInteraction]:
        """Parse a JSONL conversation file."""
        interactions = []

        with open(file_path) as f:
            lines = f.readlines()[-limit * 2 :]  # Get last N*2 lines (user + assistant pairs)

        user_msg = None
        for line in lines:
            try:
                data = json.loads(line.strip())
                role = data.get("role", "")
                content = data.get("content", "")

                if isinstance(content, list):
                    # Handle structured content
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )

                if role == "user":
                    user_msg = content
                elif role == "assistant" and user_msg:
                    interactions.append(
                        LLMInteraction(
                            timestamp=datetime.fromisoformat(
                                data.get("timestamp", datetime.now().isoformat())
                            )
                            if "timestamp" in data
                            else datetime.now(),
                            provider="claude_code",
                            model=data.get("model", "claude"),
                            user_message=user_msg[:500],
                            assistant_response=content[:1000],
                            metadata={"file": str(file_path)},
                        )
                    )
                    user_msg = None
            except (json.JSONDecodeError, KeyError):
                continue

        return interactions

    def _parse_sessions_db(self, db_path: Path, limit: int = 10) -> list[LLMInteraction]:
        """Parse sessions database."""
        interactions = []

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Try to find conversation data
            cursor = conn.execute(
                """
                SELECT * FROM messages
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit * 2,),
            )

            rows = cursor.fetchall()
            conn.close()

            user_msg = None
            for row in rows:
                role = row.get("role", "")
                content = row.get("content", "")

                if role == "user":
                    user_msg = content
                elif role == "assistant" and user_msg:
                    interactions.append(
                        LLMInteraction(
                            timestamp=datetime.fromisoformat(row.get("created_at", "")),
                            provider="claude_code",
                            model="claude",
                            user_message=user_msg[:500],
                            assistant_response=content[:1000],
                        )
                    )
                    user_msg = None
        except Exception:
            pass

        return interactions


class LLMInteractionStore:
    """Store and retrieve LLM interactions as memories."""

    def __init__(self, db_path: str = "~/memory/data/llm_interactions.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT,
                user_message TEXT,
                assistant_response TEXT,
                tokens_used INTEGER DEFAULT 0,
                metadata_json TEXT,
                processed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_processed ON interactions(processed)"
        )
        conn.commit()
        conn.close()

    def store_interaction(self, interaction: LLMInteraction) -> int:
        """Store an interaction."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            """
            INSERT INTO interactions
            (timestamp, provider, model, user_message, assistant_response, tokens_used, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                interaction.timestamp.isoformat(),
                interaction.provider,
                interaction.model,
                interaction.user_message,
                interaction.assistant_response,
                interaction.tokens_used,
                json.dumps(interaction.metadata) if interaction.metadata else None,
            ),
        )
        conn.commit()
        interaction_id = cursor.lastrowid
        conn.close()
        return interaction_id

    def get_unprocessed(self, limit: int = 100) -> list[tuple[int, LLMInteraction]]:
        """Get unprocessed interactions."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT * FROM interactions
            WHERE processed = 0
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        results = []
        for row in cursor.fetchall():
            interaction = LLMInteraction(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                provider=row["provider"],
                model=row["model"] or "",
                user_message=row["user_message"] or "",
                assistant_response=row["assistant_response"] or "",
                tokens_used=row["tokens_used"] or 0,
                metadata=json.loads(row["metadata_json"])
                if row["metadata_json"]
                else None,
            )
            results.append((row["id"], interaction))

        conn.close()
        return results

    def mark_processed(self, interaction_ids: list[int]) -> None:
        """Mark interactions as processed."""
        if not interaction_ids:
            return

        conn = sqlite3.connect(str(self.db_path))
        placeholders = ",".join("?" * len(interaction_ids))
        conn.execute(
            f"UPDATE interactions SET processed = 1 WHERE id IN ({placeholders})",
            interaction_ids,
        )
        conn.commit()
        conn.close()

    def to_memory_entry(self, interaction: LLMInteraction) -> MemoryEntry:
        """Convert an interaction to a memory entry."""
        # Create a summary of the interaction
        summary = f"Asked {interaction.provider}: {interaction.user_message[:200]}"
        if len(interaction.user_message) > 200:
            summary += "..."

        return MemoryEntry(
            content=f"Q: {interaction.user_message}\n\nA: {interaction.assistant_response}",
            summary=summary,
            memory_type=MemoryType.INTERACTION,
            domains=["ai", interaction.provider],
            tags=["llm_interaction", interaction.provider, interaction.model],
            source_id=f"{interaction.provider}:{interaction.timestamp.isoformat()}",
        )


async def sync_claude_code_interactions(limit: int = 50) -> list[MemoryEntry]:
    """Sync Claude Code interactions to memory."""
    tracker = ClaudeCodeTracker()
    store = LLMInteractionStore()

    # Get recent sessions
    interactions = await tracker.get_recent_sessions(limit=limit)

    memories = []
    for interaction in interactions:
        # Store in interaction database
        store.store_interaction(interaction)

        # Convert to memory
        memory = store.to_memory_entry(interaction)
        memories.append(memory)

    return memories
