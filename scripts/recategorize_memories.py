#!/usr/bin/env python3
"""
Recategorize memories into appropriate tiers and memory types.

Tiers:
- SHORT_TERM: Recent activity, temporary notes (last 24 hours)
- LONG_TERM: Important but not core (emails, browsing, events)
- PERSISTENT: Core identity, skills, preferences, relationships

Memory Types:
- FACT: Verifiable information
- BELIEF: Held convictions
- PREFERENCE: Likes/dislikes
- SKILL: Known capabilities
- RELATIONSHIP: Connections between entities
- EVENT: Timestamped occurrences
- CONTEXT: Domain/situational information
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path


def categorize_memory(content: str, domains: list, tags: list, created_at: datetime) -> tuple[str, str]:
    """
    Analyze memory content and return (tier, memory_type).
    """
    content_lower = content.lower()

    # Determine memory type based on content patterns
    memory_type = "fact"  # default

    # SKILL patterns
    skill_patterns = [
        r"expertise in", r"proficient in", r"skilled at", r"experience with",
        r"years of experience", r"specialist in", r"certified", r"developer",
        r"engineer", r"architect", r"programming", r"coding", r"python",
        r"javascript", r"typescript", r"react", r"node", r"aws", r"azure",
        r"docker", r"kubernetes", r"devops", r"full.?stack", r"backend",
        r"frontend", r"database", r"sql", r"api", r"microservices"
    ]
    if any(re.search(p, content_lower) for p in skill_patterns):
        if "resume" in content_lower or "cv" in content_lower or "experience" in content_lower:
            memory_type = "skill"

    # PREFERENCE patterns
    preference_patterns = [
        r"i (prefer|like|love|enjoy|hate|dislike)", r"my favorite",
        r"i always", r"i never", r"i usually", r"settings", r"configuration"
    ]
    if any(re.search(p, content_lower) for p in preference_patterns):
        memory_type = "preference"

    # RELATIONSHIP patterns (people, organizations)
    relationship_patterns = [
        r"contact", r"colleague", r"friend", r"manager", r"team",
        r"client", r"partner", r"@.*\.com", r"linkedin\.com/in/",
        r"worked with", r"reports to", r"manages"
    ]
    if any(re.search(p, content_lower) for p in relationship_patterns):
        if "personal" in domains or "contact" in str(tags).lower():
            memory_type = "relationship"

    # EVENT patterns
    event_patterns = [
        r"meeting", r"appointment", r"schedule", r"calendar",
        r"on \d{1,2}/\d{1,2}", r"at \d{1,2}:\d{2}", r"tomorrow",
        r"next week", r"deadline", r"due date", r"event",
        r"conference", r"webinar", r"interview"
    ]
    if any(re.search(p, content_lower) for p in event_patterns):
        memory_type = "event"

    # CONTEXT patterns
    context_patterns = [
        r"project:", r"context:", r"background:", r"overview:",
        r"summary:", r"status:", r"update:", r"note:"
    ]
    if any(re.search(p, content_lower) for p in context_patterns):
        memory_type = "context"

    # BELIEF patterns
    belief_patterns = [
        r"i (believe|think|feel that)", r"in my opinion",
        r"i'm convinced", r"i strongly", r"my view is"
    ]
    if any(re.search(p, content_lower) for p in belief_patterns):
        memory_type = "belief"

    # Determine tier based on content and recency
    now = datetime.now()
    age_hours = (now - created_at).total_seconds() / 3600 if created_at else 1000

    tier = "persistent"  # default

    # SHORT_TERM: Very recent or temporary
    if age_hours < 24:
        tier = "short_term"
    elif age_hours < 72 and "notification" in content_lower:
        tier = "short_term"

    # LONG_TERM: Emails, browsing history, events
    elif "email" in domains or "communication" in domains:
        # Most emails go to long-term, but important ones to persistent
        if "starred" in str(tags).lower() or "important" in str(tags).lower():
            tier = "persistent"
        else:
            tier = "long_term"
    elif "browsing" in domains:
        tier = "long_term"
    elif memory_type == "event":
        tier = "long_term"

    # PERSISTENT: Core identity, skills, preferences, relationships
    elif memory_type in ("skill", "preference", "relationship", "belief"):
        tier = "persistent"
    elif "personal" in domains or "documents" in domains:
        tier = "persistent"

    return tier, memory_type


def recategorize_all():
    """Read all memories and recategorize them."""
    db_path = Path("~/memory/data/persistent/core.sqlite").expanduser()

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all memories
    cursor = conn.execute("""
        SELECT id, content, domains_json, tags_json, created_at, memory_type, tier
        FROM memories
        WHERE is_active = 1
    """)

    rows = cursor.fetchall()
    print(f"Found {len(rows)} memories to recategorize")

    # Count changes
    tier_changes = {"short_term": 0, "long_term": 0, "persistent": 0}
    type_changes = {}

    updates = []

    for row in rows:
        import json

        content = row["content"] or ""
        domains = json.loads(row["domains_json"]) if row["domains_json"] else []
        tags = json.loads(row["tags_json"]) if row["tags_json"] else []

        try:
            created_at = datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
        except (ValueError, TypeError):
            created_at = datetime.now()

        new_tier, new_type = categorize_memory(content, domains, tags, created_at)

        # Track changes
        tier_changes[new_tier] = tier_changes.get(new_tier, 0) + 1
        type_changes[new_type] = type_changes.get(new_type, 0) + 1

        # Only update if changed
        if new_tier != row["tier"] or new_type != row["memory_type"]:
            updates.append((new_tier, new_type, row["id"]))

    print("\nNew tier distribution:")
    for tier, count in sorted(tier_changes.items()):
        print(f"  {tier}: {count}")

    print("\nNew type distribution:")
    for mtype, count in sorted(type_changes.items(), key=lambda x: -x[1]):
        print(f"  {mtype}: {count}")

    print(f"\n{len(updates)} memories need updating...")

    # Apply updates
    if updates:
        cursor = conn.cursor()
        cursor.executemany("""
            UPDATE memories
            SET tier = ?, memory_type = ?, updated_at = ?
            WHERE id = ?
        """, [(t, mt, datetime.now().isoformat(), id) for t, mt, id in updates])
        conn.commit()
        print(f"Updated {len(updates)} memories")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    recategorize_all()
