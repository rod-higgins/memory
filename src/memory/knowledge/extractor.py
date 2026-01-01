"""
Knowledge Extraction Pipeline.

Extracts distilled knowledge from raw content sources:
- Emails → Skills, experiences, contacts, projects
- Documents → Facts, insights
- Browser history → Interests, research topics

Stores ONLY embeddings and summaries, NOT raw content.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from memory.knowledge.store import KnowledgeEntry, KnowledgeStore, KnowledgeType


@dataclass
class ExtractionResult:
    """Result of knowledge extraction."""

    entries: list[KnowledgeEntry]
    source_count: int
    categories_found: list[str]
    processing_time_ms: float


class OllamaExtractor:
    """Uses Ollama to extract knowledge from content."""

    def __init__(self, model: str = "tinyllama"):
        self.model = model
        self.base_url = "http://localhost:11434"

    async def complete(self, prompt: str, max_tokens: int = 300) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": 0.2},
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()["response"].strip()
            except Exception as e:
                print(f"Ollama error: {e}")
                return ""

    async def extract_from_email(self, subject: str, body: str) -> list[dict]:
        """Extract knowledge from an email."""
        prompt = f"""Extract key knowledge from this email. Return JSON array of insights:

Subject: {subject}
Body (excerpt): {body[:1000]}

Extract:
- Skills mentioned (technologies, tools)
- Projects or work discussed
- Organizations or companies
- Key decisions or actions

Return as JSON array:
[{{"type": "skill|project|experience|contact", "summary": "brief description", "category": "technology|business|personal", "entities": ["named entities"]}}]

JSON:"""
        result = await self.complete(prompt, max_tokens=400)
        return self._parse_json_array(result)

    async def extract_from_document(self, title: str, content: str) -> list[dict]:
        """Extract knowledge from a document."""
        prompt = f"""Extract key knowledge from this document:

Title: {title}
Content (excerpt): {content[:1500]}

Extract the most important facts, insights, and actionable information.

Return as JSON array:
[{{"type": "fact|insight|skill", "summary": "brief description (max 200 chars)", "category": "category", "entities": ["key terms"]}}]

JSON:"""
        result = await self.complete(prompt, max_tokens=400)
        return self._parse_json_array(result)

    async def synthesize_profile(self, knowledge_items: list[str]) -> dict:
        """Synthesize a profile from extracted knowledge."""
        combined = "\n".join(knowledge_items[:20])

        prompt = f"""Based on these knowledge items, create a professional profile:

{combined}

Return as JSON:
{{
  "expertise": ["top skills"],
  "roles": ["job roles held"],
  "technologies": ["technologies used"],
  "industries": ["industry experience"],
  "summary": "2-sentence professional summary"
}}

JSON:"""
        result = await self.complete(prompt, max_tokens=400)

        try:
            if "{" in result:
                start = result.find("{")
                end = result.rfind("}") + 1
                return json.loads(result[start:end])
        except json.JSONDecodeError:
            pass

        return {}

    def _parse_json_array(self, text: str) -> list[dict]:
        """Parse JSON array from LLM output."""
        try:
            if "[" in text:
                start = text.find("[")
                end = text.rfind("]") + 1
                if end > start:
                    return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return []


class KnowledgeExtractor:
    """
    Extracts and distills knowledge from raw content sources.

    This is the main pipeline for converting raw data into
    searchable, embeddings-based knowledge entries.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore | None = None,
        source_db_path: str = "~/memory/data/persistent/core.sqlite",
    ):
        self.store = knowledge_store
        self.source_db = Path(source_db_path).expanduser()
        self.llm = OllamaExtractor()

    async def initialize(self) -> None:
        """Initialize the extractor."""
        if not self.store:
            self.store = KnowledgeStore()
        await self.store.initialize()

    async def close(self) -> None:
        """Close connections."""
        if self.store:
            await self.store.close()

    async def extract_all(self, limit: int = 500) -> ExtractionResult:
        """
        Extract knowledge from all sources.
        """
        start = datetime.now()
        all_entries: list[KnowledgeEntry] = []
        categories = set()

        # Extract from emails
        email_entries = await self.extract_from_emails(limit=limit)
        all_entries.extend(email_entries)

        # Build and store profile
        profile_entry = await self.build_profile()
        if profile_entry:
            all_entries.append(profile_entry)

        for entry in all_entries:
            categories.add(entry.category)

        elapsed = (datetime.now() - start).total_seconds() * 1000

        return ExtractionResult(
            entries=all_entries,
            source_count=len(all_entries),
            categories_found=list(categories),
            processing_time_ms=elapsed,
        )

    async def extract_from_emails(self, limit: int = 200) -> list[KnowledgeEntry]:
        """Extract knowledge from email content."""
        print("Extracting knowledge from emails...")

        if not self.source_db.exists():
            print(f"  Source database not found: {self.source_db}")
            return []

        conn = sqlite3.connect(str(self.source_db))
        conn.row_factory = sqlite3.Row

        # Get sent emails (user-authored - most valuable)
        cursor = conn.execute(
            """
            SELECT id, content, summary, tags_json
            FROM memories
            WHERE tags_json LIKE '%sent%'
            AND domains_json LIKE '%email%'
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        print(f"  Processing {len(rows)} sent emails...")

        entries: list[KnowledgeEntry] = []
        processed = 0

        for row in rows:
            content = row["content"]

            # Extract subject and body
            subject = ""
            body = content
            if content.startswith("Subject:"):
                lines = content.split("\n")
                subject = lines[0].replace("Subject:", "").strip()
                body = "\n".join(lines[4:])  # Skip headers

            # Skip very short emails
            if len(body) < 50:
                continue

            # Extract knowledge using LLM
            try:
                knowledge_items = await self.llm.extract_from_email(subject, body)

                for item in knowledge_items[:3]:  # Max 3 per email
                    if not item.get("summary"):
                        continue

                    entry = KnowledgeEntry(
                        knowledge_type=self._map_type(item.get("type", "fact")),
                        summary=item["summary"][:500],
                        category=item.get("category", "general"),
                        entities=item.get("entities", [])[:5],
                        tags=["email_derived"],
                        confidence=0.75,
                    )

                    # Store in knowledge base
                    await self.store.store(entry)
                    entries.append(entry)

                processed += 1
                if processed % 20 == 0:
                    print(f"    Processed {processed} emails, extracted {len(entries)} knowledge items")

            except Exception as e:
                print(f"    Error processing email: {e}")

            # Rate limiting
            await asyncio.sleep(0.2)

        conn.close()
        print(f"  Extracted {len(entries)} knowledge entries from emails")
        return entries

    async def build_profile(self) -> KnowledgeEntry | None:
        """Build a synthesized profile from all knowledge."""
        print("Building synthesized profile...")

        if not self.store:
            return None

        # Get all knowledge summaries
        stats = await self.store.get_stats()
        if stats.get("total_entries", 0) == 0:
            return None

        # Get skill entries
        skills = await self.store.get_by_category("technology")
        skill_summaries = [s.summary for s in skills[:20]]

        # Get experience entries
        experiences = await self.store.get_by_category("business")
        exp_summaries = [e.summary for e in experiences[:10]]

        all_summaries = skill_summaries + exp_summaries

        if not all_summaries:
            return None

        # Synthesize profile
        profile = await self.llm.synthesize_profile(all_summaries)

        if not profile:
            return None

        # Create profile entry
        profile_summary = profile.get("summary", "")
        if not profile_summary:
            expertise = profile.get("expertise", [])[:5]
            roles = profile.get("roles", [])[:3]
            profile_summary = (
                f"Professional with expertise in {', '.join(expertise)}. Experience as {', '.join(roles)}."
            )

        entry = KnowledgeEntry(
            knowledge_type=KnowledgeType.FACT,
            summary=profile_summary[:500],
            category="profile",
            subcategory="personal",
            entities=profile.get("expertise", []) + profile.get("technologies", []),
            tags=["profile", "synthesized"],
            confidence=0.9,
        )

        await self.store.store(entry)
        print(f"  Created profile: {profile_summary[:100]}...")
        return entry

    def _map_type(self, type_str: str) -> KnowledgeType:
        """Map string type to KnowledgeType."""
        mapping = {
            "skill": KnowledgeType.SKILL,
            "project": KnowledgeType.PROJECT,
            "experience": KnowledgeType.EXPERIENCE,
            "contact": KnowledgeType.CONTACT,
            "fact": KnowledgeType.FACT,
            "insight": KnowledgeType.INSIGHT,
            "preference": KnowledgeType.PREFERENCE,
        }
        return mapping.get(type_str.lower(), KnowledgeType.FACT)


async def run_extraction():
    """Run knowledge extraction pipeline."""
    extractor = KnowledgeExtractor()
    await extractor.initialize()

    print("=" * 60)
    print("Knowledge Extraction Pipeline")
    print("=" * 60)

    result = await extractor.extract_all(limit=100)

    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"Entries created: {len(result.entries)}")
    print(f"Categories: {', '.join(result.categories_found)}")
    print(f"Time: {result.processing_time_ms:.0f}ms")

    # Show stats
    stats = await extractor.store.get_stats()
    print("\nKnowledge Store Stats:")
    print(f"  Total entries: {stats.get('total_entries', 0)}")
    print(f"  By type: {stats.get('by_type', {})}")

    await extractor.close()


if __name__ == "__main__":
    asyncio.run(run_extraction())
