#!/usr/bin/env python3
"""
PLM Content Synthesis Pipeline

This script processes all ingested content and derives meaningful insights:
1. Summarizes email content
2. Extracts key facts and experiences
3. Builds a comprehensive personal profile
4. Creates searchable knowledge entries
"""

import asyncio
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx


@dataclass
class PersonalProfile:
    """Synthesized personal profile from all content."""

    name: str = "Rod Higgins"
    expertise: list[str] = None
    roles: list[str] = None
    organizations: list[str] = None
    technologies: list[str] = None
    industries: list[str] = None
    locations: list[str] = None
    key_projects: list[str] = None
    contacts: list[str] = None

    def __post_init__(self):
        self.expertise = self.expertise or []
        self.roles = self.roles or []
        self.organizations = self.organizations or []
        self.technologies = self.technologies or []
        self.industries = self.industries or []
        self.locations = self.locations or []
        self.key_projects = self.key_projects or []
        self.contacts = self.contacts or []


class OllamaSynthesizer:
    """Uses Ollama to synthesize insights from content."""

    def __init__(self, model: str = "tinyllama"):
        self.model = model
        self.base_url = "http://localhost:11434"

    async def complete(self, prompt: str, max_tokens: int = 500) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": 0.3},
                    },
                    timeout=90.0,
                )
                response.raise_for_status()
                return response.json()["response"].strip()
            except Exception as e:
                print(f"Ollama error: {e}")
                return ""

    async def get_embedding(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text[:2000]},
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except Exception:
                return []

    async def summarize_email(self, content: str) -> str:
        """Create a meaningful summary of an email."""
        prompt = f"""Summarize this email in one sentence. Focus on the key action, decision, or information:

{content[:2000]}

One sentence summary:"""
        return await self.complete(prompt, max_tokens=100)

    async def extract_facts(self, content: str) -> list[str]:
        """Extract key facts from content."""
        prompt = f"""Extract 3-5 key facts from this content. Be specific and factual:

{content[:2000]}

Key facts (one per line):"""
        result = await self.complete(prompt, max_tokens=200)
        facts = [f.strip("- •").strip() for f in result.split("\n") if f.strip()]
        return facts[:5]

    async def extract_profile_info(self, emails: list[str]) -> dict:
        """Extract profile information from a batch of emails."""
        combined = "\n---\n".join(emails[:10])

        prompt = f"""Based on these email snippets, extract information about the person who wrote them:

{combined[:4000]}

Extract in JSON format:
{{
  "expertise": ["list of skills/technologies"],
  "roles": ["job titles mentioned"],
  "organizations": ["companies/organizations mentioned"],
  "technologies": ["specific technologies used"],
  "locations": ["places mentioned"]
}}

JSON:"""
        result = await self.complete(prompt, max_tokens=400)

        try:
            # Extract JSON from response
            if "{" in result:
                start = result.find("{")
                end = result.rfind("}") + 1
                if end > start:
                    return json.loads(result[start:end])
        except json.JSONDecodeError:
            pass

        return {}


class PLMSynthesizer:
    """Main PLM synthesis engine."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.llm = OllamaSynthesizer()
        self.profile = PersonalProfile()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    async def run_full_synthesis(self):
        """Run complete synthesis pipeline."""
        print("=" * 60)
        print("PLM Content Synthesis Pipeline")
        print("=" * 60)

        # Step 1: Analyze all emails and extract insights
        await self.synthesize_emails()

        # Step 2: Build personal profile
        await self.build_profile()

        # Step 3: Create derived knowledge entries
        await self.create_knowledge_entries()

        # Step 4: Generate embeddings for search
        await self.generate_embeddings()

        print("\n" + "=" * 60)
        print("Synthesis Complete!")
        print("=" * 60)

    async def synthesize_emails(self):
        """Process all emails and create summaries."""
        print("\n[1/4] Synthesizing email content...")

        conn = self.get_connection()
        cursor = conn.execute("""
            SELECT id, content, summary, tags_json
            FROM memories
            WHERE domains_json LIKE '%email%'
            AND (summary IS NULL OR length(summary) < 30)
            LIMIT 200
        """)

        rows = cursor.fetchall()
        print(f"  Found {len(rows)} emails needing summarization")

        processed = 0
        for row in rows:
            content = row["content"]

            # Skip very short content
            if len(content) < 100:
                continue

            # Generate summary
            summary = await self.llm.summarize_email(content)

            if summary and len(summary) > 20:
                conn.execute(
                    "UPDATE memories SET summary = ? WHERE id = ?",
                    (summary, row["id"])
                )
                processed += 1

                if processed % 20 == 0:
                    print(f"    Processed {processed} emails...")
                    conn.commit()

        conn.commit()
        conn.close()
        print(f"  Summarized {processed} emails")

    async def build_profile(self):
        """Build comprehensive personal profile from all content."""
        print("\n[2/4] Building personal profile...")

        conn = self.get_connection()

        # Get sent emails (user-authored content)
        cursor = conn.execute("""
            SELECT content FROM memories
            WHERE tags_json LIKE '%sent%'
            AND domains_json LIKE '%email%'
            ORDER BY created_at DESC
            LIMIT 50
        """)

        sent_emails = [row["content"][:500] for row in cursor.fetchall()]

        if sent_emails:
            print(f"  Analyzing {len(sent_emails)} sent emails...")

            # Extract profile info in batches
            for i in range(0, len(sent_emails), 10):
                batch = sent_emails[i:i + 10]
                info = await self.llm.extract_profile_info(batch)

                # Merge into profile
                self.profile.expertise.extend(info.get("expertise", []))
                self.profile.roles.extend(info.get("roles", []))
                self.profile.organizations.extend(info.get("organizations", []))
                self.profile.technologies.extend(info.get("technologies", []))
                self.profile.locations.extend(info.get("locations", []))

        # Deduplicate
        self.profile.expertise = list(set(self.profile.expertise))[:20]
        self.profile.roles = list(set(self.profile.roles))[:10]
        self.profile.organizations = list(set(self.profile.organizations))[:15]
        self.profile.technologies = list(set(self.profile.technologies))[:20]
        self.profile.locations = list(set(self.profile.locations))[:10]

        conn.close()

        print(f"  Extracted profile:")
        print(f"    - {len(self.profile.expertise)} expertise areas")
        print(f"    - {len(self.profile.roles)} roles")
        print(f"    - {len(self.profile.organizations)} organizations")
        print(f"    - {len(self.profile.technologies)} technologies")

    async def create_knowledge_entries(self):
        """Create derived knowledge entries from synthesis."""
        print("\n[3/4] Creating knowledge entries...")

        conn = self.get_connection()

        # Create profile memory
        if self.profile.expertise or self.profile.technologies:
            profile_content = f"""Personal Profile for {self.profile.name}:

EXPERTISE: {', '.join(self.profile.expertise[:10])}

ROLES: {', '.join(self.profile.roles[:5])}

TECHNOLOGIES: {', '.join(self.profile.technologies[:15])}

ORGANIZATIONS: {', '.join(self.profile.organizations[:10])}

LOCATIONS: {', '.join(self.profile.locations[:5])}
"""
            import uuid
            profile_id = str(uuid.uuid4())

            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, summary, memory_type, tier, truth_category,
                 domains_json, tags_json, confidence_overall, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile_id,
                profile_content,
                f"Professional profile: {', '.join(self.profile.expertise[:3])}",
                "fact",
                "persistent",
                "absolute",
                json.dumps(["personal", "profile", "identity"]),
                json.dumps(["synthesized", "profile", "expertise"]),
                0.95,
                1,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))

            print("  Created personal profile entry")

        # Create expertise entries
        for tech in self.profile.technologies[:10]:
            tech_id = str(uuid.uuid4())
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, content, summary, memory_type, tier, truth_category,
                 domains_json, tags_json, confidence_overall, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tech_id,
                f"Has experience with {tech} based on email communications and project work.",
                f"Technology expertise: {tech}",
                "skill",
                "persistent",
                "inferred",
                json.dumps(["technology", "skills", tech.lower()]),
                json.dumps(["synthesized", "skill", tech.lower()]),
                0.8,
                1,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ))

        conn.commit()
        conn.close()
        print(f"  Created {len(self.profile.technologies[:10])} technology expertise entries")

    async def generate_embeddings(self):
        """Generate embeddings for semantic search."""
        print("\n[4/4] Generating embeddings for search...")

        conn = self.get_connection()

        # Get memories without embeddings (using summary for embedding)
        cursor = conn.execute("""
            SELECT id, summary, content
            FROM memories
            WHERE summary IS NOT NULL AND length(summary) > 20
            AND embedding IS NULL
            LIMIT 500
        """)

        rows = cursor.fetchall()
        print(f"  Processing {len(rows)} memories for embeddings...")

        processed = 0
        for row in rows:
            text = row["summary"] or row["content"][:500]
            embedding = await self.llm.get_embedding(text)

            if embedding:
                conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (json.dumps(embedding), row["id"])
                )
                processed += 1

                if processed % 50 == 0:
                    print(f"    Generated {processed} embeddings...")
                    conn.commit()

        conn.commit()
        conn.close()
        print(f"  Generated {processed} embeddings")


async def main():
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "~/memory/data/persistent/core.sqlite"

    synthesizer = PLMSynthesizer(db_path)
    await synthesizer.run_full_synthesis()

    # Print final profile
    print("\n" + "=" * 60)
    print("SYNTHESIZED PERSONAL PROFILE")
    print("=" * 60)
    print(f"\nName: {synthesizer.profile.name}")
    print(f"\nExpertise: {', '.join(synthesizer.profile.expertise[:10])}")
    print(f"\nRoles: {', '.join(synthesizer.profile.roles[:5])}")
    print(f"\nTechnologies: {', '.join(synthesizer.profile.technologies[:15])}")
    print(f"\nOrganizations: {', '.join(synthesizer.profile.organizations[:10])}")


if __name__ == "__main__":
    asyncio.run(main())
