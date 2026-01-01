"""
Enhanced Continuous Learning System for PLM.

Implements advanced learning strategies:
- Spaced repetition for memory consolidation
- Adaptive learning rates based on confidence
- Forgetting curve modeling
- Multi-strategy knowledge extraction
- Skill progression tracking
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


class LearningStrategy(str, Enum):
    """Different strategies for learning from interactions."""

    DISTILLATION = "distillation"  # Learn from teacher model responses
    CORRECTION = "correction"  # Learn from user corrections
    REINFORCEMENT = "reinforcement"  # Learn from positive feedback
    CONTRAST = "contrast"  # Learn what NOT to do from negative feedback
    CONSOLIDATION = "consolidation"  # Periodic memory consolidation


@dataclass
class LearnedConcept:
    """A concept learned from interactions."""

    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    concept_type: str = "fact"  # fact, preference, skill, belief
    confidence: float = 0.5
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    access_count: int = 1
    source_interactions: list[str] = field(default_factory=list)

    # Spaced repetition
    next_review: datetime = field(default_factory=datetime.now)
    review_interval_days: float = 1.0
    ease_factor: float = 2.5  # SM-2 ease factor

    def calculate_strength(self) -> float:
        """Calculate memory strength using forgetting curve."""
        # Ebbinghaus forgetting curve: R = e^(-t/S)
        # Where t = time since last seen, S = stability
        days_since = (datetime.now() - self.last_seen).days
        stability = self.review_interval_days * self.ease_factor
        retention = math.exp(-days_since / max(stability, 0.1))
        return retention * self.confidence

    def update_for_review(self, quality: int) -> None:
        """
        Update spaced repetition parameters after review.

        Args:
            quality: Review quality (0-5, where 5 is perfect recall)
        """
        # SM-2 algorithm
        self.access_count += 1
        self.last_seen = datetime.now()

        if quality >= 3:
            # Successful recall
            if self.access_count == 1:
                self.review_interval_days = 1
            elif self.access_count == 2:
                self.review_interval_days = 6
            else:
                self.review_interval_days *= self.ease_factor

            self.ease_factor = max(
                1.3,
                self.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            )
        else:
            # Failed recall - reset
            self.review_interval_days = 1
            self.ease_factor = max(1.3, self.ease_factor - 0.2)

        self.next_review = datetime.now() + timedelta(days=self.review_interval_days)
        self.confidence = min(1.0, self.confidence * (0.8 + quality * 0.04))


@dataclass
class SkillProgression:
    """Tracks progression in a specific skill domain."""

    skill_name: str
    level: int = 1  # 1-10
    experience_points: float = 0.0
    interactions: int = 0
    successful_applications: int = 0
    last_practiced: datetime = field(default_factory=datetime.now)
    subskills: dict[str, float] = field(default_factory=dict)

    def add_experience(self, points: float, success: bool = True) -> bool:
        """
        Add experience and potentially level up.

        Returns True if leveled up.
        """
        self.experience_points += points
        self.interactions += 1
        if success:
            self.successful_applications += 1
        self.last_practiced = datetime.now()

        # Level up threshold: level * 100 XP
        threshold = self.level * 100
        if self.experience_points >= threshold and self.level < 10:
            self.experience_points -= threshold
            self.level += 1
            return True
        return False


class EnhancedLearner:
    """
    Enhanced continuous learning with multiple strategies.

    Key features:
    1. Multi-strategy extraction (facts, preferences, skills)
    2. Spaced repetition for memory consolidation
    3. Adaptive confidence based on corroboration
    4. Skill progression tracking
    5. Memory consolidation (simulating sleep)
    """

    def __init__(self, data_dir: str = "~/memory/data"):
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self._concepts: dict[str, LearnedConcept] = {}
        self._skills: dict[str, SkillProgression] = {}
        self._pending_reviews: list[str] = []

        # Pattern extractors
        self._preference_patterns = [
            r"i\s+(?:prefer|like|love|enjoy|want)\s+(.+?)(?:\.|$|,)",
            r"i\s+(?:don't like|hate|dislike|avoid)\s+(.+?)(?:\.|$|,)",
            r"(?:always|usually|typically)\s+(?:use|do|choose)\s+(.+?)(?:\.|$|,)",
            r"(?:never|rarely)\s+(?:use|do|choose)\s+(.+?)(?:\.|$|,)",
        ]

        self._fact_patterns = [
            r"(?:my|i)\s+(?:am|work as|live in|have)\s+(.+?)(?:\.|$|,)",
            r"(?:i've been|i have been)\s+(.+?)(?:\s+for|$|,)",
            r"(?:my name is|i'm called)\s+(.+?)(?:\.|$|,)",
        ]

        self._skill_patterns = [
            r"(?:i can|i know how to|i'm good at)\s+(.+?)(?:\.|$|,)",
            r"(?:i've learned|i understand)\s+(.+?)(?:\.|$|,)",
            r"(?:expert in|proficient with)\s+(.+?)(?:\.|$|,)",
        ]

        # Load existing data
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted learning state."""
        concepts_file = self.data_dir / "learned_concepts.json"
        if concepts_file.exists():
            with open(concepts_file) as f:
                data = json.load(f)
                for c in data.get("concepts", []):
                    concept = LearnedConcept(
                        id=c["id"],
                        content=c["content"],
                        concept_type=c["concept_type"],
                        confidence=c["confidence"],
                        first_seen=datetime.fromisoformat(c["first_seen"]),
                        last_seen=datetime.fromisoformat(c["last_seen"]),
                        access_count=c["access_count"],
                        review_interval_days=c.get("review_interval_days", 1.0),
                        ease_factor=c.get("ease_factor", 2.5),
                    )
                    self._concepts[concept.id] = concept

        skills_file = self.data_dir / "skill_progressions.json"
        if skills_file.exists():
            with open(skills_file) as f:
                data = json.load(f)
                for s in data.get("skills", []):
                    skill = SkillProgression(
                        skill_name=s["skill_name"],
                        level=s["level"],
                        experience_points=s["experience_points"],
                        interactions=s["interactions"],
                        successful_applications=s["successful_applications"],
                        subskills=s.get("subskills", {}),
                    )
                    self._skills[skill.skill_name] = skill

    def _save_state(self) -> None:
        """Persist learning state."""
        concepts_data = {
            "concepts": [
                {
                    "id": c.id,
                    "content": c.content,
                    "concept_type": c.concept_type,
                    "confidence": c.confidence,
                    "first_seen": c.first_seen.isoformat(),
                    "last_seen": c.last_seen.isoformat(),
                    "access_count": c.access_count,
                    "review_interval_days": c.review_interval_days,
                    "ease_factor": c.ease_factor,
                }
                for c in self._concepts.values()
            ]
        }

        skills_data = {
            "skills": [
                {
                    "skill_name": s.skill_name,
                    "level": s.level,
                    "experience_points": s.experience_points,
                    "interactions": s.interactions,
                    "successful_applications": s.successful_applications,
                    "subskills": s.subskills,
                }
                for s in self._skills.values()
            ]
        }

        with open(self.data_dir / "learned_concepts.json", "w") as f:
            json.dump(concepts_data, f, indent=2)

        with open(self.data_dir / "skill_progressions.json", "w") as f:
            json.dump(skills_data, f, indent=2)

    async def learn_from_interaction(
        self,
        user_message: str,
        assistant_response: str,
        feedback: str | None = None,
        domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Extract and store learning from an interaction.

        Returns summary of what was learned.
        """
        results = {
            "preferences": [],
            "facts": [],
            "skills": [],
            "skill_updates": [],
        }

        # Extract preferences
        for pattern in self._preference_patterns:
            matches = re.findall(pattern, user_message.lower(), re.IGNORECASE)
            for match in matches:
                concept = await self._add_or_update_concept(
                    content=match.strip(),
                    concept_type="preference",
                    confidence=0.7 if feedback == "good" else 0.5,
                )
                results["preferences"].append(concept.content)

        # Extract facts
        for pattern in self._fact_patterns:
            matches = re.findall(pattern, user_message.lower(), re.IGNORECASE)
            for match in matches:
                concept = await self._add_or_update_concept(
                    content=match.strip(),
                    concept_type="fact",
                    confidence=0.8,  # Facts from user are high confidence
                )
                results["facts"].append(concept.content)

        # Extract skills
        for pattern in self._skill_patterns:
            matches = re.findall(pattern, user_message.lower(), re.IGNORECASE)
            for match in matches:
                concept = await self._add_or_update_concept(
                    content=match.strip(),
                    concept_type="skill",
                    confidence=0.6,
                )
                results["skills"].append(concept.content)

        # Update skill progressions based on domains
        if domains:
            for domain in domains:
                if domain not in self._skills:
                    self._skills[domain] = SkillProgression(skill_name=domain)

                # Award XP based on feedback
                xp = 10 if feedback == "good" else 5 if feedback is None else 2
                success = feedback != "bad"

                if self._skills[domain].add_experience(xp, success):
                    results["skill_updates"].append({
                        "skill": domain,
                        "new_level": self._skills[domain].level,
                    })

        # Save state
        self._save_state()

        return results

    async def _add_or_update_concept(
        self,
        content: str,
        concept_type: str,
        confidence: float,
    ) -> LearnedConcept:
        """Add a new concept or update existing one."""
        # Check for similar existing concept
        content_lower = content.lower()
        for concept in self._concepts.values():
            if concept.content.lower() == content_lower:
                # Update existing
                concept.last_seen = datetime.now()
                concept.access_count += 1
                # Increase confidence with corroboration
                concept.confidence = min(1.0, concept.confidence + 0.1)
                return concept

        # Create new concept
        concept = LearnedConcept(
            content=content,
            concept_type=concept_type,
            confidence=confidence,
        )
        self._concepts[concept.id] = concept
        return concept

    async def consolidate_memories(self) -> dict[str, Any]:
        """
        Perform memory consolidation (like sleep).

        This process:
        1. Strengthens frequently accessed memories
        2. Weakens rarely accessed memories
        3. Identifies concepts ready for promotion
        4. Schedules reviews based on spaced repetition
        """
        now = datetime.now()
        results = {
            "strengthened": 0,
            "weakened": 0,
            "due_for_review": 0,
            "forgotten": 0,
        }

        to_remove = []

        for concept_id, concept in self._concepts.items():
            strength = concept.calculate_strength()

            if strength < 0.1:
                # Very weak - consider forgetting
                if concept.access_count < 2:
                    to_remove.append(concept_id)
                    results["forgotten"] += 1
                else:
                    # Weaken but keep
                    concept.confidence *= 0.9
                    results["weakened"] += 1

            elif concept.next_review <= now:
                # Due for review
                results["due_for_review"] += 1
                self._pending_reviews.append(concept_id)

            elif concept.access_count > 5 and strength > 0.8:
                # Strong memory - reinforce
                concept.confidence = min(1.0, concept.confidence * 1.05)
                results["strengthened"] += 1

        # Remove forgotten concepts
        for concept_id in to_remove:
            del self._concepts[concept_id]

        self._save_state()
        return results

    async def review_concept(
        self,
        concept_id: str,
        recalled: bool,
        quality: int = 3,
    ) -> None:
        """
        Review a concept (spaced repetition update).

        Args:
            concept_id: ID of concept to review
            recalled: Whether user recalled the concept
            quality: Quality of recall (0-5)
        """
        if concept_id not in self._concepts:
            return

        concept = self._concepts[concept_id]
        concept.update_for_review(quality if recalled else 0)

        if concept_id in self._pending_reviews:
            self._pending_reviews.remove(concept_id)

        self._save_state()

    def get_due_reviews(self, limit: int = 10) -> list[LearnedConcept]:
        """Get concepts due for review."""
        now = datetime.now()
        due = [
            c for c in self._concepts.values()
            if c.next_review <= now
        ]
        # Sort by oldest first
        due.sort(key=lambda x: x.next_review)
        return due[:limit]

    def get_skill_summary(self) -> dict[str, Any]:
        """Get summary of skill progressions."""
        return {
            skill.skill_name: {
                "level": skill.level,
                "xp": skill.experience_points,
                "interactions": skill.interactions,
                "success_rate": (
                    skill.successful_applications / skill.interactions
                    if skill.interactions > 0 else 0
                ),
            }
            for skill in sorted(
                self._skills.values(),
                key=lambda x: x.level,
                reverse=True
            )
        }

    def get_concept_summary(self) -> dict[str, Any]:
        """Get summary of learned concepts."""
        by_type = {}
        for concept in self._concepts.values():
            if concept.concept_type not in by_type:
                by_type[concept.concept_type] = []
            by_type[concept.concept_type].append({
                "content": concept.content,
                "confidence": concept.confidence,
                "strength": concept.calculate_strength(),
            })

        return {
            "total": len(self._concepts),
            "by_type": by_type,
            "pending_reviews": len(self._pending_reviews),
        }

    def get_strongest_concepts(self, limit: int = 20) -> list[LearnedConcept]:
        """Get the strongest (most confident) concepts."""
        concepts = list(self._concepts.values())
        concepts.sort(key=lambda x: x.calculate_strength(), reverse=True)
        return concepts[:limit]

    def export_for_training(self) -> list[dict[str, Any]]:
        """Export learned concepts as training data."""
        training_data = []

        for concept in self.get_strongest_concepts(100):
            if concept.concept_type == "preference":
                training_data.append({
                    "messages": [
                        {"role": "user", "content": f"What do I prefer regarding {concept.content}?"},
                        {"role": "assistant", "content": f"You prefer {concept.content}."},
                    ],
                    "weight": concept.confidence,
                })
            elif concept.concept_type == "fact":
                training_data.append({
                    "messages": [
                        {"role": "user", "content": f"What do you know about me and {concept.content}?"},
                        {"role": "assistant", "content": f"You {concept.content}."},
                    ],
                    "weight": concept.confidence,
                })
            elif concept.concept_type == "skill":
                training_data.append({
                    "messages": [
                        {"role": "user", "content": f"What skills do I have related to {concept.content}?"},
                        {"role": "assistant", "content": f"You know how to {concept.content}."},
                    ],
                    "weight": concept.confidence,
                })

        return training_data
