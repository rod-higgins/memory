"""
Training data preparation for Personal SLM.

Converts memories into formats suitable for language model fine-tuning.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.schema.memory_entry import MemoryEntry, MemoryType


@dataclass
class TrainingExample:
    """A single training example for the SLM."""

    # Input context
    instruction: str
    context: str | None = None

    # Expected output
    response: str = ""

    # Metadata for weighting
    memory_type: str = "fact"
    truth_category: str = "inferred"
    confidence: float = 0.5
    domains: list[str] = field(default_factory=list)


class MemoryDataset:
    """
    Dataset of training examples derived from memories.

    Converts memories into instruction-tuning format for SLM training.
    """

    def __init__(self, memories: list[MemoryEntry]):
        self._memories = memories
        self._examples: list[TrainingExample] = []

    def prepare(self) -> list[TrainingExample]:
        """Prepare all training examples."""
        self._examples = []

        for memory in self._memories:
            examples = self._memory_to_examples(memory)
            self._examples.extend(examples)

        return self._examples

    def _memory_to_examples(self, memory: MemoryEntry) -> list[TrainingExample]:
        """Convert a single memory into training examples."""
        examples = []

        # Create different training patterns based on memory type
        if memory.memory_type == MemoryType.PREFERENCE:
            examples.extend(self._preference_examples(memory))
        elif memory.memory_type == MemoryType.BELIEF:
            examples.extend(self._belief_examples(memory))
        elif memory.memory_type == MemoryType.SKILL:
            examples.extend(self._skill_examples(memory))
        elif memory.memory_type == MemoryType.FACT:
            examples.extend(self._fact_examples(memory))
        else:
            examples.extend(self._general_examples(memory))

        return examples

    def _preference_examples(self, memory: MemoryEntry) -> list[TrainingExample]:
        """Generate training examples for preferences."""
        examples = []
        domain_context = ", ".join(memory.domains[:3]) if memory.domains else "general"

        # Direct recall
        examples.append(
            TrainingExample(
                instruction=f"What are the user's preferences regarding {domain_context}?",
                response=memory.content,
                memory_type="preference",
                truth_category=memory.truth_category.value,
                confidence=memory.confidence.overall,
                domains=memory.domains,
            )
        )

        # Contextual application
        examples.append(
            TrainingExample(
                instruction=f"When working with {domain_context}, what should I keep in mind about the user's preferences?",
                response=f"The user has expressed: {memory.content}",
                memory_type="preference",
                truth_category=memory.truth_category.value,
                confidence=memory.confidence.overall,
                domains=memory.domains,
            )
        )

        return examples

    def _belief_examples(self, memory: MemoryEntry) -> list[TrainingExample]:
        """Generate training examples for beliefs."""
        examples = []
        domain_context = ", ".join(memory.domains[:3]) if memory.domains else "general"

        examples.append(
            TrainingExample(
                instruction=f"What does the user believe about {domain_context}?",
                response=memory.content,
                memory_type="belief",
                truth_category=memory.truth_category.value,
                confidence=memory.confidence.overall,
                domains=memory.domains,
            )
        )

        return examples

    def _skill_examples(self, memory: MemoryEntry) -> list[TrainingExample]:
        """Generate training examples for skills."""
        examples = []

        examples.append(
            TrainingExample(
                instruction="What are the user's areas of expertise?",
                response=memory.content,
                memory_type="skill",
                truth_category=memory.truth_category.value,
                confidence=memory.confidence.overall,
                domains=memory.domains,
            )
        )

        if memory.domains:
            for domain in memory.domains[:2]:
                examples.append(
                    TrainingExample(
                        instruction=f"What is the user's experience with {domain}?",
                        response=memory.content,
                        memory_type="skill",
                        truth_category=memory.truth_category.value,
                        confidence=memory.confidence.overall,
                        domains=memory.domains,
                    )
                )

        return examples

    def _fact_examples(self, memory: MemoryEntry) -> list[TrainingExample]:
        """Generate training examples for facts."""
        examples = []

        # Only include high-confidence facts
        if memory.confidence.overall >= 0.6:
            examples.append(
                TrainingExample(
                    instruction="What relevant facts should be considered?",
                    context=", ".join(memory.domains[:3]) if memory.domains else None,
                    response=memory.content,
                    memory_type="fact",
                    truth_category=memory.truth_category.value,
                    confidence=memory.confidence.overall,
                    domains=memory.domains,
                )
            )

        return examples

    def _general_examples(self, memory: MemoryEntry) -> list[TrainingExample]:
        """Generate general training examples."""
        examples = []

        examples.append(
            TrainingExample(
                instruction="What context is relevant here?",
                context=", ".join(memory.domains[:3]) if memory.domains else None,
                response=memory.content,
                memory_type=memory.memory_type.value,
                truth_category=memory.truth_category.value,
                confidence=memory.confidence.overall,
                domains=memory.domains,
            )
        )

        return examples

    def to_jsonl(self, path: Path) -> int:
        """Export examples to JSONL format for training."""
        if not self._examples:
            self.prepare()

        count = 0
        with open(path, "w") as f:
            for example in self._examples:
                # Convert to instruction-tuning format
                item = {
                    "instruction": example.instruction,
                    "input": example.context or "",
                    "output": example.response,
                    "metadata": {
                        "memory_type": example.memory_type,
                        "truth_category": example.truth_category,
                        "confidence": example.confidence,
                        "domains": example.domains,
                    },
                }
                f.write(json.dumps(item) + "\n")
                count += 1

        return count

    def to_chat_format(self, path: Path) -> int:
        """Export examples to chat/conversation format."""
        if not self._examples:
            self.prepare()

        count = 0
        with open(path, "w") as f:
            for example in self._examples:
                # Chat format with system, user, assistant turns
                messages = [
                    {
                        "role": "system",
                        "content": "You are a personal AI assistant with deep knowledge of the user's preferences, beliefs, and context.",
                    },
                    {
                        "role": "user",
                        "content": example.instruction + (f"\n\nContext: {example.context}" if example.context else ""),
                    },
                    {"role": "assistant", "content": example.response},
                ]

                item = {"messages": messages}
                f.write(json.dumps(item) + "\n")
                count += 1

        return count

    def __len__(self) -> int:
        if not self._examples:
            self.prepare()
        return len(self._examples)

    def __iter__(self) -> Iterator[TrainingExample]:
        if not self._examples:
            self.prepare()
        return iter(self._examples)


async def prepare_training_data(
    storage_path: str = "~/memory/data",
    output_path: str = "~/memory/training",
    min_confidence: float = 0.4,
) -> dict[str, Any]:
    """
    Prepare training data from stored memories.

    Args:
        storage_path: Path to memory storage
        output_path: Path to save training data
        min_confidence: Minimum confidence threshold

    Returns:
        Statistics about the prepared data
    """
    from memory.storage.manager import StorageManager

    output_dir = Path(output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize storage
    storage = StorageManager(base_path=storage_path)
    await storage.initialize()

    try:
        # Collect all memories
        all_memories: list[MemoryEntry] = []

        for tier in ["persistent", "long_term", "short_term"]:
            store = getattr(storage, tier)
            memories = await store.list_all(limit=10000)
            all_memories.extend(memories)

        # Filter by confidence
        filtered = [m for m in all_memories if m.confidence.overall >= min_confidence]

        # Create dataset
        dataset = MemoryDataset(filtered)
        dataset.prepare()

        # Export in multiple formats
        jsonl_count = dataset.to_jsonl(output_dir / "memories.jsonl")
        chat_count = dataset.to_chat_format(output_dir / "memories_chat.jsonl")

        return {
            "total_memories": len(all_memories),
            "filtered_memories": len(filtered),
            "training_examples": len(dataset),
            "jsonl_path": str(output_dir / "memories.jsonl"),
            "chat_path": str(output_dir / "memories_chat.jsonl"),
            "jsonl_count": jsonl_count,
            "chat_count": chat_count,
        }

    finally:
        await storage.close()
