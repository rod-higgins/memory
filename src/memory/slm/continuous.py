"""
Continuous Learning System for Personal SLM.

Implements Google Titan-style continuous learning where the model
improves with each interaction. Combines:
- Knowledge distillation from larger models
- User-specific pruning
- Incremental fine-tuning
- ZipLLM compression techniques
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class LearningEvent:
    """A single learning event from an interaction."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # The interaction
    user_query: str = ""
    assistant_response: str = ""
    teacher_model: str = "claude"  # Which model provided the response

    # Extracted learning
    learned_facts: list[str] = field(default_factory=list)
    learned_preferences: list[str] = field(default_factory=list)
    domains_activated: list[str] = field(default_factory=list)

    # Quality signals
    user_feedback: str | None = None  # "good", "bad", "corrected"
    correction: str | None = None  # If user corrected the response

    # For distillation
    distillation_target: str | None = None  # Formatted for training


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning."""

    # Learning triggers
    min_interactions_before_update: int = 10
    min_hours_between_updates: float = 24.0

    # Distillation
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5  # Balance between hard/soft targets

    # Pruning
    pruning_threshold: float = 0.01  # Remove weights below this importance
    prune_after_n_updates: int = 5

    # Compression
    target_model_size_mb: int = 500
    quantization_bits: int = 4

    # Storage
    learning_log_path: str = "~/memory/data/learning_log.jsonl"
    model_checkpoints_path: str = "~/memory/models/checkpoints"


class ContinuousLearner:
    """
    Continuous learning system for the Personal SLM.

    Implements Titan-style memory where each interaction can update
    the model's understanding. Key innovations:

    1. **Interaction Logging**: Every interaction is logged with metadata
    2. **Knowledge Distillation**: Learn from Claude/GPT responses
    3. **Incremental Updates**: Small, frequent updates vs full retraining
    4. **User-Specific Pruning**: Remove pathways unused by this user
    5. **Compression**: Maintain small footprint via ZipLLM techniques
    """

    def __init__(self, config: ContinuousLearningConfig | None = None):
        self._config = config or ContinuousLearningConfig()
        self._log_path = Path(self._config.learning_log_path).expanduser()
        self._checkpoints_path = Path(self._config.model_checkpoints_path).expanduser()

        self._pending_events: list[LearningEvent] = []
        self._update_count = 0
        self._last_update: datetime | None = None

        # Domain activation tracking for pruning
        self._domain_activations: dict[str, int] = {}

        # Ensure directories exist
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoints_path.mkdir(parents=True, exist_ok=True)

    async def record_interaction(
        self,
        user_query: str,
        assistant_response: str,
        teacher_model: str = "claude",
        domains: list[str] | None = None,
    ) -> LearningEvent:
        """
        Record an interaction for future learning.

        This is called after each interaction with an LLM.
        """
        event = LearningEvent(
            user_query=user_query,
            assistant_response=assistant_response,
            teacher_model=teacher_model,
            domains_activated=domains or [],
        )

        # Extract learning from the interaction
        event = await self._extract_learning(event)

        # Log the event
        self._log_event(event)

        # Track domain activations for pruning decisions
        for domain in event.domains_activated:
            self._domain_activations[domain] = self._domain_activations.get(domain, 0) + 1

        # Add to pending events
        self._pending_events.append(event)

        # Check if we should trigger an update
        if self._should_update():
            await self.trigger_update()

        return event

    async def record_feedback(
        self,
        event_id: str,
        feedback: str,
        correction: str | None = None,
    ) -> None:
        """
        Record user feedback on a previous interaction.

        Feedback significantly impacts learning - corrections are
        given high weight in the next update.
        """
        # Find the event
        for event in self._pending_events:
            if event.id == event_id:
                event.user_feedback = feedback
                event.correction = correction

                # Re-log with feedback
                self._log_event(event)

                # Corrections trigger faster updates
                if correction:
                    await self._handle_correction(event)

                break

    async def _extract_learning(self, event: LearningEvent) -> LearningEvent:
        """Extract learnable content from an interaction."""
        # Look for explicit preferences
        query_lower = event.user_query.lower()

        if any(word in query_lower for word in ["prefer", "like", "want", "always", "never"]):
            event.learned_preferences.append(event.user_query)

        # Format for distillation training
        event.distillation_target = self._format_for_distillation(event)

        return event

    def _format_for_distillation(self, event: LearningEvent) -> str:
        """Format an interaction for knowledge distillation."""
        # Create training example format
        return json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "You are a personal AI assistant with deep knowledge of the user."},
                    {"role": "user", "content": event.user_query},
                    {"role": "assistant", "content": event.assistant_response},
                ],
                "metadata": {
                    "teacher": event.teacher_model,
                    "domains": event.domains_activated,
                    "feedback": event.user_feedback,
                },
            }
        )

    def _log_event(self, event: LearningEvent) -> None:
        """Append event to the learning log."""
        with open(self._log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": event.id,
                        "timestamp": event.timestamp,
                        "user_query": event.user_query,
                        "assistant_response": event.assistant_response[:500],  # Truncate for log
                        "teacher_model": event.teacher_model,
                        "learned_facts": event.learned_facts,
                        "learned_preferences": event.learned_preferences,
                        "domains_activated": event.domains_activated,
                        "user_feedback": event.user_feedback,
                        "has_correction": event.correction is not None,
                    }
                )
                + "\n"
            )

    def _should_update(self) -> bool:
        """Determine if we should trigger a model update."""
        # Check interaction count
        if len(self._pending_events) < self._config.min_interactions_before_update:
            return False

        # Check time since last update
        if self._last_update:
            hours_since = (datetime.now() - self._last_update).total_seconds() / 3600
            if hours_since < self._config.min_hours_between_updates:
                return False

        return True

    async def trigger_update(self) -> dict[str, Any]:
        """
        Trigger an incremental model update.

        This performs:
        1. Knowledge distillation from pending interactions
        2. Incremental fine-tuning
        3. Optional pruning (every N updates)
        """
        if not self._pending_events:
            return {"status": "no_pending_events"}

        results = {
            "events_processed": len(self._pending_events),
            "distillation": None,
            "pruning": None,
            "compression": None,
        }

        # Prepare training data from pending events
        training_data = await self._prepare_training_data()
        results["training_examples"] = len(training_data)

        # Perform incremental distillation/fine-tuning
        distill_result = await self._perform_distillation(training_data)
        results["distillation"] = distill_result

        # Periodic pruning
        self._update_count += 1
        if self._update_count % self._config.prune_after_n_updates == 0:
            prune_result = await self._perform_pruning()
            results["pruning"] = prune_result

        # Clear pending events
        self._pending_events = []
        self._last_update = datetime.now()

        return results

    async def _prepare_training_data(self) -> list[dict[str, Any]]:
        """Prepare training data from pending events."""
        training_data = []

        for event in self._pending_events:
            if event.distillation_target:
                data = json.loads(event.distillation_target)

                # Weight corrections higher
                if event.correction:
                    data["weight"] = 2.0
                    # Use correction as target instead
                    data["messages"][-1]["content"] = event.correction
                elif event.user_feedback == "good":
                    data["weight"] = 1.5
                elif event.user_feedback == "bad":
                    data["weight"] = 0.5
                else:
                    data["weight"] = 1.0

                training_data.append(data)

        return training_data

    async def _perform_distillation(
        self,
        training_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Perform knowledge distillation update.

        Uses soft targets from teacher model responses to update
        the student (personal SLM).
        """
        # Save training data for the trainer
        output_path = self._checkpoints_path / f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        with open(output_path, "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")

        return {
            "status": "prepared",
            "training_file": str(output_path),
            "examples": len(training_data),
            "note": "Run trainer.incremental_update() to apply",
        }

    async def _perform_pruning(self) -> dict[str, Any]:
        """
        Perform user-specific pruning.

        Removes model weights that are unused by this user's
        interaction patterns.
        """
        # Analyze domain activations
        total_activations = sum(self._domain_activations.values())
        if total_activations == 0:
            return {"status": "skipped", "reason": "no_activation_data"}

        # Find domains to potentially prune
        domain_ratios = {domain: count / total_activations for domain, count in self._domain_activations.items()}

        active_domains = [d for d, r in domain_ratios.items() if r >= 0.01]
        inactive_domains = [d for d, r in domain_ratios.items() if r < 0.01]

        return {
            "status": "analyzed",
            "active_domains": active_domains,
            "inactive_domains": inactive_domains,
            "recommendation": f"Consider pruning weights related to: {inactive_domains[:5]}",
            "note": "Actual pruning requires model access",
        }

    async def _handle_correction(self, event: LearningEvent) -> None:
        """Handle a user correction with priority learning."""
        # Corrections are important - create a high-priority training example
        correction_data = {
            "messages": [
                {"role": "system", "content": "You are a personal AI assistant. Learn from this correction."},
                {"role": "user", "content": event.user_query},
                {
                    "role": "assistant",
                    "content": event.correction,  # Use the corrected response
                },
            ],
            "weight": 3.0,  # High weight for corrections
            "is_correction": True,
        }

        # Save immediately for priority learning
        corrections_path = self._checkpoints_path / "corrections.jsonl"
        with open(corrections_path, "a") as f:
            f.write(json.dumps(correction_data) + "\n")

    def get_learning_stats(self) -> dict[str, Any]:
        """Get statistics about the learning system."""
        return {
            "pending_events": len(self._pending_events),
            "update_count": self._update_count,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "domain_activations": dict(sorted(self._domain_activations.items(), key=lambda x: x[1], reverse=True)[:10]),
            "log_path": str(self._log_path),
        }


class ZipLLMCompressor:
    """
    Model compression using ZipLLM techniques.

    Combines:
    1. Structured pruning (remove entire attention heads/FFN neurons)
    2. Quantization (4-bit or 2-bit)
    3. Knowledge distillation (maintain quality)
    """

    def __init__(self, target_size_mb: int = 500):
        self._target_size_mb = target_size_mb

    async def compress_model(
        self,
        model_path: str,
        output_path: str,
    ) -> dict[str, Any]:
        """
        Compress a model using ZipLLM techniques.

        This is a placeholder - actual implementation requires
        the model weights and specific compression libraries.
        """
        return {
            "status": "not_implemented",
            "note": "Requires: torch, bitsandbytes, transformers",
            "steps": [
                "1. Load model",
                "2. Compute importance scores for each weight",
                "3. Prune low-importance weights (structured)",
                "4. Quantize remaining weights to 4-bit",
                "5. Fine-tune to recover quality (distillation)",
                "6. Save compressed model",
            ],
            "expected_compression": "3B params → ~500MB",
        }

    def estimate_compression(
        self,
        original_size_mb: int,
        pruning_ratio: float = 0.5,
        quantization_bits: int = 4,
    ) -> dict[str, Any]:
        """Estimate compressed model size."""
        # Original assumed to be 16-bit
        size_after_prune = original_size_mb * (1 - pruning_ratio)
        size_after_quant = size_after_prune * (quantization_bits / 16)

        return {
            "original_size_mb": original_size_mb,
            "after_pruning_mb": size_after_prune,
            "after_quantization_mb": size_after_quant,
            "compression_ratio": original_size_mb / size_after_quant,
        }
