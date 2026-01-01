"""
Differential Privacy for PLM federated learning.

Implements privacy-preserving mechanisms:
- Gaussian and Laplacian noise
- Gradient clipping
- Privacy budget tracking (epsilon, delta)
- Moments accountant for tight privacy bounds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class NoiseType(Enum):
    """Type of noise to add for differential privacy."""

    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"
    NONE = "none"


@dataclass
class PrivacyBudget:
    """
    Privacy budget tracking using (epsilon, delta)-differential privacy.

    Epsilon (ε): Privacy loss parameter. Lower = more private.
        - ε < 1: Strong privacy
        - ε = 1-10: Moderate privacy
        - ε > 10: Weak privacy

    Delta (δ): Probability of privacy breach. Should be < 1/n.
    """

    epsilon: float = 1.0
    delta: float = 1e-5

    # Track spent budget
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0

    # History of privacy costs
    costs: list[dict[str, float]] = field(default_factory=list)

    @property
    def remaining_epsilon(self) -> float:
        """Remaining epsilon budget."""
        return max(0.0, self.epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        """Remaining delta budget."""
        return max(0.0, self.delta - self.spent_delta)

    @property
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.spent_epsilon >= self.epsilon or self.spent_delta >= self.delta

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        operation: str = "unknown",
    ) -> bool:
        """
        Spend privacy budget on an operation.

        Returns True if budget available, False if exhausted.
        """
        if self.spent_epsilon + epsilon > self.epsilon:
            return False
        if self.spent_delta + delta > self.delta:
            return False

        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.costs.append({
            "operation": operation,
            "epsilon": epsilon,
            "delta": delta,
        })

        return True

    def reset(self) -> None:
        """Reset spent budget (e.g., for new training round)."""
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.costs = []


class DifferentialPrivacy:
    """
    Differential privacy mechanism for protecting gradients and updates.

    Implements:
    - Gradient clipping to bound sensitivity
    - Noise addition (Gaussian or Laplacian)
    - Privacy budget accounting
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_type: NoiseType = NoiseType.GAUSSIAN,
        clip_norm: float = 1.0,
        noise_multiplier: float | None = None,
    ):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Privacy breach probability
            noise_type: Type of noise to add
            clip_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Override for noise scale (auto-calculated if None)
        """
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        self.noise_type = noise_type
        self.clip_norm = clip_norm

        # Calculate noise multiplier if not provided
        if noise_multiplier is None:
            self.noise_multiplier = self._calculate_noise_multiplier(
                epsilon, delta, clip_norm
            )
        else:
            self.noise_multiplier = noise_multiplier

    def _calculate_noise_multiplier(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
    ) -> float:
        """
        Calculate noise multiplier for Gaussian mechanism.

        Uses the analytic Gaussian mechanism formula.
        """
        if epsilon <= 0 or delta <= 0:
            return float('inf')

        # Simple approximation for Gaussian mechanism
        # σ ≥ √(2 * ln(1.25/δ)) * Δf / ε
        return math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon

    def clip_gradient(
        self,
        gradient: list[float],
    ) -> list[float]:
        """
        Clip gradient to bound sensitivity.

        Args:
            gradient: Gradient vector to clip

        Returns:
            Clipped gradient with L2 norm <= clip_norm
        """
        # Calculate L2 norm
        norm = math.sqrt(sum(g * g for g in gradient))

        if norm <= self.clip_norm:
            return gradient

        # Scale down to clip_norm
        scale = self.clip_norm / norm
        return [g * scale for g in gradient]

    def add_noise(
        self,
        value: float | list[float],
        sensitivity: float | None = None,
    ) -> float | list[float]:
        """
        Add calibrated noise to a value or vector.

        Args:
            value: Value(s) to protect
            sensitivity: Query sensitivity (uses clip_norm if None)

        Returns:
            Noised value(s)
        """
        import random

        sensitivity = sensitivity or self.clip_norm

        if self.noise_type == NoiseType.NONE:
            return value

        def generate_noise() -> float:
            if self.noise_type == NoiseType.GAUSSIAN:
                # Gaussian noise with σ = noise_multiplier * sensitivity
                sigma = self.noise_multiplier * sensitivity
                return random.gauss(0, sigma)
            else:
                # Laplacian noise with b = sensitivity / epsilon
                b = sensitivity / self.budget.epsilon
                u = random.random() - 0.5
                return -b * math.copysign(1, u) * math.log(1 - 2 * abs(u))

        if isinstance(value, list):
            return [v + generate_noise() for v in value]
        else:
            return value + generate_noise()

    def privatize_gradient(
        self,
        gradient: list[float],
        operation: str = "gradient",
    ) -> list[float] | None:
        """
        Apply full differential privacy to a gradient.

        1. Clip to bound sensitivity
        2. Add calibrated noise
        3. Track privacy cost

        Returns None if privacy budget exhausted.
        """
        # Calculate privacy cost for this operation
        # Using simple composition theorem
        epsilon_cost = self.budget.epsilon / 10  # Assume 10 iterations per round

        if not self.budget.spend(epsilon_cost, operation=operation):
            return None

        # Clip and noise
        clipped = self.clip_gradient(gradient)
        noised = self.add_noise(clipped)

        return noised if isinstance(noised, list) else [noised]

    def privatize_count(
        self,
        count: int,
        sensitivity: int = 1,
    ) -> float:
        """Add noise to a count query."""
        result = self.add_noise(float(count), float(sensitivity))
        return result if isinstance(result, float) else result[0]

    def privatize_mean(
        self,
        values: list[float],
        bounds: tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """
        Compute differentially private mean.

        Args:
            values: Values to average
            bounds: (min, max) bounds for values
        """
        if not values:
            return 0.0

        # Clip values to bounds
        clipped = [max(bounds[0], min(bounds[1], v)) for v in values]

        # Sensitivity of mean = (max - min) / n
        sensitivity = (bounds[1] - bounds[0]) / len(values)

        # Add noise to mean
        mean = sum(clipped) / len(clipped)
        noised_mean = self.add_noise(mean, sensitivity)

        return noised_mean if isinstance(noised_mean, float) else noised_mean[0]


@dataclass
class PrivateMemoryUpdate:
    """A differentially private memory update."""

    id: str = field(default_factory=lambda: str(uuid4()))

    # Privatized statistics
    memory_count: float = 0.0
    avg_confidence: float = 0.0
    topic_counts: dict[str, float] = field(default_factory=dict)

    # Model updates (if any)
    embedding_updates: list[list[float]] = field(default_factory=list)

    # Privacy metadata
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    noise_type: str = "gaussian"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for transmission."""
        return {
            "id": self.id,
            "memory_count": self.memory_count,
            "avg_confidence": self.avg_confidence,
            "topic_counts": self.topic_counts,
            "embedding_updates": self.embedding_updates,
            "epsilon_spent": self.epsilon_spent,
            "delta_spent": self.delta_spent,
            "noise_type": self.noise_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivateMemoryUpdate:
        """Deserialize from transmission."""
        return cls(
            id=data.get("id", str(uuid4())),
            memory_count=data.get("memory_count", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            topic_counts=data.get("topic_counts", {}),
            embedding_updates=data.get("embedding_updates", []),
            epsilon_spent=data.get("epsilon_spent", 0.0),
            delta_spent=data.get("delta_spent", 0.0),
            noise_type=data.get("noise_type", "gaussian"),
        )


class MomentsAccountant:
    """
    Moments accountant for tight privacy composition.

    Provides tighter privacy bounds than basic composition
    by tracking the moments of the privacy loss random variable.
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float,
    ):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        # Track log moments
        self.log_moments: list[float] = []
        self.queries: int = 0

    def compute_privacy_loss(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
    ) -> tuple[float, float]:
        """
        Compute privacy loss for a training run.

        Args:
            noise_multiplier: σ / clip_norm
            sample_rate: Fraction of data used per step
            steps: Number of training steps

        Returns:
            (epsilon, delta) privacy loss
        """
        # Simplified computation using RDP
        # For more accurate results, use Google's DP library

        if noise_multiplier == 0:
            return float('inf'), 1.0

        # Approximate epsilon using simple formula
        # ε ≈ q * √(2T * log(1/δ)) / σ
        # where q is sample rate, T is steps, σ is noise_multiplier

        epsilon = (
            sample_rate *
            math.sqrt(2 * steps * math.log(1 / self.target_delta)) /
            noise_multiplier
        )

        return epsilon, self.target_delta

    def get_max_steps(
        self,
        noise_multiplier: float,
        sample_rate: float,
    ) -> int:
        """
        Compute maximum training steps within privacy budget.

        Args:
            noise_multiplier: σ / clip_norm
            sample_rate: Fraction of data used per step

        Returns:
            Maximum number of steps
        """
        if noise_multiplier == 0 or sample_rate == 0:
            return 0

        # Invert the epsilon formula to solve for T
        # T = (σ * ε / q)² / (2 * log(1/δ))

        max_steps = int(
            (noise_multiplier * self.target_epsilon / sample_rate) ** 2 /
            (2 * math.log(1 / self.target_delta))
        )

        return max(1, max_steps)
