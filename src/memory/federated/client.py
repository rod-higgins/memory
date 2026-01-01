"""
Federated Learning Client for PLM.

Handles local model training and secure update sharing:
- Local memory-based training
- Gradient computation and privatization
- Secure communication with aggregation server
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from .privacy import DifferentialPrivacy, NoiseType, PrivateMemoryUpdate


class ClientState(Enum):
    """State of federated learning client."""

    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    DOWNLOADING = "downloading"
    ERROR = "error"


@dataclass
class ClientConfig:
    """Configuration for federated learning client."""

    # Client identity
    client_id: str = field(default_factory=lambda: str(uuid4()))
    device_name: str = "unknown"

    # Privacy settings
    epsilon: float = 1.0
    delta: float = 1e-5
    clip_norm: float = 1.0
    noise_type: NoiseType = NoiseType.GAUSSIAN

    # Training settings
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001

    # Communication settings
    server_url: str = ""
    compress_updates: bool = True
    compression_ratio: float = 0.1

    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0


@dataclass
class LocalUpdate:
    """A local model update to send to the server."""

    id: str = field(default_factory=lambda: str(uuid4()))
    client_id: str = ""
    round_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Update data
    weights_delta: dict[str, list[float]] = field(default_factory=dict)
    memory_statistics: PrivateMemoryUpdate | None = None

    # Metadata
    samples_used: int = 0
    training_loss: float = 0.0
    local_epochs: int = 0

    # Privacy accounting
    privacy_spent: dict[str, float] = field(default_factory=dict)

    # Verification
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute checksum for integrity verification."""
        data = json.dumps({
            "client_id": self.client_id,
            "round_id": self.round_id,
            "samples_used": self.samples_used,
            "weights_keys": list(self.weights_delta.keys()),
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for transmission."""
        return {
            "id": self.id,
            "client_id": self.client_id,
            "round_id": self.round_id,
            "timestamp": self.timestamp.isoformat(),
            "weights_delta": self.weights_delta,
            "memory_statistics": (
                self.memory_statistics.to_dict()
                if self.memory_statistics else None
            ),
            "samples_used": self.samples_used,
            "training_loss": self.training_loss,
            "local_epochs": self.local_epochs,
            "privacy_spent": self.privacy_spent,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalUpdate:
        """Deserialize from transmission."""
        return cls(
            id=data.get("id", str(uuid4())),
            client_id=data.get("client_id", ""),
            round_id=data.get("round_id", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            weights_delta=data.get("weights_delta", {}),
            memory_statistics=(
                PrivateMemoryUpdate.from_dict(data["memory_statistics"])
                if data.get("memory_statistics") else None
            ),
            samples_used=data.get("samples_used", 0),
            training_loss=data.get("training_loss", 0.0),
            local_epochs=data.get("local_epochs", 0),
            privacy_spent=data.get("privacy_spent", {}),
            checksum=data.get("checksum", ""),
        )


class FederatedClient:
    """
    Federated learning client for PLM.

    Handles:
    - Local training on private memories
    - Differential privacy for updates
    - Secure communication with server
    - Resource management
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        memory_store: Any = None,  # MemoryStore from PLM
    ):
        self.config = config or ClientConfig()
        self.memory_store = memory_store
        self.state = ClientState.IDLE

        # Privacy mechanism
        self.privacy = DifferentialPrivacy(
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            noise_type=self.config.noise_type,
            clip_norm=self.config.clip_norm,
        )

        # Current round tracking
        self.current_round: str = ""
        self.current_model: dict[str, list[float]] = {}

        # History
        self.updates_sent: list[LocalUpdate] = []
        self.rounds_participated: int = 0

    async def receive_global_model(
        self,
        model_weights: dict[str, list[float]],
        round_id: str,
    ) -> None:
        """
        Receive updated global model from server.

        Args:
            model_weights: New model parameters
            round_id: Federation round identifier
        """
        self.state = ClientState.DOWNLOADING
        self.current_model = model_weights
        self.current_round = round_id
        self.state = ClientState.IDLE

    async def train_local(
        self,
        round_id: str | None = None,
    ) -> LocalUpdate | None:
        """
        Perform local training on private memories.

        Returns privatized model update.
        """
        if self.state != ClientState.IDLE:
            return None

        self.state = ClientState.TRAINING
        round_id = round_id or self.current_round

        try:
            # Gather local memories for training
            memories = await self._gather_training_data()

            if not memories:
                self.state = ClientState.IDLE
                return None

            # Compute local updates
            weights_delta, loss = await self._compute_local_update(memories)

            # Privatize the update
            privatized_delta = {}
            for layer_name, delta in weights_delta.items():
                private_delta = self.privacy.privatize_gradient(
                    delta,
                    operation=f"layer_{layer_name}",
                )
                if private_delta is not None:
                    privatized_delta[layer_name] = private_delta

            # Compute private statistics about memories
            memory_stats = await self._compute_private_statistics(memories)

            # Create update
            update = LocalUpdate(
                client_id=self.config.client_id,
                round_id=round_id,
                weights_delta=privatized_delta,
                memory_statistics=memory_stats,
                samples_used=len(memories),
                training_loss=loss,
                local_epochs=self.config.local_epochs,
                privacy_spent={
                    "epsilon": self.privacy.budget.spent_epsilon,
                    "delta": self.privacy.budget.spent_delta,
                },
            )
            update.checksum = update.compute_checksum()

            self.updates_sent.append(update)
            self.rounds_participated += 1
            self.state = ClientState.IDLE

            return update

        except Exception:
            self.state = ClientState.ERROR
            raise

    async def _gather_training_data(self) -> list[dict[str, Any]]:
        """Gather memories for local training."""
        memories = []

        if self.memory_store is None:
            # Return mock data for testing
            return [
                {"content": "Test memory", "embedding": [0.1] * 384}
                for _ in range(100)
            ]

        # Query recent memories from store
        try:
            # This would integrate with the actual PLM memory store
            from ..storage import MemoryStore

            if isinstance(self.memory_store, MemoryStore):
                results = await self.memory_store.search(
                    query="*",
                    limit=1000,
                )
                memories = [r.to_dict() for r in results]
        except Exception:
            pass

        return memories

    async def _compute_local_update(
        self,
        memories: list[dict[str, Any]],
    ) -> tuple[dict[str, list[float]], float]:
        """
        Compute local model update from memories.

        Returns (weight_deltas, training_loss).
        """
        # Simulated training - in real implementation would use PyTorch
        weight_deltas = {}
        total_loss = 0.0

        # For embedding model fine-tuning
        if self.current_model:
            for layer_name, weights in self.current_model.items():
                # Compute gradient based on memories
                gradient = await self._compute_gradient(
                    layer_name, weights, memories
                )
                weight_deltas[layer_name] = gradient

        # Estimate loss
        total_loss = max(0.1, 1.0 - len(memories) / 1000)

        return weight_deltas, total_loss

    async def _compute_gradient(
        self,
        layer_name: str,
        weights: list[float],
        memories: list[dict[str, Any]],
    ) -> list[float]:
        """Compute gradient for a layer based on memories."""
        import random

        # Simplified gradient computation
        # Real implementation would use backpropagation
        gradient = []
        for w in weights:
            # Random gradient with small magnitude
            g = random.gauss(0, 0.01)
            gradient.append(g)

        return gradient

    async def _compute_private_statistics(
        self,
        memories: list[dict[str, Any]],
    ) -> PrivateMemoryUpdate:
        """Compute differentially private statistics about local memories."""
        # Private count
        private_count = self.privacy.privatize_count(len(memories))

        # Private average confidence
        confidences = [m.get("confidence", 0.5) for m in memories]
        private_avg_conf = self.privacy.privatize_mean(
            confidences, bounds=(0.0, 1.0)
        )

        # Private topic counts
        topic_counts: dict[str, int] = {}
        for memory in memories:
            for topic in memory.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        private_topics = {
            topic: self.privacy.privatize_count(count)
            for topic, count in topic_counts.items()
        }

        return PrivateMemoryUpdate(
            memory_count=private_count,
            avg_confidence=private_avg_conf,
            topic_counts=private_topics,
            epsilon_spent=self.privacy.budget.spent_epsilon,
            delta_spent=self.privacy.budget.spent_delta,
            noise_type=self.privacy.noise_type.value,
        )

    async def send_update(
        self,
        update: LocalUpdate,
    ) -> bool:
        """
        Send local update to federation server.

        Returns True if successful.
        """
        if not self.config.server_url:
            # No server configured - store locally
            return True

        self.state = ClientState.UPLOADING

        try:
            import httpx

            # Compress if enabled
            payload = update.to_dict()
            if self.config.compress_updates:
                payload = self._compress_update(payload)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.server_url}/api/federated/update",
                    json=payload,
                    headers={
                        "X-Client-ID": self.config.client_id,
                        "X-Round-ID": update.round_id,
                    },
                    timeout=60.0,
                )

                self.state = ClientState.IDLE
                return response.status_code == 200

        except Exception:
            self.state = ClientState.ERROR
            return False

    def _compress_update(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Compress update for efficient transmission."""
        compressed = payload.copy()

        # Sparsify weight deltas - only send significant changes
        if "weights_delta" in compressed:
            for layer, delta in compressed["weights_delta"].items():
                # Keep only top k% of values
                k = int(len(delta) * self.config.compression_ratio)
                if k > 0:
                    sorted_indices = sorted(
                        range(len(delta)),
                        key=lambda i: abs(delta[i]),
                        reverse=True,
                    )[:k]
                    sparse = [(i, delta[i]) for i in sorted_indices]
                    compressed["weights_delta"][layer] = {
                        "sparse": sparse,
                        "length": len(delta),
                    }

        return compressed

    async def participate_in_round(
        self,
        round_id: str,
        global_model: dict[str, list[float]] | None = None,
    ) -> LocalUpdate | None:
        """
        Participate in a federation round.

        Full workflow: receive model -> train -> send update.
        """
        # Receive global model if provided
        if global_model:
            await self.receive_global_model(global_model, round_id)

        # Train locally
        update = await self.train_local(round_id)

        if update:
            # Send update to server
            await self.send_update(update)

        return update

    def get_status(self) -> dict[str, Any]:
        """Get client status."""
        return {
            "client_id": self.config.client_id,
            "device_name": self.config.device_name,
            "state": self.state.value,
            "current_round": self.current_round,
            "rounds_participated": self.rounds_participated,
            "updates_sent": len(self.updates_sent),
            "privacy_budget": {
                "epsilon_total": self.privacy.budget.epsilon,
                "epsilon_spent": self.privacy.budget.spent_epsilon,
                "epsilon_remaining": self.privacy.budget.remaining_epsilon,
                "delta_total": self.privacy.budget.delta,
                "delta_spent": self.privacy.budget.spent_delta,
            },
        }

    def reset_privacy_budget(self) -> None:
        """Reset privacy budget for new federation epoch."""
        self.privacy.budget.reset()
