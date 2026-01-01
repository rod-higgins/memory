"""
Federated Learning Server for PLM.

Coordinates federated learning across multiple clients:
- Round management
- Secure aggregation of updates
- Model distribution
- Client selection
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from .client import LocalUpdate
from .privacy import PrivateMemoryUpdate


class AggregationStrategy(Enum):
    """Strategy for aggregating client updates."""

    FEDAVG = "fedavg"  # Federated Averaging
    FEDPROX = "fedprox"  # FedProx with proximal term
    SCAFFOLD = "scaffold"  # SCAFFOLD variance reduction
    MEDIAN = "median"  # Coordinate-wise median (Byzantine robust)
    TRIMMED_MEAN = "trimmed_mean"  # Trimmed mean (Byzantine robust)


class RoundState(Enum):
    """State of a federation round."""

    PENDING = "pending"
    ACTIVE = "active"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ServerConfig:
    """Configuration for federated learning server."""

    # Server identity
    server_id: str = field(default_factory=lambda: str(uuid4()))

    # Round settings
    round_duration_seconds: int = 3600  # 1 hour
    min_clients_per_round: int = 2
    max_clients_per_round: int = 100
    client_selection_fraction: float = 1.0  # Select all available

    # Aggregation
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    fedprox_mu: float = 0.01  # Proximal term coefficient

    # Byzantine robustness
    trim_ratio: float = 0.1  # For trimmed mean

    # Model settings
    model_name: str = "plm-embedding"
    model_version: str = "1.0"


@dataclass
class FederationRound:
    """A single round of federated learning."""

    id: str = field(default_factory=lambda: str(uuid4()))
    round_number: int = 0
    state: RoundState = RoundState.PENDING

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    deadline: datetime | None = None

    # Participants
    selected_clients: list[str] = field(default_factory=list)
    received_updates: list[LocalUpdate] = field(default_factory=list)

    # Results
    aggregated_model: dict[str, list[float]] = field(default_factory=dict)
    aggregated_statistics: PrivateMemoryUpdate | None = None
    average_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize round information."""
        return {
            "id": self.id,
            "round_number": self.round_number,
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "selected_clients": len(self.selected_clients),
            "received_updates": len(self.received_updates),
            "average_loss": self.average_loss,
        }


@dataclass
class ClientInfo:
    """Information about a registered client."""

    client_id: str
    device_name: str = ""
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    rounds_participated: int = 0
    total_samples: int = 0
    reliability_score: float = 1.0  # Based on successful participation


class FederatedServer:
    """
    Federated learning server for PLM.

    Coordinates training across multiple devices:
    - Client registration and selection
    - Round management
    - Secure aggregation
    - Model versioning
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        initial_model: dict[str, list[float]] | None = None,
    ):
        self.config = config or ServerConfig()
        self.global_model = initial_model or {}

        # Client management
        self.clients: dict[str, ClientInfo] = {}

        # Round management
        self.current_round: FederationRound | None = None
        self.round_history: list[FederationRound] = []
        self.round_counter: int = 0

        # Statistics
        self.total_rounds_completed: int = 0
        self.total_updates_received: int = 0

    def register_client(
        self,
        client_id: str,
        device_name: str = "",
    ) -> bool:
        """Register a new client for federation."""
        if client_id in self.clients:
            # Update existing
            self.clients[client_id].last_seen = datetime.now()
            return True

        self.clients[client_id] = ClientInfo(
            client_id=client_id,
            device_name=device_name,
        )
        return True

    def unregister_client(self, client_id: str) -> bool:
        """Remove a client from federation."""
        if client_id in self.clients:
            del self.clients[client_id]
            return True
        return False

    def get_active_clients(
        self,
        since_minutes: int = 60,
    ) -> list[ClientInfo]:
        """Get clients seen recently."""
        cutoff = datetime.now() - timedelta(minutes=since_minutes)
        return [
            c for c in self.clients.values()
            if c.last_seen >= cutoff
        ]

    async def start_round(self) -> FederationRound | None:
        """Start a new federation round."""
        if self.current_round and self.current_round.state == RoundState.ACTIVE:
            return None  # Round already in progress

        # Select clients
        active_clients = self.get_active_clients()
        if len(active_clients) < self.config.min_clients_per_round:
            return None  # Not enough clients

        # Select subset of clients
        selected = self._select_clients(active_clients)

        # Create new round
        self.round_counter += 1
        self.current_round = FederationRound(
            round_number=self.round_counter,
            state=RoundState.ACTIVE,
            started_at=datetime.now(),
            deadline=datetime.now() + timedelta(
                seconds=self.config.round_duration_seconds
            ),
            selected_clients=[c.client_id for c in selected],
        )

        return self.current_round

    def _select_clients(
        self,
        available: list[ClientInfo],
    ) -> list[ClientInfo]:
        """Select clients for a round."""
        import random

        # Sort by reliability score
        sorted_clients = sorted(
            available,
            key=lambda c: c.reliability_score,
            reverse=True,
        )

        # Select top fraction
        n_select = int(len(sorted_clients) * self.config.client_selection_fraction)
        n_select = min(n_select, self.config.max_clients_per_round)
        n_select = max(n_select, self.config.min_clients_per_round)

        # Add some randomness
        selected = sorted_clients[:n_select]
        if len(sorted_clients) > n_select:
            # Replace 20% with random selection for exploration
            n_random = max(1, int(n_select * 0.2))
            remaining = sorted_clients[n_select:]
            if remaining:
                random_picks = random.sample(
                    remaining,
                    min(n_random, len(remaining)),
                )
                selected = selected[:-n_random] + random_picks

        return selected

    async def receive_update(
        self,
        update: LocalUpdate,
    ) -> bool:
        """
        Receive an update from a client.

        Returns True if update accepted.
        """
        if not self.current_round:
            return False

        if self.current_round.state != RoundState.ACTIVE:
            return False

        if update.client_id not in self.current_round.selected_clients:
            return False

        # Verify checksum
        if update.checksum != update.compute_checksum():
            return False

        # Check for duplicate
        existing_ids = [u.client_id for u in self.current_round.received_updates]
        if update.client_id in existing_ids:
            return False

        # Accept update
        self.current_round.received_updates.append(update)
        self.total_updates_received += 1

        # Update client info
        if update.client_id in self.clients:
            self.clients[update.client_id].last_seen = datetime.now()
            self.clients[update.client_id].total_samples += update.samples_used

        # Check if round can be completed
        if len(self.current_round.received_updates) >= self.config.min_clients_per_round:
            # Could auto-complete, but we'll let it run to deadline
            pass

        return True

    async def complete_round(self) -> FederationRound | None:
        """Complete the current round and aggregate updates."""
        if not self.current_round:
            return None

        if not self.current_round.received_updates:
            self.current_round.state = RoundState.FAILED
            return self.current_round

        self.current_round.state = RoundState.AGGREGATING

        # Aggregate updates
        aggregated = await self._aggregate_updates(
            self.current_round.received_updates
        )

        self.current_round.aggregated_model = aggregated["model"]
        self.current_round.aggregated_statistics = aggregated["statistics"]
        self.current_round.average_loss = aggregated["average_loss"]

        # Update global model
        self.global_model = self._apply_update(
            self.global_model,
            self.current_round.aggregated_model,
        )

        # Update client reliability scores
        for update in self.current_round.received_updates:
            if update.client_id in self.clients:
                client = self.clients[update.client_id]
                client.rounds_participated += 1
                # Increase reliability for participation
                client.reliability_score = min(
                    1.0,
                    client.reliability_score + 0.05,
                )

        # Decrease reliability for non-participants
        for client_id in self.current_round.selected_clients:
            if client_id not in [u.client_id for u in self.current_round.received_updates]:
                if client_id in self.clients:
                    self.clients[client_id].reliability_score = max(
                        0.1,
                        self.clients[client_id].reliability_score - 0.1,
                    )

        # Finalize round
        self.current_round.state = RoundState.COMPLETED
        self.current_round.completed_at = datetime.now()
        self.round_history.append(self.current_round)
        self.total_rounds_completed += 1

        completed = self.current_round
        self.current_round = None

        return completed

    async def _aggregate_updates(
        self,
        updates: list[LocalUpdate],
    ) -> dict[str, Any]:
        """Aggregate client updates based on strategy."""
        if self.config.aggregation_strategy == AggregationStrategy.FEDAVG:
            return await self._fedavg_aggregate(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.FEDPROX:
            return await self._fedprox_aggregate(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.MEDIAN:
            return await self._median_aggregate(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.TRIMMED_MEAN:
            return await self._trimmed_mean_aggregate(updates)
        else:
            return await self._fedavg_aggregate(updates)

    async def _fedavg_aggregate(
        self,
        updates: list[LocalUpdate],
    ) -> dict[str, Any]:
        """Federated averaging aggregation."""
        if not updates:
            return {"model": {}, "statistics": None, "average_loss": 0.0}

        # Weight by number of samples
        total_samples = sum(u.samples_used for u in updates)
        if total_samples == 0:
            total_samples = len(updates)  # Equal weighting

        # Aggregate weight deltas
        aggregated_model: dict[str, list[float]] = {}

        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update.weights_delta.keys())

        for layer in layer_names:
            weighted_sum: list[float] | None = None

            for update in updates:
                if layer not in update.weights_delta:
                    continue

                delta = update.weights_delta[layer]
                weight = update.samples_used / total_samples if total_samples > 0 else 1 / len(updates)

                if weighted_sum is None:
                    weighted_sum = [d * weight for d in delta]
                else:
                    for i, d in enumerate(delta):
                        if i < len(weighted_sum):
                            weighted_sum[i] += d * weight

            if weighted_sum:
                aggregated_model[layer] = weighted_sum

        # Aggregate statistics
        aggregated_stats = await self._aggregate_statistics(updates)

        # Average loss
        avg_loss = sum(u.training_loss for u in updates) / len(updates)

        return {
            "model": aggregated_model,
            "statistics": aggregated_stats,
            "average_loss": avg_loss,
        }

    async def _fedprox_aggregate(
        self,
        updates: list[LocalUpdate],
    ) -> dict[str, Any]:
        """FedProx aggregation with proximal term."""
        # Same as FedAvg but clients use proximal term during training
        return await self._fedavg_aggregate(updates)

    async def _median_aggregate(
        self,
        updates: list[LocalUpdate],
    ) -> dict[str, Any]:
        """Coordinate-wise median for Byzantine robustness."""
        if not updates:
            return {"model": {}, "statistics": None, "average_loss": 0.0}

        aggregated_model: dict[str, list[float]] = {}

        # Get all layer names
        layer_names = set()
        for update in updates:
            layer_names.update(update.weights_delta.keys())

        for layer in layer_names:
            # Collect all values for this layer
            all_values: list[list[float]] = []
            for update in updates:
                if layer in update.weights_delta:
                    all_values.append(update.weights_delta[layer])

            if not all_values:
                continue

            # Coordinate-wise median
            median_values = []
            for i in range(len(all_values[0])):
                values_at_i = sorted(v[i] for v in all_values if i < len(v))
                if values_at_i:
                    mid = len(values_at_i) // 2
                    if len(values_at_i) % 2 == 0:
                        median_values.append(
                            (values_at_i[mid - 1] + values_at_i[mid]) / 2
                        )
                    else:
                        median_values.append(values_at_i[mid])

            aggregated_model[layer] = median_values

        aggregated_stats = await self._aggregate_statistics(updates)
        avg_loss = sum(u.training_loss for u in updates) / len(updates)

        return {
            "model": aggregated_model,
            "statistics": aggregated_stats,
            "average_loss": avg_loss,
        }

    async def _trimmed_mean_aggregate(
        self,
        updates: list[LocalUpdate],
    ) -> dict[str, Any]:
        """Trimmed mean for Byzantine robustness."""
        if not updates:
            return {"model": {}, "statistics": None, "average_loss": 0.0}

        trim_count = int(len(updates) * self.config.trim_ratio)
        aggregated_model: dict[str, list[float]] = {}

        layer_names = set()
        for update in updates:
            layer_names.update(update.weights_delta.keys())

        for layer in layer_names:
            all_values: list[list[float]] = []
            for update in updates:
                if layer in update.weights_delta:
                    all_values.append(update.weights_delta[layer])

            if not all_values:
                continue

            # Coordinate-wise trimmed mean
            trimmed_values = []
            for i in range(len(all_values[0])):
                values_at_i = sorted(v[i] for v in all_values if i < len(v))
                if len(values_at_i) > 2 * trim_count:
                    # Trim extremes
                    values_at_i = values_at_i[trim_count:-trim_count] if trim_count > 0 else values_at_i
                if values_at_i:
                    trimmed_values.append(sum(values_at_i) / len(values_at_i))

            aggregated_model[layer] = trimmed_values

        aggregated_stats = await self._aggregate_statistics(updates)
        avg_loss = sum(u.training_loss for u in updates) / len(updates)

        return {
            "model": aggregated_model,
            "statistics": aggregated_stats,
            "average_loss": avg_loss,
        }

    async def _aggregate_statistics(
        self,
        updates: list[LocalUpdate],
    ) -> PrivateMemoryUpdate | None:
        """Aggregate private statistics from updates."""
        stats_updates = [
            u.memory_statistics for u in updates
            if u.memory_statistics
        ]

        if not stats_updates:
            return None

        # Sum counts (already noised)
        total_count = sum(s.memory_count for s in stats_updates)

        # Average confidence (already private means)
        avg_confidence = sum(s.avg_confidence for s in stats_updates) / len(stats_updates)

        # Merge topic counts
        merged_topics: dict[str, float] = {}
        for stats in stats_updates:
            for topic, count in stats.topic_counts.items():
                merged_topics[topic] = merged_topics.get(topic, 0) + count

        return PrivateMemoryUpdate(
            memory_count=total_count,
            avg_confidence=avg_confidence,
            topic_counts=merged_topics,
            epsilon_spent=sum(s.epsilon_spent for s in stats_updates),
            delta_spent=sum(s.delta_spent for s in stats_updates),
        )

    def _apply_update(
        self,
        current_model: dict[str, list[float]],
        update: dict[str, list[float]],
        learning_rate: float = 1.0,
    ) -> dict[str, list[float]]:
        """Apply aggregated update to model."""
        new_model = {}

        all_layers = set(current_model.keys()) | set(update.keys())

        for layer in all_layers:
            if layer in update and layer in current_model:
                # Apply delta to existing weights
                current = current_model[layer]
                delta = update[layer]
                new_model[layer] = [
                    c + learning_rate * d
                    for c, d in zip(current, delta)
                ]
            elif layer in current_model:
                new_model[layer] = current_model[layer]
            else:
                new_model[layer] = update[layer]

        return new_model

    def get_global_model(self) -> dict[str, list[float]]:
        """Get current global model."""
        return self.global_model

    def get_status(self) -> dict[str, Any]:
        """Get server status."""
        return {
            "server_id": self.config.server_id,
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "registered_clients": len(self.clients),
            "active_clients": len(self.get_active_clients()),
            "total_rounds_completed": self.total_rounds_completed,
            "total_updates_received": self.total_updates_received,
            "current_round": (
                self.current_round.to_dict()
                if self.current_round else None
            ),
            "aggregation_strategy": self.config.aggregation_strategy.value,
        }

    async def run_continuous(
        self,
        check_interval_seconds: int = 60,
        on_round_complete: Callable[[FederationRound], None] | None = None,
    ) -> None:
        """
        Run continuous federation rounds.

        Checks periodically and starts rounds when clients available.
        """
        while True:
            try:
                # Start round if none active
                if not self.current_round:
                    round_info = await self.start_round()
                    if round_info:
                        print(f"Started round {round_info.round_number}")

                # Check if round should complete
                if self.current_round:
                    now = datetime.now()
                    if self.current_round.deadline and now >= self.current_round.deadline:
                        completed = await self.complete_round()
                        if completed and on_round_complete:
                            on_round_complete(completed)

            except Exception as e:
                print(f"Federation error: {e}")

            await asyncio.sleep(check_interval_seconds)
