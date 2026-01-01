"""
Federated Learning API routes for PLM Web UI.

Provides endpoints for:
- Server status and management
- Client registration
- Round management
- Update submission
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError:
    # Stubs for when FastAPI not installed
    class APIRouter:
        def get(self, *args, **kwargs):
            def decorator(f):
                return f

            return decorator

        def post(self, *args, **kwargs):
            def decorator(f):
                return f

            return decorator

        def delete(self, *args, **kwargs):
            def decorator(f):
                return f

            return decorator

    class BaseModel:
        pass

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail


from ..federated import (
    AggregationStrategy,
    ClientConfig,
    FederatedClient,
    FederatedServer,
    ServerConfig,
)
from ..federated.client import LocalUpdate

router = APIRouter(prefix="/api/federated", tags=["federated"])

# Global server instance
_server: FederatedServer | None = None
_client: FederatedClient | None = None


def get_server() -> FederatedServer:
    """Get or create federation server."""
    global _server
    if _server is None:
        _server = FederatedServer()
    return _server


def get_client() -> FederatedClient:
    """Get or create federation client."""
    global _client
    if _client is None:
        _client = FederatedClient()
    return _client


# Request/Response models
class ServerConfigRequest(BaseModel):
    round_duration_seconds: int = 3600
    min_clients_per_round: int = 2
    max_clients_per_round: int = 100
    aggregation_strategy: str = "fedavg"


class ClientConfigRequest(BaseModel):
    device_name: str = ""
    epsilon: float = 1.0
    delta: float = 1e-5
    local_epochs: int = 3
    server_url: str = ""


class ClientRegistration(BaseModel):
    client_id: str
    device_name: str = ""


class UpdateSubmission(BaseModel):
    client_id: str
    round_id: str
    weights_delta: dict[str, list[float]] = {}
    samples_used: int = 0
    training_loss: float = 0.0


# Server endpoints
@router.get("/server/status")
async def get_server_status() -> dict[str, Any]:
    """Get federation server status."""
    server = get_server()
    return server.get_status()


@router.post("/server/configure")
async def configure_server(config: ServerConfigRequest) -> dict[str, Any]:
    """Configure federation server."""
    global _server

    strategy = AggregationStrategy.FEDAVG
    try:
        strategy = AggregationStrategy(config.aggregation_strategy)
    except ValueError:
        pass

    server_config = ServerConfig(
        round_duration_seconds=config.round_duration_seconds,
        min_clients_per_round=config.min_clients_per_round,
        max_clients_per_round=config.max_clients_per_round,
        aggregation_strategy=strategy,
    )
    _server = FederatedServer(config=server_config)

    return {"status": "configured", "config": server_config.__dict__}


@router.post("/server/round/start")
async def start_round() -> dict[str, Any]:
    """Start a new federation round."""
    server = get_server()
    round_info = await server.start_round()

    if round_info is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot start round - not enough clients or round in progress",
        )

    return round_info.to_dict()


@router.post("/server/round/complete")
async def complete_round() -> dict[str, Any]:
    """Complete the current federation round."""
    server = get_server()
    completed = await server.complete_round()

    if completed is None:
        raise HTTPException(
            status_code=400,
            detail="No active round to complete",
        )

    return completed.to_dict()


@router.get("/server/clients")
async def get_clients() -> list[dict[str, Any]]:
    """Get registered clients."""
    server = get_server()
    return [
        {
            "client_id": c.client_id,
            "device_name": c.device_name,
            "registered_at": c.registered_at.isoformat(),
            "last_seen": c.last_seen.isoformat(),
            "rounds_participated": c.rounds_participated,
            "reliability_score": c.reliability_score,
        }
        for c in server.clients.values()
    ]


@router.post("/server/clients/register")
async def register_client(registration: ClientRegistration) -> dict[str, Any]:
    """Register a client with the server."""
    server = get_server()
    success = server.register_client(
        registration.client_id,
        registration.device_name,
    )

    return {
        "success": success,
        "client_id": registration.client_id,
    }


@router.delete("/server/clients/{client_id}")
async def unregister_client(client_id: str) -> dict[str, Any]:
    """Unregister a client from the server."""
    server = get_server()
    success = server.unregister_client(client_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Client {client_id} not found",
        )

    return {"success": True, "client_id": client_id}


@router.post("/server/update")
async def submit_update(submission: UpdateSubmission) -> dict[str, Any]:
    """Submit a local update to the server."""
    server = get_server()

    update = LocalUpdate(
        client_id=submission.client_id,
        round_id=submission.round_id,
        weights_delta=submission.weights_delta,
        samples_used=submission.samples_used,
        training_loss=submission.training_loss,
    )
    update.checksum = update.compute_checksum()

    success = await server.receive_update(update)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Update rejected - invalid round, client, or duplicate",
        )

    return {"success": True, "update_id": update.id}


@router.get("/server/model")
async def get_global_model() -> dict[str, Any]:
    """Get the current global model."""
    server = get_server()
    model = server.get_global_model()

    return {
        "model_name": server.config.model_name,
        "model_version": server.config.model_version,
        "layers": list(model.keys()),
        "total_parameters": sum(len(v) for v in model.values()),
    }


@router.get("/server/rounds")
async def get_round_history() -> list[dict[str, Any]]:
    """Get history of completed rounds."""
    server = get_server()
    return [r.to_dict() for r in server.round_history[-20:]]


# Client endpoints
@router.get("/client/status")
async def get_client_status() -> dict[str, Any]:
    """Get federation client status."""
    client = get_client()
    return client.get_status()


@router.post("/client/configure")
async def configure_client(config: ClientConfigRequest) -> dict[str, Any]:
    """Configure federation client."""
    global _client

    client_config = ClientConfig(
        device_name=config.device_name,
        epsilon=config.epsilon,
        delta=config.delta,
        local_epochs=config.local_epochs,
        server_url=config.server_url,
    )
    _client = FederatedClient(config=client_config)

    return {"status": "configured", "client_id": client_config.client_id}


@router.post("/client/train")
async def train_local() -> dict[str, Any]:
    """Perform local training and generate update."""
    client = get_client()
    update = await client.train_local()

    if update is None:
        raise HTTPException(
            status_code=400,
            detail="Training failed or client busy",
        )

    return update.to_dict()


@router.post("/client/participate/{round_id}")
async def participate_in_round(round_id: str) -> dict[str, Any]:
    """Participate in a federation round."""
    client = get_client()
    update = await client.participate_in_round(round_id)

    if update is None:
        raise HTTPException(
            status_code=400,
            detail="Participation failed",
        )

    return update.to_dict()


@router.get("/client/privacy")
async def get_privacy_status() -> dict[str, Any]:
    """Get privacy budget status."""
    client = get_client()
    budget = client.privacy.budget

    return {
        "epsilon_total": budget.epsilon,
        "epsilon_spent": budget.spent_epsilon,
        "epsilon_remaining": budget.remaining_epsilon,
        "delta_total": budget.delta,
        "delta_spent": budget.spent_delta,
        "is_exhausted": budget.is_exhausted,
        "operations": len(budget.costs),
    }


@router.post("/client/privacy/reset")
async def reset_privacy_budget() -> dict[str, Any]:
    """Reset privacy budget."""
    client = get_client()
    client.reset_privacy_budget()

    return {"status": "reset", "epsilon_remaining": client.privacy.budget.epsilon}
