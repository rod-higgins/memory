"""
Federated Learning for PLM.

Enables privacy-preserving learning across multiple devices:
- Differential privacy for data protection
- Federated averaging for distributed training
- Secure aggregation for private model updates
- Model compression for efficient communication
"""

from .client import (
    ClientConfig,
    FederatedClient,
    LocalUpdate,
)
from .compression import (
    CompressionMethod,
    ModelCompressor,
)
from .privacy import (
    DifferentialPrivacy,
    NoiseType,
    PrivacyBudget,
)
from .server import (
    AggregationStrategy,
    FederatedServer,
    ServerConfig,
)

__all__ = [
    # Privacy
    "DifferentialPrivacy",
    "PrivacyBudget",
    "NoiseType",
    # Client
    "FederatedClient",
    "LocalUpdate",
    "ClientConfig",
    # Server
    "FederatedServer",
    "AggregationStrategy",
    "ServerConfig",
    # Compression
    "ModelCompressor",
    "CompressionMethod",
]
