# Federated Learning

## Overview

The federated learning module enables privacy-preserving model training across multiple devices or users. Each device trains on local data, and only model updates (not raw data) are shared and aggregated.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       FEDERATED LEARNING SYSTEM                          │
│                                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│   │ Device 1 │  │ Device 2 │  │ Device 3 │  │ Device N │               │
│   │          │  │          │  │          │  │          │               │
│   │ Local    │  │ Local    │  │ Local    │  │ Local    │               │
│   │ Training │  │ Training │  │ Training │  │ Training │               │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│        │             │             │             │                      │
│        │    Compressed + Encrypted Updates       │                      │
│        └─────────────┴─────────────┴─────────────┘                      │
│                              │                                          │
│                              ▼                                          │
│                    ┌─────────────────┐                                  │
│                    │   Aggregation   │                                  │
│                    │     Server      │                                  │
│                    └────────┬────────┘                                  │
│                              │                                          │
│                    ┌─────────▼─────────┐                                │
│                    │  Secure Aggregate │                                │
│                    │   (FedAvg, etc.)  │                                │
│                    └─────────┬─────────┘                                │
│                              │                                          │
│                    ┌─────────▼─────────┐                                │
│                    │  Global Model     │                                │
│                    │    Update         │                                │
│                    └───────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Differential Privacy

Adds calibrated noise to protect individual data points.

```python
from memory.federated import DifferentialPrivacy, PrivacyConfig

config = PrivacyConfig(
    epsilon=1.0,              # Privacy budget (lower = more private)
    delta=1e-5,               # Probability of privacy breach
    max_grad_norm=1.0,        # Gradient clipping threshold
    noise_multiplier=None,    # Auto-computed from epsilon/delta
)

dp = DifferentialPrivacy(config)

# Apply to gradients
private_gradients = dp.apply(gradients)

# Check privacy budget
remaining = dp.get_privacy_spent()
print(f"Privacy spent: ε={remaining.epsilon:.2f}, δ={remaining.delta:.2e}")
```

### 2. Federated Client

Local training component that runs on each device.

```python
from memory.federated import FederatedClient, ClientConfig

config = ClientConfig(
    client_id="device-123",
    local_epochs=3,
    batch_size=32,
    learning_rate=1e-4,
    privacy_config=privacy_config,
    compression_config=compression_config,
)

client = FederatedClient(
    config=config,
    model=model,
    data_loader=local_data,
)

# Train on local data
local_update = await client.train_round()

# Update includes:
# - Compressed gradient updates
# - Privacy-preserved weights
# - Training metadata
```

### 3. Federated Server

Coordinates training across all clients.

```python
from memory.federated import FederatedServer, ServerConfig, AggregationStrategy

config = ServerConfig(
    min_clients=3,
    max_clients=100,
    rounds=10,
    aggregation=AggregationStrategy.FEDERATED_AVERAGING,
    selection_strategy="random",  # or "importance", "contribution"
)

server = FederatedServer(
    config=config,
    global_model=model,
)

# Run training round
await server.run_round()

# Get aggregated model
updated_model = server.get_global_model()
```

### 4. Model Compression

Reduces communication overhead for model updates.

```python
from memory.federated import ModelCompressor, CompressionConfig

config = CompressionConfig(
    method="top_k",           # "top_k", "random_k", "quantization"
    compression_ratio=0.1,    # Keep top 10% of gradients
    quantization_bits=8,      # For quantization method
)

compressor = ModelCompressor(config)

# Compress before sending
compressed = compressor.compress(gradients)
print(f"Compression ratio: {compressor.get_ratio():.1%}")

# Decompress on server
decompressed = compressor.decompress(compressed)
```

## Aggregation Strategies

### Federated Averaging (FedAvg)

Simple weighted average of client updates:

```
w_global = Σ (n_k / n) * w_k
```

Where `n_k` is the number of samples on client `k`.

### Federated SGD

Aggregate gradients instead of weights:

```
g_global = Σ (n_k / n) * g_k
w_global = w_global - η * g_global
```

### Secure Aggregation

Cryptographic protocol ensuring server cannot see individual updates:

```python
from memory.federated import SecureAggregator

aggregator = SecureAggregator(
    num_clients=10,
    threshold=6,  # Minimum clients needed
)

# Each client encrypts their update
encrypted_updates = [aggregator.encrypt(u, client_id) for u, client_id in updates]

# Server aggregates without seeing individual updates
aggregated = aggregator.secure_aggregate(encrypted_updates)
```

## Privacy Guarantees

### Differential Privacy Levels

| Epsilon (ε) | Privacy Level | Use Case |
|-------------|---------------|----------|
| 0.1 - 1.0 | Strong | Sensitive personal data |
| 1.0 - 5.0 | Moderate | General personal preferences |
| 5.0 - 10.0 | Weak | Non-sensitive patterns |
| > 10.0 | Minimal | Public information |

### Privacy Accounting

Track cumulative privacy loss across training rounds:

```python
from memory.federated import PrivacyAccountant

accountant = PrivacyAccountant(
    target_epsilon=5.0,
    target_delta=1e-5,
)

for round in range(num_rounds):
    # Train round
    ...

    # Update accounting
    accountant.step(noise_multiplier, sample_rate)

    if accountant.is_budget_exhausted():
        print("Privacy budget exhausted!")
        break

print(f"Total privacy: ε={accountant.epsilon:.2f}")
```

## Complete Example

```python
import asyncio
from memory.federated import (
    FederatedServer,
    FederatedClient,
    ServerConfig,
    ClientConfig,
    PrivacyConfig,
    CompressionConfig,
    AggregationStrategy,
)

async def run_federated_training():
    # Privacy configuration
    privacy = PrivacyConfig(
        epsilon=2.0,
        delta=1e-5,
        max_grad_norm=1.0,
    )

    # Compression configuration
    compression = CompressionConfig(
        method="top_k",
        compression_ratio=0.1,
    )

    # Server configuration
    server_config = ServerConfig(
        min_clients=3,
        max_clients=10,
        rounds=5,
        aggregation=AggregationStrategy.FEDERATED_AVERAGING,
    )

    # Initialize server
    server = FederatedServer(
        config=server_config,
        global_model=base_model,
    )

    # Create clients
    clients = []
    for i in range(5):
        client_config = ClientConfig(
            client_id=f"device-{i}",
            local_epochs=2,
            batch_size=16,
            privacy_config=privacy,
            compression_config=compression,
        )

        client = FederatedClient(
            config=client_config,
            model=base_model.copy(),
            data_loader=get_local_data(i),
        )
        clients.append(client)

    # Run federated training
    for round in range(server_config.rounds):
        print(f"Round {round + 1}/{server_config.rounds}")

        # Select clients for this round
        selected = server.select_clients(clients)

        # Train locally on each client
        updates = []
        for client in selected:
            update = await client.train_round()
            updates.append(update)

        # Aggregate on server
        await server.aggregate(updates)

        # Broadcast updated model
        for client in clients:
            client.receive_global_model(server.get_global_model())

        print(f"  Participants: {len(selected)}")
        print(f"  Privacy spent: ε={server.privacy_accountant.epsilon:.2f}")

    # Get final model
    final_model = server.get_global_model()
    return final_model

# Run
model = asyncio.run(run_federated_training())
```

## Configuration Reference

### PrivacyConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon` | float | 1.0 | Privacy budget |
| `delta` | float | 1e-5 | Privacy breach probability |
| `max_grad_norm` | float | 1.0 | Gradient clipping threshold |
| `noise_multiplier` | float | None | Noise scale (auto-computed) |
| `mechanism` | str | "gaussian" | Noise mechanism |

### ClientConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client_id` | str | required | Unique client identifier |
| `local_epochs` | int | 3 | Local training epochs |
| `batch_size` | int | 32 | Local batch size |
| `learning_rate` | float | 1e-4 | Local learning rate |
| `privacy_config` | PrivacyConfig | None | Privacy settings |
| `compression_config` | CompressionConfig | None | Compression settings |

### ServerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_clients` | int | 3 | Minimum clients per round |
| `max_clients` | int | 100 | Maximum clients per round |
| `rounds` | int | 10 | Number of training rounds |
| `aggregation` | AggregationStrategy | FEDERATED_AVERAGING | Aggregation method |
| `selection_strategy` | str | "random" | Client selection method |

### CompressionConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | "top_k" | Compression method |
| `compression_ratio` | float | 0.1 | Ratio of values to keep |
| `quantization_bits` | int | 8 | Bits for quantization |

## Security Considerations

1. **Transport Security**: All communication should use TLS
2. **Client Authentication**: Verify client identity before accepting updates
3. **Byzantine Tolerance**: Robust aggregation handles malicious clients
4. **Secure Aggregation**: Individual updates never visible to server

---

*See [ARCHITECTURE.md](./ARCHITECTURE.md) for overall system architecture.*
*See [ATTENTION.md](./ATTENTION.md) for memory-augmented attention.*
