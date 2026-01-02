# PLM: Personal Language Model

**A framework for creating personalized AI from your digital life.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/rod-higgins/memory/actions/workflows/ci.yml/badge.svg)](https://github.com/rod-higgins/memory/actions/workflows/ci.yml)

---

## Table of Contents

### Documentation

| Document | Description |
|----------|-------------|
| [VISION.md](docs/VISION.md) | Philosophy and motivation behind PLM |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical architecture and system design |
| [THEORY.md](docs/THEORY.md) | Theoretical foundations - from Markov chains to PLM |
| [ATTENTION.md](docs/ATTENTION.md) | Memory-augmented attention implementation |
| [FEDERATED.md](docs/FEDERATED.md) | Federated learning and privacy-preserving training |
| [MULTIMODAL.md](docs/MULTIMODAL.md) | Image, audio, and video processing |
| [MOBILE.md](docs/MOBILE.md) | Mobile companion app and sync architecture |
| [API.md](docs/API.md) | Complete API reference |

### Quick Links

- [What is PLM?](#what-is-plm)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture)
- [Data Sources](#universal-data-sources)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)

---

## What is PLM?

PLM (Personal Language Model) is an open-source framework that enables anyone to create their own personalized language model from their digital footprint. Unlike generic LLMs trained on internet-scale data, a PLM is trained specifically on YOUR data - your writings, communications, preferences, and history.

**The PLM doesn't replace large language models - it enhances them** by providing deep personal context that makes every interaction more relevant.

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Digital Life                         │
│   Social Media • Documents • Photos • Code • Messages        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Personal Memory System (PMS)                    │
│   Three-tier memory: Short-term → Long-term → Persistent    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             Personal Language Model (PLM)                    │
│   ~500MB • Runs Locally • Continuously Learning              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Enhanced Context for Any LLM                       │
│   Any model via context injection or direct integration      │
└─────────────────────────────────────────────────────────────┘
```

## Why PLM?

| Generic LLM | Personal Language Model |
|-------------|------------------------|
| Trained on internet | Trained on YOUR data |
| Statistically probable | Personally relevant |
| No memory | Three-tier memory hierarchy |
| Same for everyone | Unique to each user |
| ~100GB+ model | ~500MB personalized model |
| Cloud-dependent | Runs locally, fully private |
| Static after training | Continuously learning |

## Core Features

### Personal Memory System (PMS)

Three-tier storage mirroring human memory:

| Tier | Purpose | Promotion Criteria |
|------|---------|-------------------|
| **Short-term** | Session context, unvalidated | Recent interactions |
| **Long-term** | Validated facts, searchable | Corroborated 2+ times, high confidence |
| **Persistent** | Core identity, absolute truths | Stable beliefs, verified facts |

### Memory-Augmented Attention

Native integration of memory into transformer attention:

```
Attention([Q_input, Q_memory], [K_context, K_memory], [V_context, V_memory])
```

See [ATTENTION.md](docs/ATTENTION.md) for the complete implementation.

### Federated Learning

Privacy-preserving model training across devices:

- Differential privacy with configurable epsilon
- Secure aggregation protocols
- Model compression for efficient transfer

See [FEDERATED.md](docs/FEDERATED.md) for details.

### Multimodal Processing

Process images, audio, and video:

- Image analysis with object detection and OCR
- Audio transcription with speaker diarization
- Video processing with scene detection

See [MULTIMODAL.md](docs/MULTIMODAL.md) for capabilities.

### Universal Data Sources

Adapters for 50+ data sources:

| Category | Sources |
|----------|---------|
| **Social Media** | Twitter, Facebook, Instagram, LinkedIn, Reddit |
| **Communications** | Gmail, iMessage, WhatsApp, Slack, Discord |
| **Documents** | Google Docs, Notion, Obsidian, Apple Notes |
| **Media** | Photos (with OCR), YouTube, Spotify |
| **Code** | GitHub, GitLab, local git repos |
| **AI History** | Various LLM conversation logs |

## Installation

```bash
# Clone the repository
git clone https://github.com/rod-higgins/memory.git
cd memory

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Optional: Install with all features
pip install -e ".[all]"

# Optional: Development dependencies
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For local embeddings
pip install -e ".[embeddings]"

# For PLM training (requires GPU)
pip install -e ".[tuning]"

# For multimodal processing
pip install -e ".[multimodal]"

# For web UI
pip install -e ".[web]"
```

## Quick Start

### 1. Initialize the System

```bash
python -m memory.cli init
```

### 2. Ingest Your Data

```bash
# Ingest AI conversation history
python -m memory.cli ingest claude-history

# Ingest local git repositories
python -m memory.cli ingest git-repos --path ~/code

# Ingest documents
python -m memory.cli ingest documents --path ~/Documents
```

### 3. Search Your Memories

```bash
python -m memory.cli search "programming preferences"
```

### 4. Use in Python

```python
from memory import MemoryAPI
import asyncio

async def main():
    api = MemoryAPI()
    await api.initialize()

    # Store a memory
    await api.remember("I prefer Python for data processing")

    # Get personalized context
    context = await api.get_context("How should I structure this project?")
    print(context)

    # Augment a prompt for any LLM
    augmented = await api.augment_prompt("Help me with this code")
    print(augmented.memory_context)

asyncio.run(main())
```

### 5. Train Your PLM (Optional, requires GPU)

```python
from memory.slm import PersonalSLMTrainer

async def train():
    trainer = PersonalSLMTrainer()

    # Prepare training data from memories
    await trainer.prepare_data()

    # Train with LoRA
    trainer.train()

    # Merge and save
    trainer.merge_and_save()

    # Export to Ollama
    trainer.export_to_ollama("my-plm")

# Run: ollama run my-plm
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│                                                                         │
│   Universal adapters for any data source                                │
│   Social • Documents • Media • Code • Communications • AI History       │
│                              │                                          │
│                              ▼                                          │
│                    ┌─────────────────┐                                  │
│                    │   DataPoint     │  Unified format for all data     │
│                    └─────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERSONAL MEMORY SYSTEM (PMS)                         │
│                                                                         │
│   SHORT-TERM          LONG-TERM            PERSISTENT                   │
│   (In-Memory)         (LanceDB)            (SQLite)                     │
│   • Session cache     • Vector search      • Core identity              │
│   • Unvalidated       • Validated facts    • Absolute truths            │
│                                                                         │
│   Automatic promotion based on confidence, corroboration, time          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERSONAL LANGUAGE MODEL (PLM)                        │
│                                                                         │
│   Memory-Augmented Attention:                                           │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │ Attention([Q_input, Q_memory], [K_context, K_memory],          │    │
│   │           [V_context, V_memory])                               │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   Training Pipeline:                                                    │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│   │ Knowledge  │→│   LoRA     │→│  Pruning   │→│ Compression│          │
│   │Distillation│ │Fine-tuning │ │(user-spec) │ │  (ZipLLM)  │          │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INTEGRATION LAYER                                  │
│                                                                         │
│   Federated Learning    │   Mobile Sync    │   Multimodal Processing   │
│   • Privacy-preserving  │   • Offline-first│   • Image/Audio/Video    │
│   • Secure aggregation  │   • Conflict res │   • Scene detection      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
plm/
├── src/memory/
│   ├── schema/          # Data models (MemoryEntry, Identity, etc.)
│   ├── storage/         # Three-tier storage backends
│   ├── ingestion/       # Data source adapters (50+)
│   ├── query/           # Semantic and keyword search
│   ├── export/          # LLM format exporters
│   ├── llm/             # LLM integration and augmentation
│   ├── slm/             # PLM training, inference, continuous learning
│   ├── attention/       # Memory-augmented attention layers
│   ├── federated/       # Federated learning components
│   ├── mobile/          # Mobile companion app backend
│   ├── multimodal/      # Image, audio, video processing
│   └── api/             # High-level unified API
├── docs/
│   ├── VISION.md        # Philosophy and motivation
│   ├── ARCHITECTURE.md  # Technical implementation details
│   ├── THEORY.md        # From Markov chains to PLM
│   ├── ATTENTION.md     # Memory-augmented attention
│   ├── FEDERATED.md     # Federated learning
│   ├── MULTIMODAL.md    # Multimodal processing
│   ├── MOBILE.md        # Mobile companion app
│   └── API.md           # API reference
├── tests/               # Test suite
├── config/
│   └── default.toml     # Default configuration
└── scripts/             # Utility scripts
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Short-term storage | In-memory dict | Fast session cache |
| Long-term storage | LanceDB | <10ms vector search |
| Persistent storage | SQLite + FTS5 | Human-inspectable, portable |
| Embeddings | sentence-transformers | Local, private embeddings |
| Base model | Llama 3.2 3B | Foundation for PLM |
| Fine-tuning | LoRA/QLoRA (PEFT) | Efficient training |
| Inference | Ollama | Local model serving |
| Compression | bitsandbytes | 4-bit quantization |

## Privacy

**All processing is local.** Your data never leaves your machine.

- Memories are stored locally in `data/`
- Trained models are stored locally in `models/`
- No API calls for embeddings (sentence-transformers runs locally)
- No telemetry or data collection
- Federated learning uses differential privacy

The `.gitignore` is configured to exclude all personal data, making it safe to fork and contribute.

## Contributing

This is a universal framework - contributions are welcome!

### Areas for Contribution

- **Data Source Adapters** - Add support for new data sources
- **Training Improvements** - Better fine-tuning techniques
- **Compression** - More efficient model compression
- **Export Formats** - Support for more LLM formats
- **Documentation** - Improve guides and examples

### Development Setup

```bash
# Clone and setup
git clone https://github.com/rod-higgins/memory.git
cd memory
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
mypy src/
```

## Roadmap

- [x] Three-tier memory system
- [x] Memory-augmented attention
- [x] Federated learning
- [x] Multimodal processing
- [x] Mobile companion app
- [x] CI/CD pipeline
- [x] Web UI for memory management
- [x] 50+ data source adapters
- [x] Continuous learning with corrections
- [ ] Model compression (ZipLLM)
- [ ] Cross-device sync
- [ ] Plugin ecosystem

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by Google's Titan memory architecture
- Built on the transformer revolution
- ZipLLM compression techniques
- The open-source AI community

---

**PLM: Your AI, trained on your life, running on your machine.**

*Create your Personal Language Model today.*
