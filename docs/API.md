# API Reference

## Overview

This document provides a complete reference for the PLM Python API.

## Core API

### MemoryAPI

The main entry point for interacting with the memory system.

```python
from memory import MemoryAPI

api = MemoryAPI(data_dir="~/.plm/data")
await api.initialize()
```

#### Methods

##### remember

Store a new memory.

```python
memory = await api.remember(
    content="I prefer Python for data processing",
    memory_type="PREFERENCE",          # FACT, BELIEF, PREFERENCE, SKILL, EVENT, CONTEXT
    truth_category="OPINION",           # ABSOLUTE, CONTEXTUAL, OPINION, INFERRED
    domains=["programming", "python"],
    tags=["language", "preference"],
    confidence=0.9,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | str | required | Memory content |
| `memory_type` | str | "FACT" | Type of memory |
| `truth_category` | str | "INFERRED" | Truth category |
| `domains` | list[str] | [] | Knowledge domains |
| `tags` | list[str] | [] | Tags for organization |
| `confidence` | float | 0.5 | Initial confidence (0-1) |
| `metadata` | dict | {} | Additional metadata |

##### get_context

Get relevant context for a query.

```python
context = await api.get_context(
    query="How should I structure my project?",
    max_memories=10,
    min_confidence=0.6,
    domains=["python", "architecture"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Query text |
| `max_memories` | int | 10 | Maximum memories to return |
| `min_confidence` | float | 0.5 | Minimum confidence threshold |
| `domains` | list[str] | None | Filter by domains |
| `tier` | str | None | Filter by tier |

##### augment_prompt

Augment a prompt with personal context.

```python
augmented = await api.augment_prompt(
    prompt="Help me write a Python function",
    format="xml",                       # xml, json, markdown
    include_identity=True,
    max_context_tokens=1000,
)

print(augmented.augmented_prompt)
print(augmented.memory_context)
print(augmented.memories_used)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Original prompt |
| `format` | str | "xml" | Output format |
| `include_identity` | bool | True | Include identity profile |
| `max_context_tokens` | int | 2000 | Max context size |

##### search

Search memories semantically.

```python
results = await api.search(
    query="programming preferences",
    limit=20,
    threshold=0.7,
)

for memory, score in results:
    print(f"{score:.2f}: {memory.content}")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Search query |
| `limit` | int | 10 | Maximum results |
| `threshold` | float | 0.5 | Minimum similarity |
| `domains` | list[str] | None | Filter by domains |
| `memory_type` | str | None | Filter by type |
| `tier` | str | None | Filter by tier |

##### get_beliefs

Get beliefs in a specific domain.

```python
beliefs = await api.get_beliefs(
    domain="programming",
    include_opinions=True,
)
```

##### get_preferences

Get user preferences.

```python
preferences = await api.get_preferences(
    category="tools",  # Optional category filter
)
```

##### validate

Validate or invalidate a memory.

```python
await api.validate(
    memory_id="mem-123",
    is_valid=True,
    feedback="This is correct",
)
```

##### forget

Remove a memory.

```python
await api.forget(memory_id="mem-123")
```

## Schema

### MemoryEntry

Core data model for memories.

```python
from memory.schema import (
    MemoryEntry,
    MemoryType,
    TruthCategory,
    MemoryTier,
    ConfidenceScore,
    MemorySource,
)

entry = MemoryEntry(
    content="I prefer tabs over spaces",
    memory_type=MemoryType.PREFERENCE,
    truth_category=TruthCategory.OPINION,
    tier=MemoryTier.LONG_TERM,
    confidence=ConfidenceScore(
        overall=0.9,
        source_reliability=1.0,
        recency=0.8,
        corroboration_count=3,
    ),
    domains=["programming", "style"],
    tags=["formatting", "preference"],
    sources=[
        MemorySource(
            source_type="code_review",
            source_id="pr-123",
            extracted_at=datetime.now(),
        ),
    ],
)
```

#### MemoryType

| Value | Description |
|-------|-------------|
| `FACT` | Verifiable information |
| `BELIEF` | Held convictions |
| `PREFERENCE` | Likes/dislikes |
| `SKILL` | Known capabilities |
| `EVENT` | Timestamped occurrences |
| `CONTEXT` | Domain/situational info |

#### TruthCategory

| Value | Description |
|-------|-------------|
| `ABSOLUTE` | Empirically verifiable |
| `CONTEXTUAL` | True in specific contexts |
| `OPINION` | Personal beliefs |
| `INFERRED` | Derived from patterns |

#### MemoryTier

| Value | Description |
|-------|-------------|
| `SHORT_TERM` | Session cache |
| `LONG_TERM` | Validated, searchable |
| `PERSISTENT` | Core identity |

### IdentityProfile

User identity information.

```python
from memory.schema import IdentityProfile

profile = IdentityProfile(
    name="Rod Higgins",
    email="rod@example.com",
    domains=["drupal", "python", "government"],
    preferences={
        "programming_style": "functional",
        "editor": "vscode",
    },
    skills=["drupal", "python", "react"],
)
```

## Storage

### StorageManager

Manages all three storage tiers.

```python
from memory.storage import StorageManager

manager = StorageManager(data_dir="~/.plm/data")
await manager.initialize()

# Store in short-term
await manager.store(entry, tier=MemoryTier.SHORT_TERM)

# Retrieve
memory = await manager.get(memory_id)

# Search across all tiers
results = await manager.search(query, limit=10)

# Promote to next tier
await manager.promote(memory_id)
```

### Individual Stores

```python
from memory.storage import DictStore, SQLiteStore, LanceDBStore

# Short-term (in-memory)
short_term = DictStore()

# Long-term (vector search)
long_term = LanceDBStore(path="~/.plm/data/lancedb")

# Persistent (SQLite)
persistent = SQLiteStore(path="~/.plm/data/persistent.db")
```

## Query

### MemoryQuery

Build complex queries.

```python
from memory.query import MemoryQuery

query = (
    MemoryQuery()
    .text("programming preferences")
    .domains(["python", "drupal"])
    .types([MemoryType.PREFERENCE, MemoryType.BELIEF])
    .min_confidence(0.7)
    .since(datetime(2024, 1, 1))
    .limit(20)
)

results = await api.execute_query(query)
```

## Export

### MemoryExporter

Export memories in various formats.

```python
from memory.export import MemoryExporter, ExportFormat, ExportConfig

exporter = MemoryExporter()

config = ExportConfig(
    format=ExportFormat.XML,
    include_confidence=True,
    include_sources=False,
    max_tokens=2000,
)

# Export for LLM context
context = await exporter.export_for_llm(
    memories=memories,
    config=config,
)

# Export to file
await exporter.export_to_file(
    memories=memories,
    path="memories.json",
    format=ExportFormat.JSON,
)
```

### Export Formats

| Format | Use Case |
|--------|----------|
| `XML` | Best for structured context injection |
| `JSON` | API responses, programmatic use |
| `MARKDOWN` | Human readable |
| `SYSTEM_PROMPT` | Direct LLM system prompt |

## LLM Integration

### MemoryAugmenter

Augment prompts with memory context.

```python
from memory.llm import MemoryAugmenter, AugmentationConfig

config = AugmentationConfig(
    max_memories=15,
    min_confidence=0.6,
    include_identity=True,
    format="xml",
)

augmenter = MemoryAugmenter(api, config)

result = await augmenter.augment(
    prompt="Help me with this code",
    context={"current_file": "app.py"},
)

print(result.augmented_prompt)
```

### Embedding Providers

```python
from memory.llm.embeddings import get_embedding_provider

# Local embeddings
provider = get_embedding_provider("local")

# Get embeddings
embeddings = await provider.embed(["text 1", "text 2"])
```

## Ingestion

### Data Source Adapters

```python
from memory.ingestion.sources import (
    LocalGitSource,
    GitHubSource,
    ObsidianSource,
    DocumentSource,
)

# Local git repositories
git_source = LocalGitSource(repo_path="~/code/myproject")
async for datapoint in git_source.extract():
    print(datapoint.content)

# GitHub repositories
github_source = GitHubSource(
    username="rod-higgins",
    include_repos=True,
    include_issues=True,
)

# Obsidian vault
obsidian_source = ObsidianSource(vault_path="~/Notes")

# Documents
doc_source = DocumentSource(
    path="~/Documents",
    extensions=[".pdf", ".docx", ".txt"],
)
```

### Ingestion Coordinator

```python
from memory.ingestion import IngestionCoordinator

coordinator = IngestionCoordinator(api)

# Add sources
coordinator.add_source(git_source)
coordinator.add_source(doc_source)

# Run ingestion
stats = await coordinator.run(
    batch_size=100,
    skip_embeddings=False,
)

print(f"Ingested: {stats['processed']}")
print(f"Errors: {stats['errors']}")
```

## SLM Training

### PersonalSLMTrainer

Train your personal language model.

```python
from memory.slm import PersonalSLMTrainer, TrainingConfig

config = TrainingConfig(
    base_model="meta-llama/Llama-3.2-3B",
    lora_r=16,
    lora_alpha=32,
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    output_dir="./models",
)

trainer = PersonalSLMTrainer(config)

# Prepare data from memories
await trainer.prepare_data(api)

# Train
trainer.train()

# Merge LoRA weights
trainer.merge_and_save()

# Export for Ollama
trainer.export_to_ollama("my-plm")
```

### Inference

```python
from memory.slm import PersonalSLM

slm = PersonalSLM.load("./models/my-plm")

# Generate context
context = slm.generate_context(
    query="What are my coding preferences?",
    max_tokens=500,
)

# Direct inference
response = slm.generate(
    prompt="Based on what you know about me...",
    max_tokens=200,
)
```

## CLI Commands

```bash
# Initialize
plm init

# Ingest data
plm ingest claude-history
plm ingest git-repos --path ~/code
plm ingest documents --path ~/Documents
plm ingest github --user rod-higgins

# Search
plm search "programming preferences"
plm search "drupal" --domain architecture

# Memory management
plm stats
plm promote --run-cycle
plm export --format json --output memories.json

# Training
plm train --base-model llama-3.2-3b --epochs 3
plm export-ollama my-plm

# Server
plm serve --port 8000
```

## Error Handling

```python
from memory.exceptions import (
    MemoryNotFoundError,
    StorageError,
    EmbeddingError,
    IngestionError,
    ValidationError,
)

try:
    memory = await api.get(memory_id)
except MemoryNotFoundError:
    print(f"Memory {memory_id} not found")
except StorageError as e:
    print(f"Storage error: {e}")
```

## Async Context Manager

```python
from memory import MemoryAPI

async with MemoryAPI() as api:
    await api.remember("Context managed memory")
    results = await api.search("memory")
# Automatically cleaned up
```

---

*See [ARCHITECTURE.md](./ARCHITECTURE.md) for system architecture.*
*See [VISION.md](./VISION.md) for project philosophy.*
