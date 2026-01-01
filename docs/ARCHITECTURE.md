# PMS/PLM Technical Architecture

## Overview

The Personal Memory System (PMS) provides the data infrastructure, while the Personal Language Model (PLM) provides the intelligence layer. Together, they create a personalized AI system that learns and improves through every interaction.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                            │
│                                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│   │  Social  │  │   Docs   │  │  Media   │  │   Code   │  │    AI    │ │
│   │  Media   │  │  Notes   │  │  Photos  │  │   Git    │  │ History  │ │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│        │             │             │             │             │        │
│        └─────────────┴─────────────┴─────────────┴─────────────┘        │
│                                    │                                    │
│                                    ▼                                    │
│                        ┌─────────────────────┐                          │
│                        │   Universal         │                          │
│                        │   DataPoint Format  │                          │
│                        └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PERSONAL MEMORY SYSTEM (PMS)                        │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    THREE-TIER STORAGE                            │   │
│   │                                                                  │   │
│   │  SHORT-TERM          LONG-TERM            PERSISTENT             │   │
│   │  (In-Memory)         (LanceDB)            (SQLite)               │   │
│   │                                                                  │   │
│   │  • Session cache     • Vector search      • Core identity        │   │
│   │  • Unvalidated       • Semantic search    • Absolute truths      │   │
│   │  • 24-72hr TTL       • Validated facts    • Stable beliefs       │   │
│   │  • Fast access       • ~1M entries        • Permanent            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    PROMOTION ENGINE                              │   │
│   │                                                                  │   │
│   │  Short-term → Long-term:                                         │   │
│   │    • Confidence ≥ 0.6 AND corroborated ≥ 2 times                │   │
│   │    • Persisted 3+ days with no contradictions                    │   │
│   │    • Explicitly validated by user                                │   │
│   │                                                                  │   │
│   │  Long-term → Persistent:                                         │   │
│   │    • Truth category = ABSOLUTE AND confidence ≥ 0.9             │   │
│   │    • Tagged as "identity" for 30+ days                          │   │
│   │    • Consistent preference accessed 10+ times over 60+ days     │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PERSONAL LANGUAGE MODEL (PLM)                         │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                  TRAINING PIPELINE                               │   │
│   │                                                                  │   │
│   │  1. Data Preparation                                             │   │
│   │     • Convert memories to instruction format                     │   │
│   │     • Generate Q&A pairs from content                            │   │
│   │     • Weight by confidence and type                              │   │
│   │                                                                  │   │
│   │  2. Knowledge Distillation                                       │   │
│   │     • Learn from Claude/GPT responses                            │   │
│   │     • Soft targets (temperature=2.0)                             │   │
│   │     • Alpha blending with hard targets                           │   │
│   │                                                                  │   │
│   │  3. LoRA Fine-tuning                                             │   │
│   │     • Rank-16 adapters                                           │   │
│   │     • Target: attention + FFN layers                             │   │
│   │     • 4-bit quantization (QLoRA)                                 │   │
│   │                                                                  │   │
│   │  4. User-Specific Pruning                                        │   │
│   │     • Track domain activations                                   │   │
│   │     • Remove unused pathways                                     │   │
│   │     • Structured pruning (heads, neurons)                        │   │
│   │                                                                  │   │
│   │  5. Compression (ZipLLM)                                         │   │
│   │     • Combined pruning + quantization                            │   │
│   │     • Distillation to recover quality                            │   │
│   │     • Target: ~500MB model                                       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                CONTINUOUS LEARNING (Titan)                       │   │
│   │                                                                  │   │
│   │  Every Interaction:                                              │   │
│   │    1. Log query + response + teacher model                       │   │
│   │    2. Extract learnable content                                  │   │
│   │    3. Queue for next training batch                              │   │
│   │                                                                  │   │
│   │  Periodic Updates:                                               │   │
│   │    • Trigger: 10 interactions OR 24 hours                        │   │
│   │    • Incremental LoRA update                                     │   │
│   │    • Merge and checkpoint                                        │   │
│   │                                                                  │   │
│   │  User Feedback:                                                  │   │
│   │    • "Good" → weight × 1.5                                       │   │
│   │    • "Bad" → weight × 0.5                                        │   │
│   │    • "Correction" → weight × 3.0, use correction as target       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      INTEGRATION LAYER                                  │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                 CONTEXT INJECTION                                │   │
│   │                                                                  │   │
│   │  Input: User query                                               │   │
│   │                                                                  │   │
│   │  Process:                                                        │   │
│   │    1. PLM analyzes query intent                                  │   │
│   │    2. Retrieve relevant memories (hybrid search)                 │   │
│   │    3. PLM generates personalized context                         │   │
│   │    4. Format for target LLM (Claude XML, GPT JSON, etc.)        │   │
│   │                                                                  │   │
│   │  Output: Augmented prompt with personal context                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                 SUPPORTED TARGETS                                │   │
│   │                                                                  │   │
│   │  • Claude (Anthropic) - XML format with user_profile tags       │   │
│   │  • GPT-4 (OpenAI) - System prompt injection                     │   │
│   │  • Ollama (Local) - Direct model replacement                    │   │
│   │  • Any LLM - Generic JSON/Markdown context                      │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Memory Schema

### MemoryEntry

```python
MemoryEntry:
    id: UUID
    content: str                      # The memory content
    summary: str | None               # AI-generated summary
    content_hash: str                 # For deduplication

    # Classification
    tier: SHORT_TERM | LONG_TERM | PERSISTENT
    truth_category: ABSOLUTE | CONTEXTUAL | OPINION | INFERRED
    memory_type: FACT | BELIEF | PREFERENCE | SKILL | EVENT | CONTEXT

    # Confidence
    confidence:
        overall: float                # 0.0 - 1.0
        source_reliability: float     # How reliable is the source
        recency: float                # Time decay factor
        corroboration_count: int      # Times confirmed
        contradiction_count: int      # Times contradicted

    # Semantic
    embedding: list[float]            # Vector for similarity search
    tags: list[str]                   # User/system tags
    domains: list[str]                # Knowledge domains
    entities: list[str]               # Named entities

    # Relationships
    related_memories: list[UUID]
    contradicts: list[UUID]
    supports: list[UUID]
    supersedes: UUID | None

    # Provenance
    sources: list[MemorySource]       # Where this came from

    # Temporal
    created_at: datetime
    updated_at: datetime
    last_accessed: datetime
    access_count: int
```

### Truth Taxonomy

| Category | Definition | Examples | Promotion Criteria |
|----------|------------|----------|-------------------|
| **ABSOLUTE** | Empirically verifiable | Dates, measurements, spellings | External verification |
| **CONTEXTUAL** | True in specific contexts | "React is best for SPAs" | Domain + confidence |
| **OPINION** | Personal beliefs | "I prefer tabs over spaces" | Consistent expression |
| **INFERRED** | Derived from patterns | "User is a morning person" | Behavioral data |

### Memory Types

| Type | Description | Weight in PLM |
|------|-------------|---------------|
| **FACT** | Verifiable information | High |
| **BELIEF** | Held convictions | Medium-High |
| **PREFERENCE** | Likes/dislikes | High |
| **SKILL** | Known capabilities | Medium |
| **EVENT** | Timestamped occurrences | Low |
| **CONTEXT** | Domain/situational info | Medium |

## PLM Training Architecture

### Phase 1: Initial Training

```
┌─────────────────────────────────────────────────────────────────┐
│                    BASE MODEL SELECTION                          │
│                                                                  │
│  Recommended: Llama 3.2 3B                                       │
│  - Small enough for local inference                              │
│  - Large enough for quality reasoning                            │
│  - Good instruction-following capability                         │
│  - Open weights for fine-tuning                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                              │
│                                                                  │
│  From PMS memories, generate:                                    │
│                                                                  │
│  Instruction Format:                                             │
│  {                                                               │
│    "instruction": "What are the user's Drupal preferences?",    │
│    "input": "",                                                  │
│    "output": "The user prefers composition over inheritance..."  │
│  }                                                               │
│                                                                  │
│  Chat Format:                                                    │
│  {                                                               │
│    "messages": [                                                 │
│      {"role": "system", "content": "Personal AI assistant..."},  │
│      {"role": "user", "content": "What do I think about X?"},   │
│      {"role": "assistant", "content": "Based on your..."}       │
│    ]                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA FINE-TUNING                              │
│                                                                  │
│  Configuration:                                                  │
│    lora_r: 16           # Rank                                   │
│    lora_alpha: 32       # Scaling                                │
│    lora_dropout: 0.05                                            │
│    target_modules:                                               │
│      - q_proj, k_proj, v_proj, o_proj  # Attention               │
│      - gate_proj, up_proj, down_proj   # FFN                     │
│                                                                  │
│  Training:                                                       │
│    epochs: 3                                                     │
│    batch_size: 4                                                 │
│    learning_rate: 2e-4                                           │
│    gradient_accumulation: 4                                      │
│    quantization: 4-bit (QLoRA)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Knowledge Distillation

```
┌─────────────────────────────────────────────────────────────────┐
│                 DISTILLATION FROM TEACHER MODELS                 │
│                                                                  │
│  Teacher: Claude, GPT-4, or other high-quality LLM              │
│  Student: Your PLM                                               │
│                                                                  │
│  Process:                                                        │
│    1. User interacts with teacher model                          │
│    2. Log: query, response, teacher_model                        │
│    3. Generate soft targets (temperature-scaled logits)          │
│    4. Train student on combination:                              │
│                                                                  │
│       Loss = α * soft_target_loss + (1-α) * hard_target_loss    │
│                                                                  │
│       Where α = 0.5 (balance between teacher and ground truth)  │
│                                                                  │
│  Result: PLM learns how teacher models respond to YOU            │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: User-Specific Pruning

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOMAIN ACTIVATION TRACKING                    │
│                                                                  │
│  Track which knowledge domains you actually use:                 │
│                                                                  │
│  Example activation counts:                                      │
│    drupal: 450 (45%)                                             │
│    python: 300 (30%)                                             │
│    government: 150 (15%)                                         │
│    react: 80 (8%)                                                │
│    medicine: 2 (0.2%)                                            │
│    law: 5 (0.5%)                                                 │
│    cooking: 13 (1.3%)                                            │
│                                                                  │
│  Pruning decision:                                               │
│    - Keep: drupal, python, government, react (>5% activation)   │
│    - Prune: medicine, law, cooking (<1% activation)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STRUCTURED PRUNING                            │
│                                                                  │
│  Identify and remove:                                            │
│    - Attention heads focused on unused domains                   │
│    - FFN neurons with low activation for your queries           │
│    - Embedding dimensions unused in your vocabulary              │
│                                                                  │
│  Method:                                                         │
│    1. Compute importance scores per head/neuron                  │
│    2. Rank by activation frequency on your data                  │
│    3. Remove bottom 50% by importance                            │
│    4. Fine-tune to recover quality                               │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Compression (ZipLLM)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZipLLM COMPRESSION PIPELINE                   │
│                                                                  │
│  Original: 3B parameters (~6GB at fp16)                          │
│                                                                  │
│  Step 1: Structured Pruning (50%)                                │
│    → 1.5B parameters (~3GB)                                      │
│                                                                  │
│  Step 2: Quantization (4-bit)                                    │
│    → 1.5B parameters (~750MB)                                    │
│                                                                  │
│  Step 3: Distillation Recovery                                   │
│    → Train pruned model on unpruned outputs                      │
│    → Recover ~95% of original quality                            │
│                                                                  │
│  Final: ~500MB model with personalized knowledge                 │
└─────────────────────────────────────────────────────────────────┘
```

## Continuous Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    TITAN-STYLE CONTINUOUS LEARNING               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                  INTERACTION PHASE                       │     │
│  │                                                          │     │
│  │  User Query ──▶ PLM Context ──▶ Claude ──▶ Response     │     │
│  │       │                                        │         │     │
│  │       └────────────────────────────────────────┘         │     │
│  │                         │                                │     │
│  │                    Log Everything                        │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                  EXTRACTION PHASE                        │     │
│  │                                                          │     │
│  │  From interaction, extract:                              │     │
│  │    - New facts learned                                   │     │
│  │    - Preferences expressed                               │     │
│  │    - Corrections made                                    │     │
│  │    - Domains activated                                   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                  ACCUMULATION PHASE                      │     │
│  │                                                          │     │
│  │  Queue learning events until:                            │     │
│  │    - 10+ interactions accumulated                        │     │
│  │    - 24+ hours since last update                         │     │
│  │    - User explicitly requests update                     │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                  UPDATE PHASE                            │     │
│  │                                                          │     │
│  │  1. Prepare training batch from accumulated events       │     │
│  │  2. Incremental LoRA update (few steps)                  │     │
│  │  3. Merge weights                                        │     │
│  │  4. Checkpoint                                           │     │
│  │  5. Optional: pruning pass                               │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              │                                   │
│                              ▼                                   │
│                    Loop continues...                             │
└─────────────────────────────────────────────────────────────────┘
```

## Inference Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PLM INFERENCE PIPELINE                        │
│                                                                  │
│  Input: "How should I structure this Drupal module?"            │
│                                                                  │
│  Step 1: Query Analysis (PLM)                                    │
│    - Detected domains: [drupal, architecture, php]              │
│    - Detected intent: technical_guidance                         │
│    - Confidence: 0.92                                            │
│                                                                  │
│  Step 2: Memory Retrieval (PMS)                                  │
│    - Semantic search for relevant memories                       │
│    - Filter by domain match                                      │
│    - Rank by confidence and recency                              │
│    - Select top 10 memories                                      │
│                                                                  │
│  Step 3: Context Generation (PLM)                                │
│    - Generate personalized context from memories                 │
│    - Include: preferences, past decisions, constraints           │
│    - Format for target LLM                                       │
│                                                                  │
│  Step 4: Prompt Augmentation                                     │
│    <user_profile>                                                │
│      <preferences>                                               │
│        - Prefers composition over inheritance                    │
│        - Uses dependency injection                               │
│        - Follows Drupal coding standards strictly               │
│      </preferences>                                              │
│      <context>                                                   │
│        - Works on government websites                            │
│        - Security is high priority                               │
│        - Has built 50+ Drupal modules                           │
│      </context>                                                  │
│    </user_profile>                                               │
│                                                                  │
│  Step 5: External LLM Call                                       │
│    - Send augmented prompt to Claude/GPT                         │
│    - Receive personalized response                               │
│                                                                  │
│  Step 6: Learning                                                │
│    - Log interaction for future training                         │
│    - Update access counts on used memories                       │
│    - Queue for next model update                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Short-term storage | Python dict / Redis | Fast, ephemeral |
| Long-term storage | LanceDB | <10ms vector search, local-first |
| Persistent storage | SQLite + FTS5 | Human-inspectable, portable |
| Embeddings | sentence-transformers | Local, private, 384-dim |
| Base model | Llama 3.2 3B | Small, capable, open |
| Fine-tuning | LoRA/QLoRA + PEFT | Efficient, low memory |
| Inference | Ollama | Easy deployment, local |
| Compression | bitsandbytes | 4-bit quantization |

## Directory Structure

```
~/memory/
├── config/
│   └── default.toml           # Configuration
├── data/
│   ├── short_term/            # Session data
│   ├── long_term/
│   │   └── lancedb/           # Vector store
│   └── persistent/
│       └── core.sqlite        # Core memories
├── models/
│   ├── training_data/         # Prepared training data
│   ├── checkpoints/           # Training checkpoints
│   ├── lora_adapter/          # LoRA weights
│   └── merged/                # Final merged model
├── src/memory/
│   ├── schema/                # Data models
│   ├── storage/               # Storage backends
│   ├── ingestion/             # Data source adapters
│   ├── query/                 # Query interface
│   ├── export/                # LLM format exporters
│   ├── llm/                   # LLM integration
│   ├── slm/                   # PLM training & inference
│   └── api/                   # High-level API
└── docs/
    ├── VISION.md              # This document
    └── ARCHITECTURE.md        # Technical details
```

---

*For the philosophical vision and motivation, see [VISION.md](./VISION.md).*
