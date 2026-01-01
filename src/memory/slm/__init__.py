"""
Personal SLM (Small Language Model) for memory-based reasoning.

This module provides:
- Training data preparation from memories
- Fine-tuning pipelines (LoRA/QLoRA)
- Personal SLM inference
- Continuous learning (Titan-style)
- Model compression (ZipLLM)
- Integration with larger LLMs

Architecture:
    Generic SLM (Llama 3B)
           │
           ▼
    Knowledge Distillation ◀── Claude/GPT responses
           │
           ▼
    User-Specific Pruning ◀── Remove unused domains
           │
           ▼
    ZipLLM Compression ◀── 4-bit quant + structured pruning
           │
           ▼
    Personal SLM (~500MB) ◀── Continuously improving
           │
           ▼
    Context Injection → External LLMs
"""

from memory.slm.continuous import (
    ContinuousLearner,
    ContinuousLearningConfig,
    LearningEvent,
    ZipLLMCompressor,
)
from memory.slm.data import MemoryDataset, prepare_training_data
from memory.slm.learning import (
    EnhancedLearner,
    LearnedConcept,
    LearningStrategy,
    SkillProgression,
)
from memory.slm.model import HybridPersonalSLM, PersonalSLM, get_personal_slm
from memory.slm.trainer import PersonalSLMTrainer, TrainingConfig

__all__ = [
    # Training
    "PersonalSLMTrainer",
    "TrainingConfig",
    # Model
    "PersonalSLM",
    "get_personal_slm",
    "HybridPersonalSLM",
    # Data
    "MemoryDataset",
    "prepare_training_data",
    # Continuous Learning
    "ContinuousLearner",
    "ContinuousLearningConfig",
    "LearningEvent",
    # Enhanced Learning
    "EnhancedLearner",
    "LearnedConcept",
    "LearningStrategy",
    "SkillProgression",
    # Compression
    "ZipLLMCompressor",
]
