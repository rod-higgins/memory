"""
Personal SLM Trainer.

Fine-tunes a small language model on personal memories using
LoRA (Low-Rank Adaptation) for efficient training.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.slm.data import prepare_training_data


@dataclass
class TrainingConfig:
    """Configuration for SLM training."""

    # Base model
    base_model: str = "meta-llama/Llama-3.2-3B"
    model_alias: str = "llama3.2:3b"  # Ollama name

    # LoRA parameters
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4

    # Output
    output_dir: str = "~/memory/models"
    model_name: str = "personal-slm"

    # Quantization
    use_4bit: bool = True  # QLoRA
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"


class PersonalSLMTrainer:
    """
    Trainer for fine-tuning a personal SLM on memory data.

    Uses LoRA/QLoRA for efficient fine-tuning on consumer hardware.

    Training Process:
    1. Load base model with quantization (4-bit)
    2. Attach LoRA adapters to attention layers
    3. Train on memory-derived instruction data
    4. Merge and save for inference

    Usage:
        trainer = PersonalSLMTrainer()
        await trainer.prepare_data()
        trainer.train()
        trainer.export_to_ollama()
    """

    def __init__(self, config: TrainingConfig | None = None):
        self._config = config or TrainingConfig()
        self._output_dir = Path(self._config.output_dir).expanduser()
        self._training_data_path: Path | None = None
        self._model = None
        self._tokenizer = None

    async def prepare_data(
        self,
        storage_path: str = "~/memory/data",
        min_confidence: float = 0.4,
    ) -> dict[str, Any]:
        """Prepare training data from memories."""
        output_path = self._output_dir / "training_data"
        output_path.mkdir(parents=True, exist_ok=True)

        stats = await prepare_training_data(
            storage_path=storage_path,
            output_path=str(output_path),
            min_confidence=min_confidence,
        )

        self._training_data_path = Path(stats["chat_path"])
        return stats

    def train(self) -> dict[str, Any]:
        """
        Train the personal SLM using LoRA.

        Requires: transformers, peft, bitsandbytes, accelerate

        Returns training statistics.
        """
        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except ImportError as e:
            return {
                "error": f"Missing dependencies: {e}",
                "install": "pip install transformers peft bitsandbytes accelerate datasets",
            }

        if not self._training_data_path or not self._training_data_path.exists():
            return {"error": "Training data not prepared. Call prepare_data() first."}

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.base_model,
            trust_remote_code=True,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Quantization config for QLoRA
        if self._config.use_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=self._config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self._config.bnb_4bit_compute_dtype),
            )
        else:
            bnb_config = None

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self._config.use_4bit:
            self._model = prepare_model_for_kbit_training(self._model)

        # LoRA config
        lora_config = LoraConfig(
            r=self._config.lora_r,
            lora_alpha=self._config.lora_alpha,
            lora_dropout=self._config.lora_dropout,
            target_modules=self._config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self._model = get_peft_model(self._model, lora_config)

        # Load dataset
        dataset = load_dataset("json", data_files=str(self._training_data_path))

        # Tokenize
        def tokenize(example: dict[str, Any]) -> dict[str, Any]:
            messages = example["messages"]
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return self._tokenizer(
                text,
                truncation=True,
                max_length=self._config.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self._output_dir / "checkpoints"),
            num_train_epochs=self._config.num_epochs,
            per_device_train_batch_size=self._config.batch_size,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            learning_rate=self._config.learning_rate,
            warmup_steps=self._config.warmup_steps,
            logging_steps=10,
            save_steps=100,
            fp16=True,
            optim="paged_adamw_8bit",
        )

        # Trainer
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False),
        )

        # Train
        result = trainer.train()

        # Save LoRA adapter
        self._model.save_pretrained(str(self._output_dir / "lora_adapter"))
        self._tokenizer.save_pretrained(str(self._output_dir / "lora_adapter"))

        return {
            "training_loss": result.training_loss,
            "epochs": self._config.num_epochs,
            "adapter_path": str(self._output_dir / "lora_adapter"),
        }

    def merge_and_save(self) -> str:
        """Merge LoRA weights with base model and save."""
        if not self._model:
            raise RuntimeError("Model not trained. Call train() first.")

        # Merge LoRA weights
        merged_model = self._model.merge_and_unload()

        # Save merged model
        output_path = self._output_dir / "merged"
        merged_model.save_pretrained(str(output_path))
        self._tokenizer.save_pretrained(str(output_path))

        return str(output_path)

    def export_to_ollama(self, model_name: str | None = None) -> dict[str, Any]:
        """
        Export the trained model to Ollama format.

        Creates a Modelfile and registers with Ollama.
        """
        model_name = model_name or self._config.model_name
        merged_path = self._output_dir / "merged"

        if not merged_path.exists():
            return {"error": "Merged model not found. Call merge_and_save() first."}

        # Create Modelfile
        modelfile_path = self._output_dir / "Modelfile"
        modelfile_content = f"""FROM {merged_path}

PARAMETER temperature 0.7
PARAMETER num_ctx 2048

SYSTEM You are a personal AI assistant with deep knowledge of the user's preferences, beliefs, skills, and context. You provide personalized responses based on this understanding.
"""
        modelfile_path.write_text(modelfile_content)

        # Register with Ollama
        try:
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "model_name": model_name,
                    "message": f"Model registered as '{model_name}'. Use: ollama run {model_name}",
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "modelfile_path": str(modelfile_path),
                }

        except FileNotFoundError:
            return {
                "success": False,
                "error": "Ollama not found. Install from https://ollama.ai",
                "modelfile_path": str(modelfile_path),
            }

    def get_training_stats(self) -> dict[str, Any]:
        """Get information about training configuration."""
        return {
            "base_model": self._config.base_model,
            "lora_rank": self._config.lora_r,
            "lora_alpha": self._config.lora_alpha,
            "epochs": self._config.num_epochs,
            "batch_size": self._config.batch_size,
            "learning_rate": self._config.learning_rate,
            "use_4bit": self._config.use_4bit,
            "output_dir": str(self._output_dir),
        }
