#!/usr/bin/env python3
"""
Train Personal Language Model (PLM) using LoRA fine-tuning.

Usage:
    # Quick test (1 epoch, TinyLlama)
    python scripts/train_plm.py --test

    # Full training (recommended with GPU)
    python scripts/train_plm.py

    # Custom config
    python scripts/train_plm.py --model meta-llama/Llama-3.2-3B --epochs 3
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


async def main():
    parser = argparse.ArgumentParser(description="Train Personal Language Model")
    parser.add_argument("--test", action="store_true", help="Quick test run (1 epoch)")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Base model (default: TinyLlama-1.1B)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", default="~/memory/models/personal-slm",
                        help="Output directory")
    args = parser.parse_args()

    from memory.slm import PersonalSLMTrainer
    from memory.slm.trainer import TrainingConfig

    # Configure for test or full training
    if args.test:
        config = TrainingConfig(
            base_model=args.model,
            output_dir=str(Path(args.output).expanduser()),
            num_epochs=1,
            batch_size=1,
            use_4bit=False,  # CPU-friendly
        )
    else:
        config = TrainingConfig(
            base_model=args.model,
            output_dir=str(Path(args.output).expanduser()),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            use_4bit=True,  # QLoRA for efficiency
        )

    trainer = PersonalSLMTrainer(config)

    print("=" * 60)
    print("PLM Training Configuration")
    print("=" * 60)
    for k, v in trainer.get_training_stats().items():
        print(f"  {k}: {v}")

    # Prepare data
    print("\n[1/4] Preparing training data...")
    stats = await trainer.prepare_data()
    print(f"  Training examples: {stats.get('training_examples', 0)}")

    # Train
    print("\n[2/4] Training model (this may take a while)...")
    result = trainer.train()
    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"  Training loss: {result.get('training_loss', 'N/A')}")
    print(f"  Adapter saved: {result.get('adapter_path', 'N/A')}")

    # Merge
    print("\n[3/4] Merging LoRA weights...")
    merged_path = trainer.merge_and_save()
    print(f"  Merged model: {merged_path}")

    # Export to Ollama
    print("\n[4/4] Exporting to Ollama...")
    export_result = trainer.export_to_ollama("personal-slm")
    if export_result.get("success"):
        print(f"  Model registered: {export_result['model_name']}")
        print(f"  Run: ollama run {export_result['model_name']}")
    else:
        print(f"  Export error: {export_result.get('error')}")
        print(f"  Modelfile: {export_result.get('modelfile_path')}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
