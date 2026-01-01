#!/usr/bin/env python3
"""
Run Personal Language Model using Ollama with memory context.

This provides a RAG-style approach using your ingested memories
without requiring full fine-tuning.

Usage:
    python scripts/run_plm.py
    python scripts/run_plm.py "What projects am I working on?"
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def get_relevant_memories(query: str, limit: int = 10) -> list[str]:
    """Retrieve relevant memories for context."""
    from memory.storage import StorageManager

    data_path = Path.home() / "memory" / "data"
    manager = StorageManager(base_path=str(data_path))
    await manager.initialize()

    # Extract keywords from query (remove common words)
    stop_words = {"what", "which", "how", "do", "does", "is", "are", "the", "a", "an",
                  "my", "i", "me", "have", "has", "in", "on", "for", "to", "of", "and"}
    words = [w.lower().rstrip("s") for w in query.split() if w.lower() not in stop_words]

    # Search with each keyword and combine results
    all_results = []
    seen_ids = set()
    for word in words:
        if len(word) >= 2:  # Skip very short words
            results, _ = await manager.text_search(word, limit=limit)
            for r in results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    all_results.append(r)

    await manager.close()
    return [m.content for m in all_results[:limit]]


def query_ollama(prompt: str, context: list[str], model: str = "tinyllama") -> str:
    """Query Ollama with memory context."""
    # Build context section - format as simple facts
    facts = "\n".join([f"- {c.strip()}" for c in context if c.strip()])

    full_prompt = f"""FACTS ABOUT THE USER:
{facts}

QUESTION: {prompt}

Using ONLY the facts above, provide a direct answer:"""

    # Query Ollama
    result = subprocess.run(
        ["ollama", "run", model, full_prompt],
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


async def interactive_mode():
    """Run interactive chat."""
    print("=" * 60)
    print("Personal Language Model (PLM) - Interactive Mode")
    print("=" * 60)
    print("Using your ingested memories for context.")
    print("Type 'quit' or 'exit' to end.\n")

    # Check if model is available
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "tinyllama" not in result.stdout.lower():
        print("Note: tinyllama not found. Pulling model...")
        subprocess.run(["ollama", "pull", "tinyllama"])

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            # Get relevant memories
            print("  [Searching memories...]")
            memories = await get_relevant_memories(query)
            print(f"  [Found {len(memories)} relevant memories]")

            # Query with context
            print("\nPLM:", end=" ")
            response = query_ollama(query, memories)
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


async def single_query(query: str, debug: bool = False):
    """Run a single query."""
    memories = await get_relevant_memories(query)

    if debug or not memories:
        print(f"[Found {len(memories)} relevant memories]")
        if memories:
            print("\n--- Relevant memories ---")
            for i, m in enumerate(memories[:5], 1):
                print(f"{i}. {m[:100]}...")
            print("--- End memories ---\n")

    if not memories:
        print("No relevant memories found for this query.")
        return

    response = query_ollama(query, memories)
    print(response)


async def main():
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        debug = "--debug" in args
        if debug:
            args.remove("--debug")
        query = " ".join(args)
        await single_query(query, debug=debug)
    else:
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
