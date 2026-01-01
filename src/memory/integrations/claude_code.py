"""
Claude Code Integration.

Provides hooks and utilities for integrating the memory system
with Claude Code CLI.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from memory.api.memory_api import MemoryAPI


class ClaudeCodeIntegration:
    """
    Integration layer for Claude Code.

    Provides:
    1. Pre-prompt hooks for context injection
    2. Post-response hooks for learning
    3. Skill definitions for memory queries
    """

    def __init__(self, base_path: str = "~/memory/data"):
        self._base_path = base_path
        self._api: MemoryAPI | None = None

    async def initialize(self) -> None:
        """Initialize the API."""
        if self._api:
            return
        self._api = MemoryAPI(base_path=self._base_path)
        await self._api.initialize()

    async def close(self) -> None:
        """Close connections."""
        if self._api:
            await self._api.close()
            self._api = None

    async def get_context_for_prompt(
        self,
        user_message: str,
        max_memories: int = 10,
    ) -> str:
        """
        Get memory context to inject into Claude Code prompts.

        This is called by the pre-prompt hook.
        """
        await self.initialize()
        if not self._api:
            return ""

        return await self._api.get_context(
            user_message,
            format="claude",
            max_memories=max_memories,
        )

    async def learn_from_response(
        self,
        user_message: str,
        assistant_response: str,
    ) -> list[dict[str, Any]]:
        """
        Learn from an interaction.

        This is called by the post-response hook.
        """
        await self.initialize()
        if not self._api:
            return []

        memories = await self._api.learn_from_interaction(user_message, assistant_response)
        return [{"id": str(m.id), "content": m.content[:100]} for m in memories]


# === Hook Scripts ===
# These can be used directly as Claude Code hooks


async def pre_prompt_hook() -> None:
    """
    Claude Code pre-prompt hook.

    Reads user message from stdin, outputs memory context to stdout.

    Usage in ~/.claude/settings.json:
    {
      "hooks": {
        "pre_prompt": "python -m memory.integrations.claude_code pre_prompt"
      }
    }
    """
    # Read the prompt from stdin
    prompt_data = sys.stdin.read()

    try:
        data = json.loads(prompt_data)
        user_message = data.get("message", "")
    except json.JSONDecodeError:
        user_message = prompt_data

    if not user_message:
        return

    integration = ClaudeCodeIntegration()
    try:
        context = await integration.get_context_for_prompt(user_message)
        if context:
            # Output context injection
            result = {
                "system_prompt_append": context,
                "memories_injected": True,
            }
            print(json.dumps(result))
    finally:
        await integration.close()


async def post_response_hook() -> None:
    """
    Claude Code post-response hook.

    Reads interaction from stdin, learns from it.

    Usage in ~/.claude/settings.json:
    {
      "hooks": {
        "post_response": "python -m memory.integrations.claude_code post_response"
      }
    }
    """
    # Read the interaction from stdin
    interaction_data = sys.stdin.read()

    try:
        data = json.loads(interaction_data)
        user_message = data.get("user_message", "")
        assistant_response = data.get("assistant_response", "")
    except json.JSONDecodeError:
        return

    if not user_message or not assistant_response:
        return

    integration = ClaudeCodeIntegration()
    try:
        memories = await integration.learn_from_response(user_message, assistant_response)
        if memories:
            result = {
                "memories_created": len(memories),
                "memories": memories,
            }
            print(json.dumps(result))
    finally:
        await integration.close()


# === Skill for memory queries ===


def generate_skill_definition() -> dict[str, Any]:
    """
    Generate a Claude Code skill definition for memory queries.

    This skill allows users to query their memories directly.
    """
    return {
        "name": "memory",
        "description": "Query and manage personal memories",
        "commands": [
            {
                "name": "search",
                "description": "Search memories for relevant information",
                "usage": "/memory search <query>",
            },
            {
                "name": "recall",
                "description": "Recall known facts about a topic",
                "usage": "/memory recall <topic>",
            },
            {
                "name": "preferences",
                "description": "Show user preferences",
                "usage": "/memory preferences [domain]",
            },
            {
                "name": "remember",
                "description": "Store a new memory",
                "usage": "/memory remember <content>",
            },
            {
                "name": "context",
                "description": "Get memory context for current query",
                "usage": "/memory context",
            },
        ],
    }


async def execute_skill_command(command: str, args: list[str]) -> str:
    """Execute a memory skill command."""
    api = MemoryAPI()
    await api.initialize()

    try:
        if command == "search":
            query = " ".join(args) if args else ""
            result = await api.search(query)
            output = [f"Found {len(result.memories)} memories:\n"]
            for i, mem in enumerate(result.memories[:10], 1):
                output.append(f"{i}. [{mem.memory_type.value}] {mem.content[:100]}")
            return "\n".join(output)

        elif command == "recall":
            topic = " ".join(args) if args else ""
            result = await api.recall(topic)
            output = [f"Recalled {len(result.memories)} memories about '{topic}':\n"]
            for mem in result.memories:
                output.append(f"- {mem.content[:150]}")
            return "\n".join(output)

        elif command == "preferences":
            domain = args[0] if args else None
            result = await api.get_preferences(domain)
            output = ["User Preferences:\n"]
            for mem in result.memories:
                output.append(f"- {mem.content[:150]}")
            return "\n".join(output) if result.memories else "No preferences found."

        elif command == "remember":
            content = " ".join(args) if args else ""
            if not content:
                return "Usage: /memory remember <content>"
            mem = await api.remember(content)
            return f"Stored memory: {mem.id}"

        elif command == "context":
            # Get context for most recent query
            context = await api.get_context("current context")
            return context

        else:
            return f"Unknown command: {command}"

    finally:
        await api.close()


# === CLI Entry Point ===


def main() -> None:
    """CLI entry point for hooks."""
    if len(sys.argv) < 2:
        print("Usage: python -m memory.integrations.claude_code <command>")
        print("Commands: pre_prompt, post_response, skill")
        sys.exit(1)

    command = sys.argv[1]

    if command == "pre_prompt":
        asyncio.run(pre_prompt_hook())
    elif command == "post_response":
        asyncio.run(post_response_hook())
    elif command == "skill":
        if len(sys.argv) < 3:
            print(json.dumps(generate_skill_definition()))
        else:
            skill_cmd = sys.argv[2]
            skill_args = sys.argv[3:]
            result = asyncio.run(execute_skill_command(skill_cmd, skill_args))
            print(result)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
