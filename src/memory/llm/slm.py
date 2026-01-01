"""
Local SLM (Small Language Model) provider for memory operations.

Handles truth categorization, entity extraction, and memory validation.
All processing is local - no external API calls.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from memory.schema.memory_entry import MemoryType, TruthCategory


class SLMProvider(ABC):
    """Abstract base class for local SLM providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion for the given prompt."""
        pass

    async def categorize_truth(self, content: str) -> TruthCategory:
        """
        Categorize content as ABSOLUTE, CONTEXTUAL, OPINION, or INFERRED.
        """
        prompt = f"""Classify this statement into exactly one category:

ABSOLUTE - Empirically verifiable facts (dates, measurements, definitions, spelling)
CONTEXTUAL - True in specific contexts (technology comparisons, best practices)
OPINION - Personal beliefs, preferences, cannot be objectively proven
INFERRED - Derived or assumed, not explicitly stated

Statement: "{content[:500]}"

Respond with only the category name in caps (ABSOLUTE, CONTEXTUAL, OPINION, or INFERRED):"""

        response = await self.complete(prompt, max_tokens=20)
        response = response.strip().upper()

        mapping = {
            "ABSOLUTE": TruthCategory.ABSOLUTE,
            "CONTEXTUAL": TruthCategory.CONTEXTUAL,
            "OPINION": TruthCategory.OPINION,
            "INFERRED": TruthCategory.INFERRED,
        }

        return mapping.get(response, TruthCategory.INFERRED)

    async def classify_memory_type(self, content: str) -> MemoryType:
        """
        Classify content as FACT, BELIEF, PREFERENCE, SKILL, EVENT, or CONTEXT.
        """
        prompt = f"""Classify this content into exactly one category:

FACT - Verifiable information or data
BELIEF - Held convictions or worldview
PREFERENCE - Likes, dislikes, or preferences
SKILL - Known capabilities or expertise
EVENT - Timestamped occurrence or action
CONTEXT - Domain or situational information

Content: "{content[:500]}"

Respond with only the category name in caps:"""

        response = await self.complete(prompt, max_tokens=20)
        response = response.strip().upper()

        mapping = {
            "FACT": MemoryType.FACT,
            "BELIEF": MemoryType.BELIEF,
            "PREFERENCE": MemoryType.PREFERENCE,
            "SKILL": MemoryType.SKILL,
            "EVENT": MemoryType.EVENT,
            "CONTEXT": MemoryType.CONTEXT,
            "RELATIONSHIP": MemoryType.RELATIONSHIP,
        }

        return mapping.get(response, MemoryType.FACT)

    async def extract_entities(self, content: str) -> list[str]:
        """
        Extract named entities from content.
        """
        prompt = f"""Extract named entities from this text. Include:
- People names
- Organizations
- Technologies/tools
- Projects
- Locations

Text: "{content[:1000]}"

Return as a JSON array of strings. Only include clear, specific entities.
Example: ["Python", "Drupal", "Fair Work Commission", "Sydney"]

JSON array:"""

        response = await self.complete(prompt, max_tokens=200)

        try:
            # Try to parse JSON from response
            response = response.strip()
            if response.startswith("["):
                end = response.find("]") + 1
                if end > 0:
                    return json.loads(response[:end])
            return []
        except json.JSONDecodeError:
            return []

    async def extract_domains(self, content: str) -> list[str]:
        """
        Extract knowledge domains from content.
        """
        prompt = f"""Identify the knowledge domains this content belongs to.

Common domains: programming, python, javascript, typescript, drupal, react, vue,
database, devops, ai, machine_learning, government, business, personal, creative

Content: "{content[:500]}"

Return as a JSON array of lowercase domain names (max 5).
Example: ["programming", "drupal", "government"]

JSON array:"""

        response = await self.complete(prompt, max_tokens=100)

        try:
            response = response.strip()
            if response.startswith("["):
                end = response.find("]") + 1
                if end > 0:
                    domains = json.loads(response[:end])
                    return [d.lower().replace(" ", "_") for d in domains[:5]]
            return []
        except json.JSONDecodeError:
            return []

    async def generate_summary(self, content: str, max_length: int = 100) -> str:
        """
        Generate a concise summary of content.
        """
        prompt = f"""Summarize this in one clear, concise sentence (max {max_length} characters):

{content[:2000]}

Summary:"""

        response = await self.complete(prompt, max_tokens=100)
        return response.strip()[:max_length]

    async def detect_contradiction(
        self,
        memory_a: str,
        memory_b: str,
    ) -> tuple[bool, str | None]:
        """
        Detect if two memories contradict each other.

        Returns (is_contradiction, explanation).
        """
        prompt = f"""Do these two statements contradict each other?

Statement A: "{memory_a[:500]}"

Statement B: "{memory_b[:500]}"

Respond with:
- "YES: <brief explanation>" if they contradict
- "NO" if they don't contradict

Response:"""

        response = await self.complete(prompt, max_tokens=100)
        response = response.strip().upper()

        if response.startswith("YES"):
            explanation = response[4:].strip(": ")
            return True, explanation if explanation else "Contradiction detected"
        return False, None

    async def synthesize_context(
        self,
        query: str,
        memories: list[str],
        user_name: str = "the user",
    ) -> str:
        """
        Synthesize meaningful insights from raw memories for a given query.

        Instead of just returning raw memory snippets, this uses the SLM to
        extract and synthesize relevant knowledge into coherent context.
        """
        if not memories:
            return ""

        # Combine memories into context
        memory_text = "\n".join([f"- {m[:300]}" for m in memories[:15]])

        prompt = f"""You are helping to build a personal profile. Based on these memory snippets about {user_name}, extract and synthesize the key insights relevant to the question.

Question: {query}

Memory snippets:
{memory_text}

Provide a concise, synthesized summary of what these memories reveal about {user_name} that's relevant to the question. Focus on:
- Key facts and experiences
- Expertise and skills demonstrated
- Patterns and preferences
- Relevant context

Be specific and factual. Only include information actually present in the memories. Keep it under 200 words.

Synthesized insights:"""

        response = await self.complete(prompt, max_tokens=400)
        return response.strip()

    async def extract_user_profile(
        self,
        memories: list[str],
        user_name: str = "the user",
    ) -> dict[str, Any]:
        """
        Extract a structured user profile from memories.
        """
        memory_text = "\n".join([f"- {m[:200]}" for m in memories[:20]])

        prompt = f"""Based on these memory snippets, extract a structured profile for {user_name}.

Memories:
{memory_text}

Create a JSON profile with these fields (only include if evidence exists):
{{
  "expertise": ["list of skills/technologies"],
  "roles": ["job titles or roles held"],
  "industries": ["industry experience"],
  "preferences": ["work preferences"],
  "current_focus": "what they're currently working on"
}}

JSON profile:"""

        response = await self.complete(prompt, max_tokens=300)

        try:
            response = response.strip()
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                if end > start:
                    return json.loads(response[start:end])
            return {}
        except json.JSONDecodeError:
            return {}


class OllamaSLM(SLMProvider):
    """
    Local SLM provider using Ollama.

    Requires Ollama to be running locally with a model pulled.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
    ):
        self._model_name = model_name
        self._base_url = base_url

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion using Ollama."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": kwargs.get("max_tokens", 100),
                            "temperature": kwargs.get("temperature", 0.1),
                        },
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()["response"]
            except httpx.HTTPError:
                # Return empty string on error (graceful degradation)
                return ""

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self._base_url}/api/tags",
                    timeout=5.0,
                )
                response.raise_for_status()
                models = response.json().get("models", [])
                return any(m.get("name", "").startswith(self._model_name.split(":")[0]) for m in models)
            except httpx.HTTPError:
                return False


class MockSLM(SLMProvider):
    """
    Mock SLM for testing when no local model is available.

    Uses simple heuristics instead of actual model inference.
    """

    @property
    def model_name(self) -> str:
        return "mock-slm"

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return heuristic-based responses."""
        prompt_lower = prompt.lower()

        # Truth categorization heuristics
        if "absolute" in prompt_lower and "contextual" in prompt_lower:
            if any(w in prompt_lower for w in ["prefer", "like", "believe", "think", "opinion"]):
                return "OPINION"
            elif any(w in prompt_lower for w in ["date", "time", "always", "never", "is a", "are a"]):
                return "ABSOLUTE"
            elif any(w in prompt_lower for w in ["better", "faster", "for", "when", "if"]):
                return "CONTEXTUAL"
            return "INFERRED"

        # Memory type heuristics
        if "fact" in prompt_lower and "belief" in prompt_lower:
            if any(w in prompt_lower for w in ["prefer", "like", "want"]):
                return "PREFERENCE"
            elif any(w in prompt_lower for w in ["believe", "think", "faith"]):
                return "BELIEF"
            elif any(w in prompt_lower for w in ["know", "can", "able", "experience"]):
                return "SKILL"
            elif any(w in prompt_lower for w in ["did", "happened", "was", "went"]):
                return "EVENT"
            return "FACT"

        # Entity extraction heuristics
        if "extract" in prompt_lower and "entities" in prompt_lower:
            return "[]"

        # Domain extraction heuristics
        if "domains" in prompt_lower:
            if "drupal" in prompt_lower:
                return '["drupal", "programming"]'
            elif "python" in prompt_lower:
                return '["python", "programming"]'
            return '["programming"]'

        # Contradiction detection
        if "contradict" in prompt_lower:
            return "NO"

        # Default summary
        if "summarize" in prompt_lower:
            return "Summary not available."

        return ""


def get_slm_provider(
    provider: str = "ollama",
    model: str = "llama3.2:3b",
    **kwargs: Any,
) -> SLMProvider:
    """
    Factory function to create an SLM provider.

    Args:
        provider: Provider type ("ollama" or "mock")
        model: Model name
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured SLM provider
    """
    if provider == "ollama":
        return OllamaSLM(model_name=model, **kwargs)
    elif provider == "mock":
        return MockSLM()
    else:
        raise ValueError(f"Unknown SLM provider: {provider}")
