"""
LLM Comparison Module.

Provides unified interface to query multiple LLMs with and without PLM context.
Supports: Claude, ChatGPT, GitHub Copilot, Amazon Q/Bedrock
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# Load environment variables if not already loaded
from dotenv import load_dotenv

env_locations = [
    Path(__file__).parent.parent.parent.parent / ".env",
    Path.cwd() / ".env",
    Path.home() / "memory" / ".env",
]
for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path)
        break


@dataclass
class LLMResponse:
    """Response from an LLM."""

    provider: str
    model: str
    response: str
    with_context: bool
    latency_ms: float
    error: str | None = None
    context_used: str | None = None


class LLMProvider:
    """Base class for LLM providers."""

    name: str = "base"
    display_name: str = "Base Provider"
    model: str = "unknown"

    async def query(
        self,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Query the LLM."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this provider is configured."""
        return False

    def get_config_help(self) -> str:
        """Get help text for configuring this provider."""
        return "Not configured"


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider."""

    name = "claude"
    display_name = "Claude"
    model = "claude-sonnet-4-20250514"

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_config_help(self) -> str:
        return "Set ANTHROPIC_API_KEY environment variable"

    async def query(
        self,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        import time

        start = time.time()

        if not self.api_key:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=0,
                error="ANTHROPIC_API_KEY not set",
            )

        # Build system prompt
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)
        if context:
            system_parts.append(f"\n\n{context}")
        full_system = "\n".join(system_parts) if system_parts else None

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": message}],
                }
                if full_system:
                    payload["system"] = full_system

                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                )

                latency = (time.time() - start) * 1000

                if response.status_code != 200:
                    return LLMResponse(
                        provider=self.name,
                        model=self.model,
                        response="",
                        with_context=bool(context),
                        latency_ms=latency,
                        error=f"API error: {response.status_code} - {response.text[:200]}",
                    )

                result = response.json()
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    response=result["content"][0]["text"],
                    with_context=bool(context),
                    latency_ms=latency,
                    context_used=context[:200] if context else None,
                )

        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start) * 1000,
                error=str(e),
            )


class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT API provider."""

    name = "chatgpt"
    display_name = "ChatGPT"
    model = "gpt-4o"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_config_help(self) -> str:
        return "Set OPENAI_API_KEY environment variable"

    async def query(
        self,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        import time

        start = time.time()

        if not self.api_key:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=0,
                error="OPENAI_API_KEY not set",
            )

        # Build messages
        messages = []
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)
        if context:
            system_parts.append(f"\n\n{context}")
        if system_parts:
            messages.append({"role": "system", "content": "\n".join(system_parts)})
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 2048,
                    },
                )

                latency = (time.time() - start) * 1000

                if response.status_code != 200:
                    return LLMResponse(
                        provider=self.name,
                        model=self.model,
                        response="",
                        with_context=bool(context),
                        latency_ms=latency,
                        error=f"API error: {response.status_code} - {response.text[:200]}",
                    )

                result = response.json()
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    response=result["choices"][0]["message"]["content"],
                    with_context=bool(context),
                    latency_ms=latency,
                    context_used=context[:200] if context else None,
                )

        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start) * 1000,
                error=str(e),
            )


class CopilotProvider(LLMProvider):
    """GitHub Copilot provider via gh CLI."""

    name = "copilot"
    display_name = "GitHub Copilot"
    model = "copilot-chat"

    def __init__(self):
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                result = subprocess.run(
                    ["gh", "auth", "status"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self._available = result.returncode == 0
            except Exception:
                self._available = False
        return self._available

    def get_config_help(self) -> str:
        return "Run 'gh auth login' and ensure Copilot is enabled"

    async def query(
        self,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        import time

        start = time.time()

        if not self.is_available():
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=0,
                error="GitHub CLI not authenticated. Run 'gh auth login'",
            )

        # Build prompt with context
        full_prompt = message
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {message}"

        try:
            # Use gh copilot suggest or gh api for chat
            # Note: gh copilot requires the extension to be installed
            result = subprocess.run(
                ["gh", "api", "copilot_internal/v2/token"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # Try alternate approach - use GitHub Models API
                return await self._query_via_github_models(message, context, start)

            # Parse token and make copilot request
            token_data = json.loads(result.stdout)
            token = token_data.get("token")

            if not token:
                return await self._query_via_github_models(message, context, start)

            # Make Copilot chat request
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.githubcopilot.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Editor-Version": "vscode/1.85.0",
                        "Editor-Plugin-Version": "copilot-chat/0.12.0",
                    },
                    json={
                        "messages": [{"role": "user", "content": full_prompt}],
                        "model": "gpt-4o",
                        "stream": False,
                    },
                )

                latency = (time.time() - start) * 1000

                if response.status_code != 200:
                    return await self._query_via_github_models(message, context, start)

                result_json = response.json()
                return LLMResponse(
                    provider=self.name,
                    model="copilot-chat",
                    response=result_json["choices"][0]["message"]["content"],
                    with_context=bool(context),
                    latency_ms=latency,
                    context_used=context[:200] if context else None,
                )

        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start) * 1000,
                error=f"Copilot error: {str(e)}",
            )

    async def _query_via_github_models(
        self, message: str, context: str | None, start: float
    ) -> LLMResponse:
        """Fallback to GitHub Models API."""
        import time

        try:
            # Get GitHub token
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                # Try reading from config file
                from pathlib import Path

                import yaml

                gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
                if gh_config.exists():
                    with open(gh_config) as f:
                        hosts = yaml.safe_load(f)
                    token = hosts.get("github.com", {}).get("oauth_token")
                else:
                    return LLMResponse(
                        provider=self.name,
                        model="github-models",
                        response="",
                        with_context=bool(context),
                        latency_ms=(time.time() - start) * 1000,
                        error="Could not get GitHub token",
                    )
            else:
                token = result.stdout.strip()

            # Build prompt
            full_prompt = message
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion: {message}"

            # Use GitHub Models API (models.github.com)
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://models.inference.ai.azure.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "messages": [{"role": "user", "content": full_prompt}],
                        "model": "gpt-4o",
                        "max_tokens": 2048,
                    },
                )

                latency = (time.time() - start) * 1000

                if response.status_code != 200:
                    return LLMResponse(
                        provider=self.name,
                        model="github-models",
                        response="",
                        with_context=bool(context),
                        latency_ms=latency,
                        error=f"GitHub Models API error: {response.status_code}",
                    )

                result_json = response.json()
                return LLMResponse(
                    provider=self.name,
                    model="github-models-gpt4o",
                    response=result_json["choices"][0]["message"]["content"],
                    with_context=bool(context),
                    latency_ms=latency,
                    context_used=context[:200] if context else None,
                )

        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model="github-models",
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start) * 1000,
                error=f"GitHub Models error: {str(e)}",
            )


class AmazonQProvider(LLMProvider):
    """Amazon Bedrock provider using boto3."""

    name = "amazonq"
    display_name = "Amazon Bedrock"
    model = "amazon.titan-text-express-v1"

    def __init__(self):
        self._available: bool | None = None
        self._client = None

    def _get_client(self):
        """Get or create boto3 Bedrock Runtime client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('bedrock-runtime', region_name='us-east-1')
            except Exception:
                pass
        return self._client

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import boto3
                sts = boto3.client('sts')
                sts.get_caller_identity()
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def get_config_help(self) -> str:
        return "Configure AWS credentials and enable Bedrock model access"

    async def query(
        self,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        import time

        start = time.time()

        if not self.is_available():
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=0,
                error="AWS credentials not configured",
            )

        # Build prompt
        full_prompt = message
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion: {message}"

        try:
            client = self._get_client()
            if not client:
                return LLMResponse(
                    provider=self.name,
                    model=self.model,
                    response="",
                    with_context=bool(context),
                    latency_ms=0,
                    error="Could not create Bedrock client. Install boto3.",
                )

            # Use Amazon Titan Text Express
            response = client.invoke_model(
                modelId=self.model,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "inputText": full_prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 2048,
                        "temperature": 0.7,
                    }
                })
            )

            latency = (time.time() - start) * 1000
            result = json.loads(response['body'].read())
            output_text = result.get("results", [{}])[0].get("outputText", "")

            return LLMResponse(
                provider=self.name,
                model=self.model,
                response=output_text.strip(),
                with_context=bool(context),
                latency_ms=latency,
                context_used=context[:200] if context else None,
            )

        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model=self.model,
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start) * 1000,
                error=f"Bedrock error: {str(e)[:200]}",
            )

    async def _query_via_q_cli(
        self, message: str, context: str | None, start: float
    ) -> LLMResponse:
        """Fallback to Amazon Q Developer CLI if available."""
        import time

        try:
            # Check if q CLI is available
            result = subprocess.run(
                ["which", "q"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return LLMResponse(
                    provider=self.name,
                    model="bedrock-unavailable",
                    response="",
                    with_context=bool(context),
                    latency_ms=(time.time() - start) * 1000,
                    error="Bedrock not accessible. Enable Bedrock in AWS Console.",
                )

            # Use Q CLI
            full_prompt = message
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion: {message}"

            result = subprocess.run(
                ["q", "chat", full_prompt],
                capture_output=True,
                text=True,
                timeout=60,
            )

            latency = (time.time() - start) * 1000

            if result.returncode != 0:
                return LLMResponse(
                    provider=self.name,
                    model="amazon-q-cli",
                    response="",
                    with_context=bool(context),
                    latency_ms=latency,
                    error=f"Q CLI error: {result.stderr[:200]}",
                )

            return LLMResponse(
                provider=self.name,
                model="amazon-q-cli",
                response=result.stdout,
                with_context=bool(context),
                latency_ms=latency,
                context_used=context[:200] if context else None,
            )

        except Exception as e:
            return LLMResponse(
                provider=self.name,
                model="amazon-q",
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start) * 1000,
                error=f"Amazon Q error: {str(e)}",
            )


class LLMCompare:
    """Compare responses from multiple LLMs."""

    def __init__(self):
        self.providers: dict[str, LLMProvider] = {
            "claude": ClaudeProvider(),
            "chatgpt": ChatGPTProvider(),
            "copilot": CopilotProvider(),
            "amazonq": AmazonQProvider(),
        }

    def get_available_providers(self) -> list[dict[str, Any]]:
        """Get list of available providers with status."""
        return [
            {
                "id": name,
                "name": provider.display_name,
                "model": provider.model,
                "available": provider.is_available(),
                "config_help": provider.get_config_help(),
            }
            for name, provider in self.providers.items()
        ]

    async def query_single(
        self,
        provider_id: str,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Query a single provider (LLM or SLM)."""
        import time

        # Handle Ollama/SLM providers
        if provider_id.startswith("ollama_") or provider_id.startswith("slm_"):
            return await self._query_ollama(provider_id, message, context, system_prompt)

        # Handle built-in LLM providers
        provider = self.providers.get(provider_id)
        if not provider:
            return LLMResponse(
                provider=provider_id,
                model="unknown",
                response="",
                with_context=bool(context),
                latency_ms=0,
                error=f"Unknown provider: {provider_id}",
            )

        return await provider.query(message, context, system_prompt)

    async def _query_ollama(
        self,
        provider_id: str,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Query an Ollama/SLM model."""
        import httpx
        import time

        # Extract model name from provider ID
        if provider_id.startswith("ollama_"):
            model_name = provider_id.replace("ollama_", "").replace("_", ":")
        else:
            # SLM from config - need to load config
            import json
            from pathlib import Path
            config_path = Path.home() / "memory" / "data" / "persistent" / "models.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                slm_id = provider_id.replace("slm_", "")
                slm = next((s for s in config.get("slms", []) if s.get("id") == slm_id), None)
                if slm:
                    model_name = slm.get("model")
                else:
                    return LLMResponse(
                        provider=provider_id,
                        model="unknown",
                        response="",
                        with_context=bool(context),
                        latency_ms=0,
                        error=f"SLM not found: {provider_id}",
                    )
            else:
                return LLMResponse(
                    provider=provider_id,
                    model="unknown",
                    response="",
                    with_context=bool(context),
                    latency_ms=0,
                    error="SLM configuration not found",
                )

        # Build messages
        messages = []
        full_system = system_prompt or "You are a helpful AI assistant."
        if context:
            full_system += f"\n\nRelevant context from the user's personal memories:\n{context}"
        messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": message})

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False,
                    },
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code != 200:
                    return LLMResponse(
                        provider=provider_id,
                        model=model_name,
                        response="",
                        with_context=bool(context),
                        latency_ms=latency_ms,
                        error=f"Ollama error: {response.text}",
                    )

                result = response.json()
                return LLMResponse(
                    provider=provider_id,
                    model=model_name,
                    response=result.get("message", {}).get("content", "No response"),
                    with_context=bool(context),
                    latency_ms=latency_ms,
                )

        except httpx.ConnectError:
            return LLMResponse(
                provider=provider_id,
                model=model_name,
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start_time) * 1000,
                error="Ollama not running. Start with: ollama serve",
            )
        except httpx.TimeoutException:
            return LLMResponse(
                provider=provider_id,
                model=model_name,
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start_time) * 1000,
                error="Ollama timeout - model may be loading",
            )
        except Exception as e:
            return LLMResponse(
                provider=provider_id,
                model=model_name,
                response="",
                with_context=bool(context),
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def query_all(
        self,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
        providers: list[str] | None = None,
    ) -> dict[str, LLMResponse]:
        """Query all (or specified) providers in parallel."""
        provider_ids = providers or list(self.providers.keys())

        tasks = [
            self.query_single(pid, message, context, system_prompt)
            for pid in provider_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            pid: (
                result
                if isinstance(result, LLMResponse)
                else LLMResponse(
                    provider=pid,
                    model="error",
                    response="",
                    with_context=bool(context),
                    latency_ms=0,
                    error=str(result),
                )
            )
            for pid, result in zip(provider_ids, results)
        }

    async def compare(
        self,
        message: str,
        context: str | None = None,
        system_prompt: str | None = None,
        providers: list[str] | None = None,
    ) -> dict[str, dict[str, LLMResponse]]:
        """
        Compare responses with and without context.

        Returns dict with 'with_context' and 'without_context' keys,
        each containing responses from all providers.
        """
        # Query with context
        with_context_task = self.query_all(
            message, context, system_prompt, providers
        )

        # Query without context
        without_context_task = self.query_all(
            message, None, system_prompt, providers
        )

        with_context, without_context = await asyncio.gather(
            with_context_task, without_context_task
        )

        return {
            "with_context": with_context,
            "without_context": without_context,
        }
