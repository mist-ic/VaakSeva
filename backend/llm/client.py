"""
LLM client for VaakSeva.

Backends (in order of preference for production):
  1. SarvamLLMClient  - Sarvam-30B via hosted API (free April 2026, Hindi-native)
  2. GroqLLMClient    - Llama-3.3-70B via Groq LPU (250-500 tok/s, speed fallback)
  3. VLLMClient       - self-hosted vLLM (production GPU deployment)
  4. OllamaClient     - local Ollama (CPU development)

All backends implement BaseLLMClient and use OpenAI-compatible /v1/chat/completions.
Switching between backends is a one-line config change (LLM_BACKEND env var).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLMClient(ABC):
    @abstractmethod
    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion. Returns full response string."""

    @abstractmethod
    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Stream tokens. Yields token string chunks."""


# ---------------------------------------------------------------------------
# Shared OpenAI-compatible helper
# ---------------------------------------------------------------------------


async def _openai_chat_complete(
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    extra_headers: dict | None = None,
) -> str:
    """
    Call an OpenAI-compatible /v1/chat/completions endpoint.

    extra_headers allows passing API keys for Sarvam (api-subscription-key)
    and Groq (Authorization: Bearer <key>).
    Retries on connection errors and 5xx. Does not retry 4xx.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "LLM HTTP error (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, exc,
            )
            last_exc = exc
            if exc.response.status_code < 500:
                raise  # don't retry 4xx
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "LLM connection error (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, exc,
            )
            last_exc = exc

        if attempt < max_retries:
            await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"LLM failed after {max_retries + 1} attempts") from last_exc


# ---------------------------------------------------------------------------
# Sarvam LLM client (primary)
# ---------------------------------------------------------------------------


class SarvamLLMClient(BaseLLMClient):
    """
    Sarvam-30B via Sarvam AI hosted API.

    Why primary: free as of April 2026, Hindi-native (trained on 16T tokens
    across 22 Indian languages), 63.3% GPQA vs Llama 3.1 70B at 40.9%.
    TTFT ~1.2s on hosted API. OpenAI-compatible endpoint.

    Auth: api-subscription-key header (not Authorization: Bearer).
    """

    def __init__(self) -> None:
        self._base_url = settings.sarvam_base_url
        self._model = settings.sarvam_llm_model
        self._api_key = settings.sarvam_api_key

        if not self._api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. "
                "Sign up at https://docs.sarvam.ai for free API access."
            )

        logger.info("SarvamLLM: %s @ %s", self._model, self._base_url)

    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        from backend.llm.prompts import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return await _openai_chat_complete(
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            extra_headers={"api-subscription-key": self._api_key},
        )

    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        response = await self.agenerate(prompt, system)
        yield response


# ---------------------------------------------------------------------------
# Groq LLM client (speed fallback)
# ---------------------------------------------------------------------------


class GroqLLMClient(BaseLLMClient):
    """
    Llama-3.3-70B via Groq LPU inference (speed fallback).

    Groq delivers 250-500 tokens/second on LPU hardware.
    Use when Sarvam API latency exceeds the pipeline budget.

    Llama 3.3-70B handles Hindi well with a strong system prompt,
    though lacks Sarvam-30B's dedicated Indic tokenizer.

    Pricing: ~$0.59/M input, $0.79/M output (effectively zero at demo volume).
    Auth: Authorization: Bearer header (standard OpenAI pattern).
    """

    def __init__(self) -> None:
        self._base_url = settings.groq_base_url
        self._model = settings.groq_model
        self._api_key = settings.groq_api_key

        if not self._api_key:
            raise ValueError("GROQ_API_KEY is not set. Get a free key at console.groq.com")

        logger.info("GroqLLM: %s @ %s", self._model, self._base_url)

    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        from backend.llm.prompts import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return await _openai_chat_complete(
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            extra_headers={"Authorization": f"Bearer {self._api_key}"},
        )

    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        response = await self.agenerate(prompt, system)
        yield response


# ---------------------------------------------------------------------------
# vLLM client (self-hosted GPU)
# ---------------------------------------------------------------------------


class VLLMClient(BaseLLMClient):
    """
    Client for a self-hosted vLLM server running Sarvam-30B.

    vLLM provides 3-5x higher throughput than Ollama via PagedAttention.
    Use for production GPU deployments where API latency is not acceptable.

    Run:
      python -m vllm.entrypoints.openai.api_server \\
        --model sarvamai/sarvam-30b --port 8000
    """

    def __init__(self) -> None:
        self._base_url = settings.vllm_base_url
        self._model = settings.vllm_model
        logger.info("vLLM client: %s @ %s", self._model, self._base_url)

    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        from backend.llm.prompts import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return await _openai_chat_complete(
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        response = await self.agenerate(prompt, system)
        yield response


# ---------------------------------------------------------------------------
# Ollama client (local dev)
# ---------------------------------------------------------------------------


class OllamaClient(BaseLLMClient):
    """
    Client for Ollama running locally (development only).

    Zero-config local setup. OpenAI-compatible endpoint on port 11434.
    Works on CPU (slow but functional for testing the pipeline).

    Run:
      ollama pull sarvam-m
      ollama serve
    """

    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        logger.info("Ollama client: %s @ %s", self._model, self._base_url)

    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        from backend.llm.prompts import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return await _openai_chat_complete(
            base_url=self._base_url,
            model=self._model,
            messages=messages,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    async def astream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        response = await self.agenerate(prompt, system)
        yield response


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_client() -> BaseLLMClient:
    """Return the configured LLM client."""
    backend = settings.llm_backend
    if backend == "sarvam":
        return SarvamLLMClient()
    elif backend == "groq":
        return GroqLLMClient()
    elif backend == "vllm":
        return VLLMClient()
    elif backend == "ollama":
        return OllamaClient()
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}")
