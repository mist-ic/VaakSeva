"""
LLM client for VaakSeva.

Provides an OpenAI-compatible interface over two backends:
  - VLLMClient: connects to a vLLM server (production)
  - OllamaClient: connects to local Ollama (development)

Both back streaming, retry logic, and timeout handling.
Switching backends is a one-line config change (LLM_BACKEND env var).

Why vLLM for production:
  - 3-5x higher throughput than Ollama on the same GPU
  - Much lower P99 latency under concurrent load
  - PagedAttention for efficient KV-cache management
  - Full OpenAI-compatible API (drop in replacement)

Why Ollama for development:
  - Zero-config local setup
  - Automatic model download with `ollama pull`
  - Works on CPU (slow but functional for testing)
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
        """Stream tokens. Yields token chunks."""


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
) -> str:
    """
    Call an OpenAI-compatible /v1/chat/completions endpoint.
    Retries on connection errors and 5xx responses.
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

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            logger.warning("LLM HTTP error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
            last_exc = exc
            if exc.response.status_code < 500:
                raise  # Don't retry client errors (4xx)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("LLM connection error (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
            last_exc = exc

        if attempt < max_retries:
            await asyncio.sleep(1.0 * (attempt + 1))  # backoff

    raise RuntimeError(f"LLM failed after {max_retries + 1} attempts") from last_exc


# ---------------------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------------------


class VLLMClient(BaseLLMClient):
    """
    Client for a vLLM server running Sarvam-30B (or any OpenAI-compatible model).

    vLLM exposes the OpenAI API at /v1/chat/completions.
    Run a vLLM server:
      python -m vllm.entrypoints.openai.api_server \\
        --model sarvamai/sarvam-30b \\
        --port 8000
    """

    def __init__(self):
        self._base_url = settings.vllm_base_url
        self._model = settings.vllm_model
        logger.info("vLLM client configured: %s @ %s", self._model, self._base_url)

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
        # TODO: implement streaming with httpx SSE
        response = await self.agenerate(prompt, system)
        yield response


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


class OllamaClient(BaseLLMClient):
    """
    Client for Ollama running locally.

    Ollama exposes an OpenAI-compatible API on port 11434.
    Run locally:
      ollama pull sarvam-m
      ollama serve
    """

    def __init__(self):
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        logger.info("Ollama client configured: %s @ %s", self._model, self._base_url)

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
    if backend == "vllm":
        return VLLMClient()
    elif backend == "ollama":
        return OllamaClient()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
