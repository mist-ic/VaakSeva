"""
Reranker module for VaakSeva.

After hybrid retrieval returns top-20 candidates, the reranker re-scores them
with a cross-encoder for much higher precision before sending to the LLM.

Default backend: Qwen3-Reranker-0.6B (enabled by default)
  - Fast on CPU (~200-400ms for top-20 candidates)
  - Still outperforms no-reranking on MTEB multilingual reranking benchmarks
  - Supports 100+ languages including Hindi

Why NOT Qwen3-Reranker-8B in production:
  - 8B cross-encoder on CPU adds 2-5 seconds per request
  - This single component would blow the 6-second end-to-end budget
  - 0.6B is the practical sweet spot for CPU deployment

Alternative: Cohere Rerank v3.5 API (~50-100ms, ~$2/1000 searches)
  - If you need 8B-quality reranking within latency budget
  - Set RERANKER_BACKEND=cohere (future implementation)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from backend.config import settings
from backend.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], float]:
        """
        Rerank chunks by relevance to query.
        Returns (reranked_chunks[:top_k], rerank_ms).
        """


# ---------------------------------------------------------------------------
# Qwen3 cross-encoder reranker
# ---------------------------------------------------------------------------


class Qwen3Reranker(BaseReranker):
    """
    Cross-encoder reranker using Qwen3-Reranker-0.6B (default).

    The model scores (query, passage) pairs and returns a relevance score.
    We pass all top-20 candidates and return the top_k by score.

    CPU latency (0.6B, top-20 candidates): ~200-400ms
    CPU latency (8B, top-20 candidates): ~2000-5000ms -- too slow for pipeline

    To use 8B for better quality: set QWEN3_RERANKER_MODEL=Qwen/Qwen3-Reranker-8B
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import CrossEncoder

        model_name = model_name or settings.qwen3_reranker_model
        device = device or settings.reranker_device

        logger.info("Loading Qwen3 reranker: %s on %s", model_name, device)
        self._model = CrossEncoder(
            model_name,
            device=device,
            trust_remote_code=True,
            max_length=256,  # sufficient for scheme document chunks
        )
        logger.info("Qwen3 reranker ready (model=%s)", model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], float]:
        if not chunks:
            return [], 0.0

        t0 = time.perf_counter()

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._model.predict(pairs)

        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = []
        for score, chunk in scored[:top_k]:
            chunk.score = float(score)
            reranked.append(chunk)

        rerank_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Reranked %d -> %d candidates in %.1fms. Top score: %.4f",
            len(chunks),
            len(reranked),
            rerank_ms,
            reranked[0].score if reranked else 0.0,
        )

        return reranked, rerank_ms


# ---------------------------------------------------------------------------
# NoOp reranker (passthrough)
# ---------------------------------------------------------------------------


class NoOpReranker(BaseReranker):
    """
    Passthrough reranker -- returns candidates in original retrieval order.

    Use only for debugging or when CPU budget is extremely tight.
    In normal operation, Qwen3-Reranker-0.6B is fast enough on CPU.
    """

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], float]:
        return chunks[:top_k], 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_reranker() -> BaseReranker:
    """Return the configured reranker."""
    backend = settings.reranker_backend
    if backend == "qwen3":
        return Qwen3Reranker()
    elif backend == "noop":
        return NoOpReranker()
    else:
        raise ValueError(f"Unknown reranker backend: {backend!r}")
