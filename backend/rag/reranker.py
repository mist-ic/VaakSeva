"""
Reranker module for VaakSeva.

After hybrid retrieval returns top-50 candidates, the reranker
re-scores them with a cross-encoder for much higher precision.

Backends:
  - Qwen3Reranker: Qwen/Qwen3-Reranker-8B (best quality, GPU recommended)
  - NoOpReranker: passthrough — returns candidates in original order (dev mode)

Qwen3-Reranker-8B consistently lifts recall@5 by 8-15% over retrieval alone
on multilingual benchmarks (BEIR, MIRACL).
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
    Cross-encoder reranker using Qwen/Qwen3-Reranker-8B.

    The model takes (query, passage) pairs and outputs a relevance score.
    We pass all candidates and return the top_k by score.

    VRAM: ~4-6 GB (FP16) or ~2-3 GB (int8).
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
            max_length=512,
        )
        logger.info("Qwen3 reranker ready")

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
            "Reranked %d -> %d candidates in %.1f ms. Top score: %.4f",
            len(chunks),
            len(reranked),
            rerank_ms,
            reranked[0].score if reranked else 0.0,
        )

        return reranked, rerank_ms


# ---------------------------------------------------------------------------
# NoOp reranker (passthrough for dev mode)
# ---------------------------------------------------------------------------


class NoOpReranker(BaseReranker):
    """
    Passthrough reranker — returns candidates in original retrieval order.

    Used in development or when GPU is not available.
    Enables testing the pipeline without loading the 8B reranker model.
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
        raise ValueError(f"Unknown reranker backend: {backend}")
