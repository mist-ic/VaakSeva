"""
Embedding module for VaakSeva.

Supports two backends:
  - Qwen3Embedder: Qwen/Qwen3-Embedding-8B (best quality, GPU recommended)
  - E5Embedder: intfloat/multilingual-e5-large-instruct (560M, CPU friendly)

Both implement the same interface so swapping is a config change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from backend.config import settings

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Query instruction for asymmetric retrieval (document vs query embeddings)
QWEN3_QUERY_INSTRUCTION = "Retrieve relevant documents for the following query: "
E5_QUERY_INSTRUCTION = "query: "
E5_DOCUMENT_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""


# ---------------------------------------------------------------------------
# Qwen3 embedder (best quality)
# ---------------------------------------------------------------------------


class Qwen3Embedder(BaseEmbedder):
    """
    Wraps Qwen/Qwen3-Embedding-8B via sentence-transformers.

    Uses the recommended query instruction for asymmetric retrieval.
    Dimension: 7168 (can be truncated with matryoshka).

    VRAM: ~4-6 GB on GPU.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        model_name = model_name or settings.qwen3_embed_model
        device = device or settings.embed_device

        logger.info("Loading Qwen3 embedder: %s on %s", model_name, device)
        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Qwen3 embedder ready (dim=%d)", self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> list[float]:
        prompted = f"{QWEN3_QUERY_INSTRUCTION}{text}"
        embedding = self._model.encode(
            prompted,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# Multilingual E5 embedder (CPU friendly fallback)
# ---------------------------------------------------------------------------


class E5Embedder(BaseEmbedder):
    """
    Wraps intfloat/multilingual-e5-large-instruct.

    560M parameters, strong on MTEB(Indic), runs on CPU.
    Uses "query: " and "passage: " prefixes for asymmetric retrieval.
    Dimension: 1024.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        model_name = model_name or settings.e5_embed_model
        device = device or settings.embed_device

        logger.info("Loading E5 embedder: %s on %s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("E5 embedder ready (dim=%d)", self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> list[float]:
        prompted = f"{E5_QUERY_INSTRUCTION}{text}"
        embedding = self._model.encode(prompted, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prompted = [f"{E5_DOCUMENT_PREFIX}{t}" for t in texts]
        embeddings = self._model.encode(
            prompted,
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedder() -> BaseEmbedder:
    """Return the configured embedder. Cached after first call."""
    backend = settings.embedder_backend
    if backend == "qwen3":
        return Qwen3Embedder()
    elif backend == "e5":
        return E5Embedder()
    else:
        raise ValueError(f"Unknown embedder backend: {backend}")


# Module-level singleton — populated lazily on first get_embedder() call
_embedder_instance: BaseEmbedder | None = None


def embedder() -> BaseEmbedder:
    """Singleton embedder accessor."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = get_embedder()
    return _embedder_instance

# E5 instruction prefix: query: / passage: asymmetric format
