"""
Embedding module for VaakSeva.

Supports two backends:
  - Qwen3Embedder: Qwen/Qwen3-Embedding-0.6B (default, #1 MTEB Multilingual class)
  - E5Embedder: intfloat/multilingual-e5-large-instruct (560M, legacy fallback)

Qwen3-Embedding-0.6B vs multilingual-e5-large-instruct:
  - Qwen3: MTEB Multilingual score 70.58 vs E5 at ~58 — measurably better Hindi retrieval
  - Qwen3-0.6B: 1024D embeddings, fast on CPU (~50-100ms per query)
  - Qwen3-8B: 7168D, use for offline indexing by setting QWEN3_EMBED_MODEL env var

For offline document indexing with best quality:
  QWEN3_EMBED_MODEL=Qwen/Qwen3-Embedding-8B python scripts/ingest.py
  (Note: 8B uses 7168D vectors -- requires re-creating the Weaviate collection)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from backend.config import settings

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Asymmetric retrieval instructions
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
# Qwen3 embedder (default -- best multilingual retrieval quality)
# ---------------------------------------------------------------------------


class Qwen3Embedder(BaseEmbedder):
    """
    Wraps Qwen/Qwen3-Embedding-0.6B (default) via sentence-transformers.

    Default config: 0.6B model, 1024D embeddings, fast on CPU.
    For offline indexing with best quality, set:
      QWEN3_EMBED_MODEL=Qwen/Qwen3-Embedding-8B

    MTEB Multilingual: Qwen3-0.6B still outperforms multilingual-e5-large.
    Both 0.6B and 8B use the same asymmetric instruction format.
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
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Qwen3 embedder ready (model=%s dim=%d)", model_name, self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> list[float]:
        """Embed with asymmetric query instruction for retrieval."""
        prompted = f"{QWEN3_QUERY_INSTRUCTION}{text}"
        embedding = self._model.encode(
            prompted,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (no prefix — Qwen3 embeds docs without instruction)."""
        embeddings = self._model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# Multilingual E5 (legacy CPU fallback)
# ---------------------------------------------------------------------------


class E5Embedder(BaseEmbedder):
    """
    Wraps intfloat/multilingual-e5-large-instruct.

    560M parameters, 1024D embeddings, runs on CPU.
    MTEB score ~58 vs Qwen3-0.6B at 70+ -- use Qwen3 unless E5 is required
    for compatibility with an existing Weaviate collection.

    Uses "query: " and "passage: " prefixes for asymmetric retrieval.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        model_name = model_name or settings.e5_embed_model
        device = device or settings.embed_device

        logger.info("Loading E5 embedder: %s on %s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("E5 embedder ready (dim=%d)", self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

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
    """Return the configured embedder."""
    backend = settings.embedder_backend
    if backend == "qwen3":
        return Qwen3Embedder()
    elif backend == "e5":
        return E5Embedder()
    else:
        raise ValueError(f"Unknown embedder backend: {backend!r}")


# Module-level singleton populated lazily on first embedder() call
_embedder_instance: BaseEmbedder | None = None


def embedder() -> BaseEmbedder:
    """Singleton embedder accessor."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = get_embedder()
    return _embedder_instance
