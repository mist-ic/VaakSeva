"""
Hybrid retrieval module for VaakSeva.

Uses Weaviate's native hybrid search:
  - Dense vector search (cosine similarity with Qwen3/E5 embeddings)
  - Sparse BM25F keyword search (built-in, server-side, no client work needed)
  - Fusion via RRF with tunable alpha parameter

Why Weaviate over Qdrant/ChromaDB:
  - BM25F is computed server-side (zero client-side work)
  - Native hybrid query in one round-trip
  - Metadata filtering built-in
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import weaviate
import weaviate.classes as wvc

from backend.config import settings
from backend.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weaviate connection
# ---------------------------------------------------------------------------


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Create a Weaviate client connected to the configured host."""
    client = weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )
    return client


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    """
    Create the GovernmentSchemes collection if it doesn't exist.

    The collection uses:
      - text2vec vectorizer: NONE (we supply our own vectors)
      - BM25 indexing: enabled automatically on all text properties
    """
    collection_name = settings.weaviate_collection

    if client.collections.exists(collection_name):
        logger.debug("Collection '%s' already exists", collection_name)
        return

    logger.info("Creating Weaviate collection: %s", collection_name)
    client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(
                name="chunk_id",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=True,
                index_searchable=False,
            ),
            wvc.config.Property(
                name="scheme_id",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=True,
                index_searchable=True,
            ),
            wvc.config.Property(
                name="scheme_name",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=True,
                index_searchable=True,
            ),
            wvc.config.Property(
                name="language",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=True,
                index_searchable=False,
            ),
            wvc.config.Property(
                name="content",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=False,
                index_searchable=True,  # BM25 index
            ),
            wvc.config.Property(
                name="chunk_index",
                data_type=wvc.config.DataType.INT,
                skip_vectorization=True,
                index_filterable=True,
                index_searchable=False,
            ),
            wvc.config.Property(
                name="source_path",
                data_type=wvc.config.DataType.TEXT,
                skip_vectorization=True,
                index_filterable=False,
                index_searchable=False,
            ),
        ],
    )
    logger.info("Collection '%s' created successfully", collection_name)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    top_k: int = 50
    rerank_top_k: int = 5
    alpha: float = 0.75   # 1.0=pure dense, 0.0=pure BM25
    language_filter: str | None = None   # "hi" or "en" or None


class HybridRetriever:
    """
    Weaviate hybrid retriever with BM25F + dense vector search.

    The alpha parameter controls the blend:
      alpha=1.0  => pure dense (cosine similarity only)
      alpha=0.0  => pure BM25F (keyword only)
      alpha=0.75 => 75% dense + 25% BM25 (default — good for Hindi RAG)
    """

    def __init__(
        self,
        client: weaviate.WeaviateClient,
        embedder,
        config: RetrievalConfig | None = None,
    ):
        self._client = client
        self._embedder = embedder
        self._config = config or RetrievalConfig(
            top_k=settings.top_k_retrieval,
            rerank_top_k=settings.top_k_rerank,
            alpha=settings.hybrid_alpha,
        )
        self._collection = client.collections.get(settings.weaviate_collection)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        alpha: float | None = None,
        language_filter: str | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        """
        Run hybrid search and return (chunks, retrieval_ms).

        Returns top_k candidates for downstream reranking.
        """
        k = top_k or self._config.top_k
        a = alpha if alpha is not None else self._config.alpha
        lang = language_filter or self._config.language_filter

        # Embed the query (dense vector)
        t0 = time.perf_counter()
        query_vector = self._embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # Build filters
        filters = None
        if lang:
            filters = wvc.query.Filter.by_property("language").equal(lang)

        # Hybrid search
        t1 = time.perf_counter()
        response = self._collection.query.hybrid(
            query=query,               # BM25F uses this
            vector=query_vector,       # dense uses this
            alpha=a,
            limit=k,
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(score=True),
            return_properties=[
                "chunk_id",
                "scheme_id",
                "scheme_name",
                "language",
                "content",
                "chunk_index",
                "source_path",
            ],
        )
        retrieval_ms = (time.perf_counter() - t1) * 1000

        chunks = []
        for obj in response.objects:
            p = obj.properties
            chunks.append(
                RetrievedChunk(
                    chunk_id=p.get("chunk_id", ""),
                    scheme_name=p.get("scheme_name", ""),
                    scheme_id=p.get("scheme_id", ""),
                    content=p.get("content", ""),
                    language=p.get("language", ""),
                    source_url=p.get("source_path"),
                    chunk_index=p.get("chunk_index", 0),
                    score=obj.metadata.score if obj.metadata else 0.0,
                )
            )

        logger.debug(
            "Hybrid retrieval: query=%r, k=%d, alpha=%.2f, results=%d, embed_ms=%.1f, retrieval_ms=%.1f",
            query[:60],
            k,
            a,
            len(chunks),
            embed_ms,
            retrieval_ms,
        )

        return chunks, retrieval_ms

# alpha=0.75 default: 75 percent dense, 25 percent BM25F
