"""
End-to-end RAG pipeline for VaakSeva.

Pipeline stages (with latency tracking at each step):
  1. Safety check (input sanitization, injection detection)
  2. Embed query
  3. Hybrid retrieve (Weaviate BM25F + dense)
  4. Rerank (Qwen3-Reranker or NoOp)
  5. Build context string from top-K chunks
  6. LLM generation with grounded Hindi prompt
  7. Output validation (fact-check against structured DB)
  8. Log full request with timings

The pipeline is designed for easy component swapping:
  - Swapping Weaviate -> Qdrant: change retriever.py only
  - Swapping Sarvam-30B -> Llama-3: change config.py only
  - Swapping Qwen3-Reranker -> bge-reranker: change reranker.py only
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from backend.config import settings
from backend.models.schemas import (
    PipelineTimings,
    QueryResponse,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full retrieval-augmented generation pipeline.

    Lazy-loads all heavy components (embedder, retriever, reranker, LLM client)
    on first use so startup is fast.
    """

    def __init__(self):
        self._embedder = None
        self._retriever = None
        self._reranker = None
        self._llm = None
        self._safety_filter = None
        self._output_validator = None
        self._obs_logger = None
        self._weaviate_client = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from backend.rag.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _get_retriever(self):
        if self._retriever is None:
            from backend.rag.retriever import HybridRetriever, get_weaviate_client, ensure_collection
            self._weaviate_client = get_weaviate_client()
            ensure_collection(self._weaviate_client)
            self._retriever = HybridRetriever(
                client=self._weaviate_client,
                embedder=self._get_embedder(),
            )
        return self._retriever

    def _get_reranker(self):
        if self._reranker is None:
            from backend.rag.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

    def _get_llm(self):
        if self._llm is None:
            from backend.llm.client import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    def _get_safety_filter(self):
        if self._safety_filter is None:
            from backend.safety.input_filter import InputFilter
            self._safety_filter = InputFilter()
        return self._safety_filter

    def _get_output_validator(self):
        if self._output_validator is None:
            from backend.safety.output_validator import OutputValidator
            self._output_validator = OutputValidator(settings.schemes_structured_path)
        return self._output_validator

    def _get_obs_logger(self):
        if self._obs_logger is None:
            from backend.observability.logger import RequestLogger
            self._obs_logger = RequestLogger(settings.log_dir)
        return self._obs_logger

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] योजना: {chunk.scheme_name}\n{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def aquery(
        self,
        query_text: str,
        user_profile: dict | None = None,
        conversation_history: list | None = None,
    ) -> QueryResponse:
        """
        Run the full RAG pipeline asynchronously.

        Returns a QueryResponse with timings and retrieved chunks.
        """
        request_id = str(uuid.uuid4())
        timings: dict[str, float] = {}
        user_profile = user_profile or {}
        conversation_history = conversation_history or []

        total_t0 = time.perf_counter()

        # 1. Safety check
        safety_filter = self._get_safety_filter()
        safety_result = safety_filter.check(query_text)
        if not safety_result.is_safe:
            logger.warning("Request %s blocked by safety filter: %s", request_id, safety_result.flagged_patterns)
            return QueryResponse(
                request_id=request_id,
                response_text="क्षमा करें, मैं इस प्रकार के प्रश्नों का उत्तर नहीं दे सकता।",
                safety=safety_result,
                timings=PipelineTimings(total_ms=(time.perf_counter() - total_t0) * 1000),
            )

        sanitized_query = safety_result.sanitized_input

        # 2. Embed
        t0 = time.perf_counter()
        embedder = self._get_embedder()
        query_vector = embedder.embed_query(sanitized_query)
        timings["embedding_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Retrieve
        retriever = self._get_retriever()
        candidates, retrieval_ms = retriever.retrieve(sanitized_query)
        timings["retrieval_ms"] = retrieval_ms

        # 4. Rerank
        reranker = self._get_reranker()
        t1 = time.perf_counter()
        top_chunks, rerank_ms = reranker.rerank(
            sanitized_query,
            candidates,
            top_k=settings.top_k_rerank,
        )
        timings["rerank_ms"] = rerank_ms

        if not top_chunks:
            return QueryResponse(
                request_id=request_id,
                response_text="इस विषय पर मेरे पास पर्याप्त जानकारी नहीं है। कृपया अपना प्रश्न अधिक विस्तार से पूछें।",
                timings=PipelineTimings(**timings),
            )

        # 5. Build context
        context = self._build_context(top_chunks)

        # 6. LLM generation
        llm = self._get_llm()
        from backend.llm.prompts import build_rag_prompt
        prompt = build_rag_prompt(
            query=sanitized_query,
            context=context,
            user_profile=user_profile,
            conversation_history=conversation_history,
        )

        t2 = time.perf_counter()
        response_text = await llm.agenerate(prompt)
        timings["llm_ms"] = (time.perf_counter() - t2) * 1000

        # 7. Output validation
        validator = self._get_output_validator()
        validation_result = validator.validate(response_text, top_chunks)

        timings["total_ms"] = (time.perf_counter() - total_t0) * 1000

        result = QueryResponse(
            request_id=request_id,
            response_text=response_text,
            retrieved_chunks=top_chunks,
            timings=PipelineTimings(**timings),
            safety=safety_result,
            output_validation=validation_result,
        )

        # 8. Log
        self._get_obs_logger().log_request(result, query_text)

        return result
