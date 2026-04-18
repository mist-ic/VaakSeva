"""
Reranker module for VaakSeva.

After hybrid retrieval returns top-20 candidates, the reranker re-scores them
with a cross-encoder for much higher precision before sending to the LLM.

Default backend: Qwen3-Reranker-0.6B (enabled by default)
  - Fast on CPU (~200-400ms for top-20 candidates)
  - Still outperforms no-reranking on MTEB multilingual reranking benchmarks
  - Supports 100+ languages including Hindi

Implementation note:
  Qwen3-Reranker-0.6B is a CausalLM (text-generation), NOT a seq-classifier.
  The correct scoring approach is:
    - Feed (query, passage) as a prompt
    - Look at the logit for token '1' vs '0' at the final position
  We implement this directly with transformers (AutoModelForCausalLM)
  to avoid sentence-transformers version compatibility issues.

Why NOT Qwen3-Reranker-8B in production:
  - 8B cross-encoder on CPU adds 2-5 seconds per request
  - This single component would blow the 6-second end-to-end budget
  - 0.6B is the practical sweet spot for CPU deployment
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import torch

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
    Qwen3-Reranker-0.6B implemented directly via transformers.

    Qwen3-Reranker is a CausalLM (text-generation) that scores relevance by
    comparing the output logits for '1' (relevant) vs '0' (not relevant)
    at the final token position. This is NOT a sequence classifier.

    We implement scoring directly with AutoModelForCausalLM to avoid
    sentence-transformers version/platform compatibility issues.

    Prompt format expected by the model:
      <|im_start|>user\n{query}\n{passage}<|im_end|>\n<|im_start|>assistant\n

    CPU latency (0.6B, 20 candidates, batch_size=4): ~200-600ms
    """

    # Qwen3-Reranker chat template for (query, passage) pairs
    _TEMPLATE = (
        "<|im_start|>user\n"
        "Given a query and a passage, judge whether the passage is relevant "
        "to the query. Answer '1' if relevant, '0' if not.\n"
        "Query: {query}\nPassage: {passage}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_name or settings.qwen3_reranker_model
        device = device or settings.reranker_device

        logger.info("Loading Qwen3 reranker: %s on %s", model_name, device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        # Qwen3 has no pad token by default; use eos
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )
        self._model.eval()
        self._device = device

        # Cache token IDs for '1' and '0'
        self._true_id = self._tokenizer.convert_tokens_to_ids("1")
        self._false_id = self._tokenizer.convert_tokens_to_ids("0")

        logger.info(
            "Qwen3 reranker ready (model=%s, true=%d, false=%d)",
            model_name, self._true_id, self._false_id,
        )

    def _score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Score (query, passage) pairs. Returns relevance score per passage."""
        prompts = [
            self._TEMPLATE.format(query=query, passage=p)
            for p in passages
        ]

        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Last token logits for '1' vs '0'
        last_logits = outputs.logits[:, -1, :]  # (batch, vocab)
        true_logits = last_logits[:, self._true_id]   # (batch,)
        false_logits = last_logits[:, self._false_id]  # (batch,)

        # Softmax to get P(relevant) in [0, 1]
        stacked = torch.stack([false_logits, true_logits], dim=-1)  # (batch, 2)
        probs = torch.softmax(stacked, dim=-1)
        scores = probs[:, 1].tolist()  # P('1' = relevant)
        return scores

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], float]:
        if not chunks:
            return [], 0.0

        t0 = time.perf_counter()

        passages = [chunk.content for chunk in chunks]
        scores = self._score_pairs(query, passages)

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
