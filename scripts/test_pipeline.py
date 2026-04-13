"""
End-to-end pipeline test script.

Runs a sample query through the full pipeline and prints timing breakdown.

Usage:
  python scripts/test_pipeline.py
  python scripts/test_pipeline.py --query "PM Kisan योजना में कितने पैसे मिलते हैं?"
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_QUERIES = [
    "PM Kisan योजना में कितने पैसे मिलते हैं?",
    "मैं 25 साल का किसान हूं, कर्नाटक से हूं, कौन सी योजनाएं मिल सकती हैं?",
    "आयुष्मान भारत कार्ड कैसे बनाएं?",
    "मनरेगा में जॉब कार्ड के लिए क्या करना होगा?",
]


async def run_test(query: str):
    from backend.rag.pipeline import RAGPipeline

    print(f"\nQuery: {query}")
    print("-" * 60)

    pipeline = RAGPipeline()

    t0 = time.perf_counter()
    result = await pipeline.aquery(query_text=query)
    total_ms = (time.perf_counter() - t0) * 1000

    print(f"\nResponse:\n{result.response_text}")
    print(f"\nTop retrieved schemes:")
    for i, chunk in enumerate(result.retrieved_chunks[:3], 1):
        print(f"  {i}. {chunk.scheme_name} (score={chunk.score:.3f})")

    print(f"\nTimings:")
    t = result.timings
    if t.embedding_ms:
        print(f"  Embedding:  {t.embedding_ms:.0f} ms")
    if t.retrieval_ms:
        print(f"  Retrieval:  {t.retrieval_ms:.0f} ms")
    if t.rerank_ms:
        print(f"  Reranking:  {t.rerank_ms:.0f} ms")
    if t.llm_ms:
        print(f"  LLM:        {t.llm_ms:.0f} ms")
    print(f"  Total:      {total_ms:.0f} ms")

    if result.safety and not result.safety.is_safe:
        print(f"\n[Safety] Blocked: {result.safety.flagged_patterns}")

    if result.output_validation and not result.output_validation.is_valid:
        print(f"\n[Validation] Flagged claims: {result.output_validation.flagged_claims}")


async def main():
    parser = argparse.ArgumentParser(description="Test VaakSeva pipeline")
    parser.add_argument("--query", type=str, help="Custom query to test")
    args = parser.parse_args()

    if args.query:
        await run_test(args.query)
    else:
        print("Running tests with sample queries...")
        for query in SAMPLE_QUERIES[:2]:  # 2 sample queries by default
            await run_test(query)


if __name__ == "__main__":
    asyncio.run(main())
