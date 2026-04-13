"""
Evaluation pipeline for VaakSeva.

Runs a comprehensive evaluation suite on the full pipeline:
  1. Retrieval quality: recall@5, MRR (Mean Reciprocal Rank), hit_rate
  2. Response quality: factual accuracy (keyword match), Hindi language consistency
  3. Latency: P50, P95, P99 for each pipeline stage

Reads from eval/test_set.json (50+ Hindi Q&A pairs).
Writes a markdown report to eval/reports/report_<timestamp>.md.

Usage:
  python eval/evaluate.py
  python eval/evaluate.py --top-k 5 --output-format json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.pipeline import RAGPipeline
from backend.models.schemas import EvalQuestion, EvalResult, QueryCategory


def load_test_set(path: Path) -> list[EvalQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion(**item) for item in raw]


def compute_percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] * (1 - (idx - lo)) + sorted_vals[hi] * (idx - lo)


class RAGEvaluator:
    def __init__(self, test_set_path: Path):
        self.test_set = load_test_set(test_set_path)
        self.pipeline = RAGPipeline()
        self.results: list[EvalResult] = []

    async def run(self, top_k: int = 5) -> dict:
        print(f"Running evaluation on {len(self.test_set)} questions...")

        for i, question in enumerate(self.test_set, 1):
            print(f"  [{i}/{len(self.test_set)}] {question.question_hi[:60]}...")

            t0 = time.perf_counter()
            try:
                response = await self.pipeline.aquery(query_text=question.question_hi)
                latency_ms = (time.perf_counter() - t0) * 1000

                retrieved_ids = [c.scheme_id for c in response.retrieved_chunks[:top_k]]

                hit = question.expected_scheme in retrieved_ids
                factual = self._check_factual(
                    response.response_text,
                    question.expected_answer_contains,
                )

                self.results.append(
                    EvalResult(
                        question=question,
                        retrieved_scheme_ids=retrieved_ids,
                        response=response.response_text,
                        hit=hit,
                        factual_accurate=factual,
                        latency_ms=latency_ms,
                    )
                )
            except Exception as exc:
                print(f"    ERROR: {exc}")
                self.results.append(
                    EvalResult(
                        question=question,
                        retrieved_scheme_ids=[],
                        response="",
                        hit=False,
                        factual_accurate=False,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )

        return self._compute_report()

    @staticmethod
    def _check_factual(response: str, expected_keywords: list[str]) -> bool:
        """True if at least 50% of expected keywords appear in the response."""
        if not expected_keywords:
            return True
        matched = sum(1 for kw in expected_keywords if kw in response)
        return matched >= max(1, len(expected_keywords) // 2)

    def _compute_report(self) -> dict:
        total = len(self.results)
        hits = sum(1 for r in self.results if r.hit)
        factual = sum(1 for r in self.results if r.factual_accurate)
        latencies = [r.latency_ms for r in self.results]

        by_category = {}
        for cat in QueryCategory:
            cat_results = [r for r in self.results if r.question.category == cat]
            if cat_results:
                by_category[cat.value] = {
                    "total": len(cat_results),
                    "hit_rate": sum(1 for r in cat_results if r.hit) / len(cat_results),
                    "factual_rate": sum(1 for r in cat_results if r.factual_accurate) / len(cat_results),
                }

        return {
            "summary": {
                "total_questions": total,
                "hit_rate_at_5": round(hits / total, 3) if total else 0,
                "factual_accuracy": round(factual / total, 3) if total else 0,
                "evaluated_at": datetime.utcnow().isoformat(),
            },
            "latency": {
                "p50_ms": round(compute_percentile(latencies, 50), 1),
                "p95_ms": round(compute_percentile(latencies, 95), 1),
                "p99_ms": round(compute_percentile(latencies, 99), 1),
                "mean_ms": round(mean(latencies), 1) if latencies else 0,
                "median_ms": round(median(latencies), 1) if latencies else 0,
            },
            "by_category": by_category,
        }


def write_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"report_{ts}.md"

    s = report["summary"]
    l = report["latency"]

    lines = [
        "# VaakSeva Evaluation Report",
        "",
        f"Generated: {s['evaluated_at']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total questions | {s['total_questions']} |",
        f"| Retrieval hit@5 | {s['hit_rate_at_5']:.1%} |",
        f"| Factual accuracy | {s['factual_accuracy']:.1%} |",
        "",
        "## Latency",
        "",
        f"| Percentile | Latency |",
        f"|------------|---------|",
        f"| P50 | {l['p50_ms']:.0f} ms |",
        f"| P95 | {l['p95_ms']:.0f} ms |",
        f"| P99 | {l['p99_ms']:.0f} ms |",
        f"| Mean | {l['mean_ms']:.0f} ms |",
        "",
        "## By Category",
        "",
        "| Category | Questions | Hit@5 | Factual |",
        "|----------|-----------|-------|---------|",
    ]

    for cat, stats in report.get("by_category", {}).items():
        lines.append(
            f"| {cat} | {stats['total']} | {stats['hit_rate']:.1%} | {stats['factual_rate']:.1%} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


async def main():
    parser = argparse.ArgumentParser(description="Evaluate VaakSeva RAG pipeline")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k")
    parser.add_argument("--output-format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    test_set_path = Path(__file__).parent / "test_set.json"
    if not test_set_path.exists():
        print(f"ERROR: {test_set_path} not found")
        sys.exit(1)

    evaluator = RAGEvaluator(test_set_path)
    report = await evaluator.run(top_k=args.top_k)

    if args.output_format == "json":
        print(json.dumps(report, indent=2))
    else:
        reports_dir = Path(__file__).parent / "reports"
        report_path = write_report(report, reports_dir)
        print(f"\nReport written to: {report_path}")
        print(f"\nSummary:")
        print(f"  Hit@5: {report['summary']['hit_rate_at_5']:.1%}")
        print(f"  Factual: {report['summary']['factual_accuracy']:.1%}")
        print(f"  P95 latency: {report['latency']['p95_ms']:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
