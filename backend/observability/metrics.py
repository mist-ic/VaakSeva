"""
Metrics aggregation for VaakSeva dashboard.

Reads structured JSONL request logs and computes:
  - Total queries (today, this week, all time)
  - Average latency breakdown by pipeline stage
  - P50/P95/P99 latencies
  - Top queried schemes
  - Voice vs text query ratio
  - Error rate
  - Language detection distribution

Results are served by the /api/metrics endpoint and consumed by
the Chart.js dashboard.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, quantiles

logger = logging.getLogger(__name__)


def _load_requests(log_dir: Path) -> list[dict]:
    """Load all request log entries from the JSONL file."""
    log_file = log_dir / "requests.jsonl"
    if not log_file.exists():
        return []

    entries = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def compute_metrics(log_dir: Path) -> dict:
    """
    Read all request logs and compute dashboard metrics.

    Returns a dict ready to be serialised as JSON for the dashboard.
    """
    entries = _load_requests(log_dir)

    if not entries:
        return _empty_metrics()

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    total = len(entries)
    today_count = 0
    week_count = 0

    latencies: dict[str, list[float]] = defaultdict(list)
    scheme_counter: Counter = Counter()
    input_type_counter: Counter = Counter()
    language_counter: Counter = Counter()
    error_count = 0
    daily_counts: dict[str, int] = defaultdict(int)

    for entry in entries:
        # Parse timestamp
        try:
            ts = datetime.fromisoformat(entry.get("timestamp", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            ts = None

        if ts:
            if ts >= today_start:
                today_count += 1
            if ts >= week_start:
                week_count += 1

            day_key = ts.strftime("%Y-%m-%d")
            daily_counts[day_key] += 1

        # Latencies
        timings = entry.get("pipeline_timings", {})
        for stage in ["stt_ms", "embedding_ms", "retrieval_ms", "rerank_ms", "llm_ms", "tts_ms", "total_ms"]:
            val = timings.get(stage)
            if val is not None:
                latencies[stage].append(float(val))

        # Top schemes
        retrieval = entry.get("retrieval", {})
        top_scheme = retrieval.get("top_scheme")
        if top_scheme:
            scheme_counter[top_scheme] += 1

        # Input type
        input_type = entry.get("input_type", "text")
        input_type_counter[input_type] += 1

        # Language
        lang = entry.get("language_detected")
        if lang:
            language_counter[lang] += 1

        # Errors
        if entry.get("error"):
            error_count += 1

    # Compute percentile latencies
    latency_stats = {}
    for stage, values in latencies.items():
        if values:
            sorted_vals = sorted(values)
            latency_stats[stage] = {
                "p50": round(_percentile(sorted_vals, 50), 1),
                "p95": round(_percentile(sorted_vals, 95), 1),
                "p99": round(_percentile(sorted_vals, 99), 1),
                "mean": round(sum(values) / len(values), 1),
            }

    # Daily query trend (last 7 days)
    daily_trend = []
    for i in range(6, -1, -1):
        day = (today_start - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_trend.append({"date": day, "count": daily_counts.get(day, 0)})

    return {
        "summary": {
            "total": total,
            "today": today_count,
            "week": week_count,
            "error_count": error_count,
            "error_rate": round(error_count / max(total, 1) * 100, 2),
        },
        "latency_stats": latency_stats,
        "top_schemes": [
            {"scheme": s, "count": c}
            for s, c in scheme_counter.most_common(10)
        ],
        "input_types": dict(input_type_counter),
        "languages": dict(language_counter),
        "daily_trend": daily_trend,
    }


def _percentile(sorted_values: list[float], p: int) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _empty_metrics() -> dict:
    return {
        "summary": {"total": 0, "today": 0, "week": 0, "error_count": 0, "error_rate": 0.0},
        "latency_stats": {},
        "top_schemes": [],
        "input_types": {},
        "languages": {},
        "daily_trend": [],
    }
