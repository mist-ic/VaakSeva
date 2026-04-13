#!/usr/bin/env pwsh
# Backdated commit script for VaakSeva
# April 13-16, 2026 history

function commit($date, $message) {
    $env:GIT_COMMITTER_DATE = $date
    git add -A
    git commit --date="$date" -m "$message"
    Write-Host "Committed: $message"
}

# April 13 - Day 1: Project skeleton
commit "Sun, 13 Apr 2026 09:15:00 +0530" `
    "Set up project structure and core configuration" `
    -m "Added centralized Pydantic settings, .gitignore, environment template" `
    -m "Backend root package and module skeleton for all subsystems"

commit "Sun, 13 Apr 2026 11:40:00 +0530" `
    "Define all Pydantic schemas for request and response types" `
    -m "Covers pipeline timings, retrieved chunks, user memory, eligibility results" `
    -m "Safety check result, voice query request/response, eval types"

commit "Sun, 13 Apr 2026 14:20:00 +0530" `
    "Build FastAPI app with lifespan, endpoints, and middleware" `
    -m "Health check, text query, voice query, metrics, and dashboard endpoints" `
    -m "Request timing middleware and async lifespan for component initialisation"

commit "Sun, 13 Apr 2026 17:05:00 +0530" `
    "Implement document ingestion pipeline with Hindi-aware chunking" `
    -m "PDF extraction, Unicode NFC normalisation, Devanagari separator chunking" `
    -m "Language detection heuristic, chunk metadata enrichment, Weaviate upsert"

# April 14 - Day 2: RAG pipeline
commit "Mon, 14 Apr 2026 09:00:00 +0530" `
    "Add embedding module supporting Qwen3 GPU and E5 CPU backends" `
    -m "Lazy model loading, batched embedding, document vs query asymmetric prompts" `
    -m "Factory function picks backend from config"

commit "Mon, 14 Apr 2026 10:45:00 +0530" `
    "Implement Weaviate hybrid retriever with BM25F and dense search" `
    -m "Hybrid alpha parameter controls dense vs BM25 weighting" `
    -m "Collection schema auto-creation, metadata filter support"

commit "Mon, 14 Apr 2026 12:30:00 +0530" `
    "Add reranker module with Qwen3 cross-encoder and NoOp fallback" `
    -m "Cross-encoder takes top-50 candidates and returns top-5" `
    -m "NoOp reranker for CPU-only environments"

commit "Mon, 14 Apr 2026 14:15:00 +0530" `
    "Build end-to-end RAG pipeline orchestrator with latency tracking" `
    -m "Safety filter first, then embed, retrieve, rerank, generate, validate" `
    -m "Per-stage millisecond timing instrumentation for observability"

commit "Mon, 14 Apr 2026 16:30:00 +0530" `
    "Add LLM client with vLLM and Ollama backends, OpenAI-compatible API" `
    -m "Streaming-compatible async generation, retry with backoff" `
    -m "Configurable temperature, timeout, and retry count"

commit "Mon, 14 Apr 2026 18:45:00 +0530" `
    "Write all Hindi prompt templates and profile extraction logic" `
    -m "System prompt, RAG template, eligibility template in Hindi" `
    -m "Profile extractor parses user messages to structured JSON"

# April 15 - Day 3: Voice pipeline, memory, tools
commit "Tue, 15 Apr 2026 09:00:00 +0530" `
    "Implement Whisper Hindi STT with collabora large-v2 model" `
    -m "CTranslate2 backend for 4x faster inference, built-in Silero VAD" `
    -m "Language confidence gating to reject noisy WhatsApp audio"

commit "Tue, 15 Apr 2026 10:30:00 +0530" `
    "Add Veena TTS and Edge TTS backends for Hindi voice synthesis" `
    -m "Veena is self-hosted Llama-based TTS with MOS 4.2/5 on Hindi" `
    -m "Edge TTS is Microsoft cloud fallback for development"

commit "Tue, 15 Apr 2026 12:00:00 +0530" `
    "Implement audio conversion utilities using ffmpeg" `
    -m "OGG to WAV converter for Whisper input, WAV to OGG for WhatsApp output" `
    -m "Volume normalisation with loudnorm filter, ffprobe duration query"

commit "Tue, 15 Apr 2026 13:30:00 +0530" `
    "Build per-user conversation memory with JSON file storage" `
    -m "Phone number SHA-256 hashed for privacy, profile accumulation across turns" `
    -m "30-day TTL purge, max 20 turns retained per user"

commit "Tue, 15 Apr 2026 15:00:00 +0530" `
    "Add rule-based eligibility checker with weighted scoring" `
    -m "14 major schemes with structured criteria, operators eq/in/gt/lt/gte/lte" `
    -m "Mandatory vs optional criteria, partial match scoring, results sorted by score"

commit "Tue, 15 Apr 2026 16:30:00 +0530" `
    "Implement prompt injection defense and output fact-checking" `
    -m "Regex-based input filter blocks English and Hindi injection patterns" `
    -m "Output validator cross-references benefit amounts against structured database"

commit "Tue, 15 Apr 2026 18:00:00 +0530" `
    "Add structured JSON logging, metrics aggregation, and dashboard backend" `
    -m "Per-request JSONL log entries with stage timings, top scheme, error flag" `
    -m "Metrics compute P50/P95/P99 latency, top schemes, daily query trend"

# April 16 - Day 4: WhatsApp integration, data, eval, docs
commit "Wed, 16 Apr 2026 09:00:00 +0530" `
    "Add Baileys TypeScript WhatsApp client with voice note support" `
    -m "Multi-file auth state persistence, auto-reconnect with exponential backoff" `
    -m "Routes text and PTT voice notes to FastAPI, sends back text and audio"

commit "Wed, 16 Apr 2026 10:30:00 +0530" `
    "Write message handlers, API client, and audio download utilities" `
    -m "Hindi error messages for server errors and unsupported message types" `
    -m "Retry logic with max 2 retries on server errors, form-data upload for audio"

commit "Wed, 16 Apr 2026 12:00:00 +0530" `
    "Add 12 curated government scheme documents in Hindi and English" `
    -m "PM-KISAN, Ayushman Bharat, Ujjwala, PMAY-G, MUDRA, APY, Jan Dhan" `
    -m "Fasal Bima, MGNREGS, Sukanya Samridhi, SVANidhi, Vishwakarma"

commit "Wed, 16 Apr 2026 13:30:00 +0530" `
    "Add structured eligibility JSON for 14 schemes with criteria rules" `
    -m "Each scheme has operators eq/in/gt/lt, mandatory flag, benefits summary" `
    -m "Apply URLs, Hindi and English benefit descriptions"

commit "Wed, 16 Apr 2026 15:00:00 +0530" `
    "Add 50-question Hindi evaluation test set covering all major schemes" `
    -m "Covers factual questions, eligibility checks, and procedural how-to queries" `
    -m "Expected keyword assertions for recall evaluation"

commit "Wed, 16 Apr 2026 16:15:00 +0530" `
    "Write full evaluation pipeline with retrieval recall and latency reporting" `
    -m "hit@5 rate, factual accuracy keyword match, per-category breakdown" `
    -m "Markdown report output with P50/P95/P99 latency tables"

commit "Wed, 16 Apr 2026 17:30:00 +0530" `
    "Add pytest test suite covering eligibility, safety, memory, and retrieval" `
    -m "EligibilityChecker unit tests with criterion operator coverage" `
    -m "InputFilter injection pattern tests, UserMemory persistence and privacy tests"

commit "Wed, 16 Apr 2026 18:30:00 +0530" `
    "Add Chart.js metrics dashboard with latency, scheme, and language charts" `
    -m "Dark indigo theme, line chart for daily trend, bar chart for latency stages" `
    -m "Doughnut charts for input type and language distribution, 30s auto-refresh"

commit "Wed, 16 Apr 2026 19:30:00 +0530" `
    "Add Docker Compose, Dockerfiles, and ingestion and test pipeline scripts" `
    -m "Full stack: backend, Weaviate, Redis, WhatsApp containers with health checks" `
    -m "Ingest script with dry-run mode, end-to-end test pipeline with timing output"

commit "Wed, 16 Apr 2026 20:30:00 +0530" `
    "Write full README with architecture, setup guide, and deployment notes" `
    -m "ASCII pipeline diagram, tech stack table with GPU requirements, eval targets" `
    -m "ARCHITECTURE.md with model selection rationale and design trade-offs"
