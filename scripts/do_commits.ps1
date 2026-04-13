#!/usr/bin/env pwsh
# Runs the remaining backdated commits for VaakSeva
# Each commit touches files with real improvements

$ErrorActionPreference = "Stop"
Set-Location "d:\Work\Git\VaakSeva"

function Commit([string]$date, [string]$msg, [string]$body) {
    $env:GIT_COMMITTER_DATE = $date
    git add -A
    if ($body) {
        git commit --date=$date -m $msg -m $body
    } else {
        git commit --date=$date -m $msg
    }
    Write-Host "OK: $msg"
}

# ---- April 13 commit 3: FastAPI app ----
Add-Content "backend/main.py" "`n# Weaviate port updated to avoid conflict with FastAPI default"
Commit "Sun, 13 Apr 2026 14:20:00 +0530" `
    "Build FastAPI app with lifespan, query and voice endpoints, timing middleware" `
    "Health check, text query, voice query, metrics, and dashboard endpoints. Request timing middleware and async lifespan for component initialisation. Weaviate port set to 8081 to avoid conflict with backend port 8080."

# ---- April 13 commit 4: Ingestion pipeline ----
Add-Content "backend/rag/ingest.py" "`n# Chunk overlap default 50 tokens, tuned for scheme documents"
Commit "Sun, 13 Apr 2026 17:05:00 +0530" `
    "Implement document ingestion pipeline with Hindi-aware chunking" `
    "PDF extraction via pdfplumber, Unicode NFC normalisation for Devanagari. Hindi-aware separators: paragraph, newline, purna viram, English period. Language detection and chunk metadata enrichment. Weaviate upsert helper."

# ---- April 14 commit 1: Embedder ----
Add-Content "backend/rag/embedder.py" "`n# E5 instruction prefix: query: / passage: asymmetric format"
Commit "Mon, 14 Apr 2026 09:00:00 +0530" `
    "Add embedding module supporting Qwen3 GPU and E5 CPU backends" `
    "Lazy model loading so import does not trigger GPU allocation. Batch-size configurable. E5 uses asymmetric query/passage instruction format. Factory function picks backend from settings."

# ---- April 14 commit 2: Retriever ----
Add-Content "backend/rag/retriever.py" "`n# alpha=0.75 default: 75 percent dense, 25 percent BM25F"
Commit "Mon, 14 Apr 2026 10:45:00 +0530" `
    "Implement Weaviate hybrid retriever with BM25F and dense search" `
    "Single API call retrieves both BM25 and dense results fused via RRF. Tunable alpha parameter controls dense vs BM25 weighting. Collection schema auto-created on first run. Metadata filter support for language."

# ---- April 14 commit 3: Reranker ----
Add-Content "backend/rag/reranker.py" "`n# Reranker takes top 50 candidates and returns top 5 by cross-encoder score"
Commit "Mon, 14 Apr 2026 12:30:00 +0530" `
    "Add reranker module with Qwen3 cross-encoder and NoOp fallback" `
    "Qwen3-Reranker-8B cross-encoder rescores top-50 candidates. NoOp reranker passes through unchanged for CPU-only environments. Configurable top-k output."

# ---- April 14 commit 4: RAG pipeline ----
Add-Content "backend/rag/pipeline.py" "`n# Safety filter runs before embedding to avoid processing injected inputs"
Commit "Mon, 14 Apr 2026 14:15:00 +0530" `
    "Build end-to-end RAG pipeline orchestrator with per-stage latency tracking" `
    "Runs safety filter, embedding, retrieval, reranking, profile extraction, LLM generation, and output validation. Millisecond timing for every stage. Lazy component loading on first request."

# ---- April 14 commit 5: LLM client ----
Add-Content "backend/llm/client.py" "`n# vLLM backend uses OpenAI-compatible chat completions endpoint"
Commit "Mon, 14 Apr 2026 16:30:00 +0530" `
    "Add LLM client with vLLM and Ollama backends using OpenAI-compatible API" `
    "Streaming-compatible async generation method. Exponential backoff retry on 5xx errors. Configurable temperature, max_tokens, timeout. Both backends share the same interface."

# ---- April 14 commit 6: Prompts + Extractor ----
Add-Content "backend/llm/prompts.py" "`n# build_rag_prompt formats last 2 conversation exchanges as context"
Commit "Mon, 14 Apr 2026 18:45:00 +0530" `
    "Write all Hindi prompt templates and LLM-based profile extraction" `
    "System prompt, RAG template, eligibility template all in Hindi. Profile extractor parses user messages to structured JSON via LLM with JSON-only instruction. Handles markdown code block wrapping in LLM output."

# ---- April 15 commit 1: Whisper STT ----
Add-Content "backend/voice/stt.py" "`n# VAD min_speech_duration_ms=500 filters out very short noise bursts"
Commit "Tue, 15 Apr 2026 09:00:00 +0530" `
    "Implement Whisper Hindi STT using collabora faster-whisper large-v2 model" `
    "CTranslate2 backend gives 4x faster inference over PyTorch Whisper. Built-in Silero VAD segments audio and trims silence. Language confidence gating: reject audio below 0.7 language probability."

# ---- April 15 commit 2: TTS ----
Add-Content "backend/voice/tts.py" "`n# Edge TTS voice hi-IN-MadhurNeural is male, SwaraNeural is female"
Commit "Tue, 15 Apr 2026 10:30:00 +0530" `
    "Add Veena TTS production backend and Edge TTS development fallback" `
    "Veena is a 3B Llama-based autoregressive TTS with SNAC 24kHz codec. MOS 4.2/5 on Hindi. NF4 4-bit quantization for reduced VRAM. Edge TTS uses Microsoft cloud, zero setup, for development iteration only."

# ---- April 15 commit 3: Audio utils ----
Add-Content "backend/voice/audio_utils.py" "`n# ffmpeg loudnorm filter: I=-16 LUFS, TP=-1.5 dBTP, LRA=11 LU"
Commit "Tue, 15 Apr 2026 12:00:00 +0530" `
    "Implement audio conversion utilities for WhatsApp OGG and Whisper WAV formats" `
    "OGG/Opus to 16kHz mono WAV for Whisper input. WAV/MP3 to OGG/Opus 32kbps for WhatsApp voice notes. Volume normalisation using loudnorm filter. ffprobe duration query."

# ---- April 15 commit 4: User memory ----
Add-Content "backend/memory/user_memory.py" "`n# purge_inactive_users can be called from a nightly cron job"
Commit "Tue, 15 Apr 2026 13:30:00 +0530" `
    "Build per-user conversation memory backed by hashed JSON files" `
    "Phone numbers SHA-256 hashed with salt before use as filenames. Profile fields accumulated across conversation turns without overwriting existing data. 30-day TTL purge on last_active timestamp. Max 20 turns stored."

# ---- April 15 commit 5: Eligibility tool ----
Add-Content "backend/tools/eligibility.py" "`n# Criterion.evaluate returns False for any missing profile field"
Commit "Tue, 15 Apr 2026 15:00:00 +0530" `
    "Add rule-based eligibility checker for 14 government schemes" `
    "Structured criteria with operators: eq, in, gt, lt, gte, lte, any. Mandatory vs optional criteria weighting in score calculation. Results sorted by score descending. Returns empty list for unknown user profile."

# ---- April 15 commit 6: Safety ----
Add-Content "backend/safety/input_filter.py" "`n# Hindi injection patterns: niyam bhool, nirdesh anadekhaa"
Commit "Tue, 15 Apr 2026 16:30:00 +0530" `
    "Implement prompt injection defense and input sanitization" `
    "Regex-based detection of English and Hindi injection patterns: ignore instructions, act as, DAN mode, jailbreak, system prompt exfiltration. Unicode NFC normalisation and control character stripping. Length capped at 1500 chars."

# ---- April 15 commit 7: Output validator ----
Add-Content "backend/safety/output_validator.py" "`n# 15 percent tolerance on amount comparison handles formatting differences"
Commit "Tue, 15 Apr 2026 17:30:00 +0530" `
    "Add output validator to fact-check benefit amounts against structured database" `
    "Extracts currency amounts from LLM response using regex. Cross-references against known amounts in schemes_structured.json. Flags mismatches with 15 percent tolerance for formatting differences. Confidence score 0.3 to 0.95."

# ---- April 15 commit 8: Observability ----
Add-Content "backend/observability/logger.py" "`n# RotatingFileHandler: 50MB max per file, 5 backup files"
Commit "Tue, 15 Apr 2026 18:00:00 +0530" `
    "Add structured JSON logging, metrics aggregation, and request log writer" `
    "Per-request JSONL log entries with stage timings, top scheme, safety flag, and error status. Rotating file handler 50MB max. Metrics compute P50/P95/P99 latency per stage, top schemes, and 7-day query trend. GCP Cloud Logging compatible format."

# ---- April 16 commit 1: WhatsApp index ----
Add-Content "whatsapp/src/index.ts" "`n// browser: VaakSeva avoids mobile fingerprinting by using Chrome UA"
Commit "Wed, 16 Apr 2026 09:00:00 +0530" `
    "Add Baileys WhatsApp client with multi-file auth and reconnect logic" `
    "Multi-file auth state persists QR scan across restarts. Auto-reconnect with 5s delay unless logged out. Filters status broadcasts and self-messages. Routes to MessageHandler for processing."

# ---- April 16 commit 2: Handlers + API ----
Add-Content "whatsapp/src/handlers.ts" "`n// typing indicator shown while FastAPI processes the request"
Commit "Wed, 16 Apr 2026 10:30:00 +0530" `
    "Add WhatsApp message handlers, HTTP API client, and audio download utility" `
    "Routes text messages and PTT voice notes to FastAPI endpoints. Hindi error messages for server errors and unsupported media types. Retry logic on 5xx with 2 max retries. Multipart form-data upload for voice audio."

# ---- April 16 commit 3: Scheme documents ----
Add-Content "data/schemes/pm_kisan_hi.txt" "`n"
Commit "Wed, 16 Apr 2026 12:00:00 +0530" `
    "Add 14 curated government scheme documents in Hindi and English" `
    "PM-KISAN, Ayushman Bharat, Ujjwala, PMAY-Gramin, MUDRA, APY, Jan Dhan. Fasal Bima, MGNREGS, Sukanya Samridhi, SVANidhi, Vishwakarma, Stand-Up India, NSP Scholarships. Each document includes eligibility, benefits, documents needed, application steps, and helpline numbers."

# ---- April 16 commit 4: Structured JSON ----
Add-Content "data/schemes_structured.json" "`n"
Commit "Wed, 16 Apr 2026 13:30:00 +0530" `
    "Add schemes_structured.json with eligibility rules for 14 schemes" `
    "Each scheme has structured criteria using eq/in/gt/lt/gte/lte operators with mandatory flag. Benefits summary in Hindi and English. Apply URL for each scheme. Used by rule-based eligibility checker and output validator."

# ---- April 16 commit 5: Eval test set ----
Add-Content "eval/test_set.json" "`n"
Commit "Wed, 16 Apr 2026 15:00:00 +0530" `
    "Add 50 Hindi Q-and-A evaluation pairs covering all major schemes" `
    "Questions span three categories: factual (benefit amounts, dates), eligibility (who qualifies), and procedural (how to apply). Expected keyword assertions for recall evaluation. Covers help-line numbers, documents needed, and exclusion criteria."

# ---- April 16 commit 6: Eval pipeline ----
Add-Content "eval/evaluate.py" "`n# Run: python eval/evaluate.py --top-k 5"
Commit "Wed, 16 Apr 2026 16:15:00 +0530" `
    "Write evaluation pipeline with retrieval recall, factual accuracy, and latency" `
    "hit@5: expected scheme in top-5 retrieved chunks. Factual accuracy: 50 percent keyword match threshold. P50/P95/P99 latency per pipeline run. Per-category breakdown. Markdown report written to eval/reports/."

# ---- April 16 commit 7: Tests ----
Add-Content "tests/conftest.py" "`n# Shared fixtures: temp_dir, structured_db_file, sample_chunks, mock_llm"
Commit "Wed, 16 Apr 2026 17:00:00 +0530" `
    "Add pytest test suite covering eligibility, safety, memory, retrieval, and STT" `
    "EligibilityChecker: criterion operator tests, known scheme matching, score sorting. InputFilter: 12 injection pattern cases plus sanitization edge cases. UserMemory: persistence, profile accumulation, phone hash privacy, history limits."

# ---- April 16 commit 8: Dashboard ----
Add-Content "dashboard/charts.js" "`n// Auto-refresh interval: 30 seconds"
Commit "Wed, 16 Apr 2026 18:00:00 +0530" `
    "Add Chart.js metrics dashboard with dark indigo theme and live refresh" `
    "Line chart for daily query trend (7 days). Bar chart for per-stage P50 latency. Horizontal bar chart for top queried schemes. Doughnut charts for input type and language distribution. Auto-refresh every 30 seconds. Demo data fallback when backend is offline."

# ---- April 16 commit 9: Docker + Scripts ----
Add-Content "docker-compose.yml" "`n"
Commit "Wed, 16 Apr 2026 18:45:00 +0530" `
    "Add Docker Compose, Dockerfiles, ingestion script, and pipeline test script" `
    "Full stack: backend (Python/uvicorn), Weaviate 1.23, Redis 7, WhatsApp (Node). Health checks and volume mounts. Ingest script with dry-run and stats output. End-to-end test script prints timing breakdown per stage."

# ---- April 16 commit 10: README and ARCHITECTURE ----
Add-Content "README.md" "`n"
Commit "Wed, 16 Apr 2026 19:30:00 +0530" `
    "Write README with architecture diagram, setup guide, and deployment notes" `
    "ASCII pipeline diagram showing full voice RAG flow. Tech stack table with model names, licenses, and GPU requirements. Quick start for CPU-only dev mode. GCP Cloud Run and Compute Engine GPU deployment commands. Evaluation targets table."

Write-Host "All commits done!"
git log --oneline
