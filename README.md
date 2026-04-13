# VaakSeva

**Self-hosted Hindi voice RAG assistant over WhatsApp for government scheme discovery. Zero commercial API dependencies. Powered by India-made sovereign AI (Sarvam-30B).**

"Vaak" = Sanskrit for speech. "Seva" = service.

---

## The Problem

400+ million Indians qualify for government welfare schemes but never claim them. The barriers are real:

- Information is fragmented across dozens of ministry websites
- Most official documents are in English
- Applying requires internet literacy most beneficiaries don't have
- No conversational interface that understands how people actually talk

A farmer in Rajasthan says "mujhe kya milega" (what can I get) into WhatsApp. He should get an answer. VaakSeva is that answer.

---

## What It Does

A WhatsApp bot that accepts Hindi text and voice input, retrieves grounded information about government schemes from a curated knowledge base, and responds in Hindi text and voice.

```
User sends: "मेरी उम्र 25 साल है, मैं किसान हूं, कौन सी सरकारी योजनाएं मिल सकती हैं?"

Pipeline:
  Embed query (Qwen3-Embedding-8B)
    |
  Hybrid retrieval (Weaviate: BM25F + dense, alpha=0.75)
    |
  Rerank (Qwen3-Reranker-8B, top-50 -> top-5)
    |
  Generate grounded Hindi response (Sarvam-30B via vLLM)
    |
  Bot sends: text + Hindi voice note (Veena TTS)
```

For voice input:
```
User sends: [60-second Hindi voice note]
  |
faster-whisper (collabora/whisper-large-v2-hindi, 5.33% WER on FLEURS)
  |
Same RAG pipeline
  |
Bot sends: text response + playable voice note (Veena TTS)
```

---

## Architecture

```
WhatsApp Web
     |
Baileys (Node.js, WebSocket)
     |
     +---> POST /api/query (text)
     |          |
     |    FastAPI (Python)
     |          |
     +---> POST /api/voice-query
            |
     [Safety Filter]  - Prompt injection defense
            |
     [Whisper STT]    - collabora Hindi, 5.33% WER  [voice only]
            |
     [Embedder]       - Qwen3-Embedding-8B / E5-large
            |
     [Weaviate]       - Hybrid BM25F + dense (alpha=0.75)
     50 candidates
            |
     [Reranker]       - Qwen3-Reranker-8B
     -> top 5 chunks
            |
     [User Memory]    - Profile + conversation history (JSON files)
            |
     [LLM]            - Sarvam-30B (vLLM, production)
                       Sarvam-M (Ollama, development)
            |
     [Output Validator] - Fact-check benefit amounts
            |
     [Veena TTS]      - maya-research/Veena (production)
     [Edge TTS]       - Microsoft (development fallback)
            |
     Baileys sends:
       - Text message
       - Voice note (OGG/Opus)
```

**Async design**: The heavy pipeline runs asynchronously. The WhatsApp webhook acknowledges immediately and processes in the background. Prevents timeout on WhatsApp's side under load.

**Observability**: Every request produces a structured JSON log entry with per-stage timing breakdown, retrieval metadata, and safety flag status.

---

## Tech Stack

| Component | Model / Tool | Source | Purpose | GPU |
|-----------|-------------|--------|---------|-----|
| LLM (prod) | Sarvam-30B (Q4_K_M) | sarvamai/sarvam-30b | Hindi-native MoE LLM, 2.4B active/token | 18-20 GB |
| LLM (dev) | Sarvam-M via Ollama | sarvamai/sarvam-m | Local dev, OpenAI-compat API | CPU |
| Embeddings | Qwen3-Embedding-8B | Qwen/Qwen3-Embedding-8B | #1 MTEB Multilingual, 70.58 | 4-6 GB |
| Embeddings (fallback) | multilingual-e5-large | intfloat/multilingual-e5-large-instruct | 560M, CPU-friendly | CPU |
| Reranker | Qwen3-Reranker-8B | Qwen/Qwen3-Reranker-8B | Cross-encoder, top-50->top-5 | 4-6 GB |
| Vector DB | Weaviate 1.23 | semitechnologies/weaviate | Native BM25F + dense hybrid | CPU |
| STT | faster-whisper large-v2-hindi | collabora/faster-whisper-large-v2-hindi | 5.33% WER on Hindi FLEURS | 1-3 GB |
| TTS (prod) | Veena 3B | maya-research/Veena | Hindi/English/Hinglish, MOS 4.2 | 2-3 GB |
| TTS (dev) | Edge TTS | Microsoft (edge-tts) | Free fallback, no GPU needed | None |
| WhatsApp | Baileys 6.x | @whiskeysockets/baileys | WebSocket to WhatsApp Web | None |
| Backend | FastAPI 0.115 | Python | Pipeline orchestrator | None |

### Why Sarvam-30B

Sarvam-30B is a Mixture-of-Experts model trained from scratch by Sarvam AI on 22 Indian languages. It activates only 2.4B parameters per token, giving it the compute cost of an 8B model while maintaining 30B-scale quality on Hindi. Apache 2.0. This is directly equivalent to what Puch AI uses internally.

Fallback: `Llama-3.1-8B-Instruct` (Q4_K_M, ~4.9GB) for maximum cost efficiency.

### Hybrid Retrieval Design

Why Weaviate over Qdrant or ChromaDB:

- **Weaviate**: BM25F computed server-side, single API call for hybrid search, metadata filtering built-in
- **Qdrant**: Requires computing sparse vectors client-side (more work, same result)
- **ChromaDB**: No native hybrid - would require implementing BM25 separately and manually fusing

```python
# Hybrid search with tunable alpha
results = collection.query.hybrid(
    query="PM Kisan क्या है",   # BM25F uses this
    vector=query_embedding,       # Dense uses this
    alpha=0.75,                   # 75% dense + 25% BM25
    limit=50,                     # Top-50 candidates for reranker
)
```

The `alpha=0.75` default works well for Hindi queries where semantic meaning matters more than exact keyword match. For scheme name lookups ("PM Kisan"), BM25 weight should increase (alpha=0.5).

---

## Evaluation Results

Run `python eval/evaluate.py` after ingesting documents.

Target benchmarks (achievable with full GPU stack):

| Metric | Target | Notes |
|--------|--------|-------|
| Retrieval hit@5 | >70% | Expected scheme in top-5 results |
| Factual accuracy | >60% | Response contains expected keywords |
| P50 total latency | <4s | Full pipeline end-to-end |
| P95 total latency | <7s | Under load |

STT evaluation uses Sarvam's LLM-as-judge framework ([sarvamai/llm_intent_entity](https://github.com/sarvamai/llm_intent_entity)):
- Intent Preservation Score (binary): does ASR output preserve meaning?
- Entity Preservation Score (0-1): fraction of named entities correctly transcribed
- Standard WER on Hindi FLEURS test set

---

## GPU Deployment

### Single A10G (24 GB) - Recommended

| Component | Memory | Notes |
|-----------|--------|-------|
| Sarvam-30B (Q4_K_M) via vLLM | 18-20 GB | MoE, 2.4B active tokens |
| Whisper large-v2-hindi (int8) | 1-2 GB | Shared time with LLM |
| Veena TTS (NF4) | 2-3 GB | Shared time with LLM |
| Weaviate + Redis | CPU | No GPU needed |
| Embeddings + Reranker | CPU | Acceptable latency on CPU |

**Cost estimate**: A10G on RunPod ~$0.60/hr, Lambda Labs ~$0.75/hr. Expected daily cost with moderate traffic: $5-10/day.

### Two GPUs (Full Stack)

- **GPU-0 (24 GB)**: Sarvam-30B via vLLM
- **GPU-1 (16 GB)**: Whisper + Qwen3-Embedding + Qwen3-Reranker + Veena

### Budget Option (Single T4 16 GB)

Swap to Sarvam-M (Q4, ~14GB) or Llama-3.1-8B (Q4, ~5GB). Everything else stays the same.

### vLLM vs Ollama

| | vLLM | Ollama |
|-|------|--------|
| Throughput (tok/s) | 80-150 | 20-40 |
| P99 latency | <5s | 8-15s |
| Concurrent requests | Yes | No |
| CPU support | No | Yes |
| Setup | Production server | `ollama pull` |
| Use case | Production | Local dev |

I tested both during development. For single-user dev: Ollama is fine. For anything with concurrent users: vLLM is necessary.

---

## Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- Docker and Docker Compose (for Weaviate and Redis)
- ffmpeg (`apt install ffmpeg` / `choco install ffmpeg` / `brew install ffmpeg`)
- For GPU stack: CUDA 12.1+, 16+ GB VRAM

### Quick Start (Dev Mode - No GPU Required)

```bash
# Clone
git clone https://github.com/mist-ic/VaakSeva.git
cd VaakSeva

# Python env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Config
cp .env.example .env
# Defaults use E5 embeddings, Ollama LLM, Edge TTS (all CPU/free)

# Start Weaviate and Redis
docker compose up weaviate redis -d

# Start Ollama (separate terminal)
ollama pull sarvam-m
ollama serve

# Ingest scheme documents (dry run first)
python scripts/ingest.py --dry-run
python scripts/ingest.py

# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload

# Start WhatsApp client (separate terminal)
cd whatsapp && npm install && npm run dev
# Scan QR code with your phone

# Test pipeline
python scripts/test_pipeline.py
```

### Production Deployment on GCP

```bash
# Authenticate
gcloud auth login
gcloud config set project revsight-492123

# Option 1: Cloud Run (CPU-only - E5 embeddings, Ollama external)
gcloud run deploy vaakseva-backend \
  --source . \
  --region asia-south1 \
  --port 8080 \
  --allow-unauthenticated \
  --set-env-vars LLM_BACKEND=ollama,EMBEDDER_BACKEND=e5,TTS_BACKEND=edge

# Option 2: Compute Engine GPU (full stack)
gcloud compute instances create vaakseva-gpu \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --zone=asia-south1-c

# SSH in and run docker compose
gcloud compute ssh vaakseva-gpu --zone=asia-south1-c
# Then run: docker compose up -d
```

For WhatsApp at production scale, replace Baileys with the official **WhatsApp Business Cloud API** (webhook-based, no session management, no QR codes):

```
Webhook -> Cloud Functions (GCP) -> Pub/Sub -> Workers -> Firestore + Cloud Run
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific modules
pytest tests/test_eligibility.py -v
pytest tests/test_safety.py -v
pytest tests/test_memory.py -v

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

---

## Dashboard

Navigate to `http://localhost:8080/dashboard` after starting the backend.

Shows:
- Daily query volume (last 7 days)
- Pipeline latency breakdown per stage (P50)
- Top queried government schemes
- Voice vs text ratio
- Hindi vs English distribution

---

## Safety

### Prompt Injection Defense

Pattern-based detection catches common injection attempts (English and Hindi):
- "ignore previous instructions"
- "act as", "pretend to be", "DAN mode", "jailbreak"
- Hindi equivalents: "नियम भूल", "निर्देश अनदेखा"

All inputs are sanitized (control characters stripped, NFC normalised) before reaching the LLM.

### Output Fact-Checking

The `OutputValidator` cross-references benefit amounts claimed in LLM responses against the `data/schemes_structured.json` ground truth. If the LLM claims PM-KISAN gives Rs 8,000 instead of Rs 6,000, it gets flagged.

---

## Conversation Memory

Multi-turn conversations are supported via per-user JSON state files:

```
Turn 1: "मैं कर्नाटक से हूं" -> stores state=Karnataka
Turn 2: "25 साल का किसान हूं" -> stores age=25, occupation=farmer
Turn 3: "क्या योजनाएं मिल सकती हैं?" -> uses accumulated profile for eligibility
```

User data is stored with SHA-256 hashed phone numbers (with salt). Files are purged after 30 days of inactivity.

---

## Limitations

- **Baileys is experimental**: Baileys uses WhatsApp Web's private API. It is not an official integration and may break with WhatsApp updates. For production at scale, use the WhatsApp Business Cloud API.
- **Voice output needs GPU**: Veena TTS requires a GPU. The Edge TTS fallback (Microsoft cloud) is not self-hosted.
- **Scheme data is curated**: The knowledge base covers ~15 major central government schemes. State-specific and newer schemes may not be included.
- **Hindi WER improves with GPU**: Running Whisper on CPU is slower and may have higher error rates. The target 5.33% WER was measured on GPU inference.
- **No real-time scheme updates**: The vector database is a snapshot. You need to re-run the ingestion pipeline when schemes change.

---

## Repository Structure

```
VaakSeva/
  backend/
    main.py              FastAPI app
    config.py            All configuration (one file, no magic numbers)
    rag/                 Ingest, embed, retrieve, rerank, pipeline
    llm/                 vLLM/Ollama client, prompts, profile extractor
    voice/               STT (Whisper), TTS (Veena/Edge), audio utils
    memory/              Per-user conversation state
    tools/               Eligibility checker
    safety/              Input filter, output validator
    observability/       Structured logging, metrics, dashboard
    models/              Pydantic schemas
  whatsapp/              Baileys TypeScript client
  data/
    schemes/             50+ Hindi/English scheme documents
    schemes_structured.json  Eligibility rules for 14 schemes
  eval/
    test_set.json        50+ Hindi Q&A evaluation pairs
    evaluate.py          Full evaluation pipeline
  dashboard/             Chart.js metrics dashboard
  scripts/               Ingest, test pipeline, generate test set
  tests/                 pytest test suite
  docker-compose.yml     Full stack: backend + Weaviate + Redis + WhatsApp
```

---

Built by [Praveen Kumar](https://github.com/mist-ic) | GCP Project: revsight-492123
