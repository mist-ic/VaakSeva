# VaakSeva

A self-hosted, voice-first Hindi AI assistant over WhatsApp that helps Indian citizens discover government welfare schemes. Built on India's own sovereign AI stack with zero dependency on commercial APIs like OpenAI or Anthropic.

"Vaak" = speech in Sanskrit. "Seva" = service.

---

## The problem

400+ million Indians qualify for government welfare schemes and never claim them. The information is scattered across dozens of ministry websites, mostly in English, and requires internet literacy to navigate. There's no voice-first interface that understands how people actually talk.

A farmer says "मुझे क्या मिल सकता है" (what can I get) into WhatsApp. He should get a grounded, spoken Hindi answer. VaakSeva does that.

---

## How it works

The bot accepts Hindi voice notes and text messages on WhatsApp. It transcribes voice, retrieves the most relevant scheme information from a curated knowledge base, generates a grounded Hindi response using Sarvam-30B, and sends back both text and a voice note reply.

**Voice query flow:**
```
[Hindi voice note on WhatsApp]
       |
Baileys downloads OGG/Opus
       |
faster-whisper large-v2-hindi  --  5.33% WER on Hindi FLEURS, GPU INT8
       |
Qwen3-Embedding-8B  --  7168D vector, GPU, ~40ms
       |
Weaviate hybrid search  --  BM25F + dense vector, alpha=0.75, top-20
       |
Qwen3-Reranker-8B  --  CausalLM logit scoring, top-20 to top-5, ~300ms GPU
       |
User memory context  --  accumulated state/age/occupation from prior turns
       |
Sarvam-30B via SGLang  --  Q4_K_M GGUF, RadixAttention, A10 GPU
       |
Veena 3B NF4  --  India's first open-source Hindi TTS, GPU, ~2s first chunk
       |
[Text reply + Hindi voice note sent back on WhatsApp]
```

**Multi-turn memory:**
```
Turn 1: "मैं कर्नाटक से हूं"          -> state=Karnataka
Turn 2: "किसान हूं, आय 2 लाख है"     -> occupation=farmer, income=200000
Turn 3: "कौन सी योजनाएं मिल सकती हैं?"  -> retrieved with full profile context
Turn 4: "PM Kisan के बारे में और बताओ"  -> detailed PM-KISAN chunks fetched
```

---

## Tech stack

| Component | Model | Notes |
|-----------|-------|-------|
| **LLM** | Sarvam-30B Q4_K_M via SGLang | Apache 2.0 MoE, 2.4B active params, 63.3% GPQA, serves from A10 GPU |
| **STT** | faster-whisper large-v2-hindi | 5.33% WER on Hindi FLEURS, best open-source Hindi STT, CTranslate2 INT8 |
| **TTS** | Veena 3B NF4 (maya-research/Veena) | India's first open-source Hindi TTS, Apache 2.0, 4 voices, ~2GB VRAM |
| **Embedder** | Qwen3-Embedding-8B | MTEB Multilingual #1 (70.58), 7168D, Apache 2.0 |
| **Reranker** | Qwen3-Reranker-8B | MTEB multilingual reranking #1, CausalLM logit scoring approach |
| **Vector DB** | Weaviate 1.32 | Native BM25F + HNSW hybrid in one query, no client-side sparse vectors |
| **LLM serving** | SGLang (RadixAttention) | 29% higher throughput than vLLM on RAG workloads due to prefix caching |
| **WhatsApp** | Baileys 6.x | WebSocket to WhatsApp Web, text + voice note handling |
| **Backend** | FastAPI + Redis/RQ | Async job queue so the webhook never times out |
| LLM (dev) | Groq Llama-3.3-70B | Speed fallback, 250-500 tok/s, swap with LLM_BACKEND=groq |
| TTS (CPU) | Kokoro-82M | TTS Arena #1, Apache 2.0, CPU-fast when no GPU available |

### Why everything is self-hosted

The whole point is sovereign AI: no data leaves your infrastructure, no rate limits, no per-call costs at scale. This is what production Indic AI looks like when you're serious about it.

### Why SGLang over vLLM

In a RAG workload, every request has the same system prompt + retrieved chunks prefix. SGLang's RadixAttention caches that prefix across requests, giving 29% higher throughput compared to vLLM's PagedAttention. For Sarvam-30B specifically (MoE architecture), SGLang also implements expert-parallel routing which gives an additional speedup.

### Why Weaviate over Qdrant or ChromaDB

Weaviate computes BM25F server-side in the same query as vector search. Qdrant requires computing sparse vectors on the client. ChromaDB has no native hybrid at all. For this project, the simplest path wins.

---

## Evaluation results

Evaluated on 50 Hindi questions covering 14 central government schemes across factual, eligibility, procedural, and general categories. Full pipeline: embed, hybrid retrieve, rerank, Sarvam-30B generate.

| Metric | Result | Target |
|--------|--------|--------|
| Retrieval hit@5 | **88%** | >70% |
| Factual accuracy | **88%** | >60% |

By category:

| Category | Questions | Hit@5 | Factual |
|----------|-----------|-------|---------|
| Factual | 24 | 87.5% | 83.3% |
| Eligibility | 13 | 84.6% | 92.3% |
| Procedural | 11 | 90.9% | 90.9% |
| General | 2 | 100% | 100% |

Latency (dev box, CPU embedder + reranker, Sarvam API over network):

| | Latency |
|-|---------|
| P50 | 6.5s |
| P95 | 9.9s |

Expected on A10 GPU (self-hosted SGLang + Whisper GPU + Veena GPU):

| Stage | Budget |
|-------|--------|
| STT | <1.0s |
| Embed | <0.05s |
| Retrieve | <0.2s |
| Rerank | <0.5s |
| LLM | <2.5s |
| TTS first chunk | <1.5s |
| **Total** | **<6s** |

---

## GPU deployment

### Single A10 (24 GB) -- ~$0.20/hr on Vast.ai

| Component | VRAM |
|-----------|------|
| Sarvam-30B Q4_K_M via SGLang | ~19 GB |
| faster-whisper INT8 | ~1.5 GB (sequential) |
| Veena 3B NF4 | ~2 GB (sequential) |
| Weaviate + Redis | CPU |

Qwen3-8B embedder and reranker run on CPU at acceptable latency (<1s each) when only one GPU is available.

### L40S (48 GB) -- best option

Runs Sarvam-30B at Q6_K (~26 GB) for better quality. All models GPU-resident simultaneously, no sequential sharing required.

### Two GPUs

- GPU-0: Sarvam-30B via SGLang
- GPU-1: Qwen3-Embed-8B + Qwen3-Reranker-8B + Veena NF4 + Whisper

Download Sarvam-30B before starting the stack:
```bash
huggingface-cli download Sumitc13/sarvam-30b-GGUF --local-dir ./models/sarvam-30b
```

---

## Setup

### Dev mode (no GPU, Sarvam API fallbacks)

```bash
git clone https://github.com/mist-ic/VaakSeva.git && cd VaakSeva

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set SARVAM_API_KEY and switch:
# LLM_BACKEND=sarvam  STT_BACKEND=sarvam  TTS_BACKEND=sarvam
# QWEN3_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
# QWEN3_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B

docker compose up weaviate redis -d
python scripts/ingest.py
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload

# WhatsApp client (separate terminal, scan QR with your phone)
cd whatsapp && npm install && npm run dev
```

### Production GPU pod (Vast.ai / RunPod)

```bash
# Download model, set MODEL_DIR in .env, then:
docker compose up -d

# SGLang takes 3-5 min on first start to load Sarvam-30B
docker compose logs -f sglang

# Ingest and run eval
docker compose exec backend python scripts/ingest.py
docker compose exec backend python eval/evaluate.py
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=backend --cov-report=html
```

---

## Dashboard

`http://localhost:8080/dashboard` -- query volume, latency per stage, top schemes queried, voice vs text ratio.

---

## Safety

Input: pattern-based injection detection (English and Hindi patterns), NFC normalisation, control character stripping.

Output: `OutputValidator` cross-references benefit amounts in LLM responses against `data/schemes_structured.json`. If the LLM hallucinates a wrong rupee figure for a scheme, it gets flagged in the structured logs.

---

## Limitations

- **Baileys uses WhatsApp Web's unofficial protocol.** There's a ban risk and it may break on WhatsApp updates. For production serving real users, swap to the official WhatsApp Business Cloud API.
- **Scheme coverage is 14 major central schemes.** State-specific schemes and recent launches need a re-ingestion run.
- **Veena TTS requires a GPU.** CPU inference takes 30-60s per response, which is unusable. Use Kokoro-82M (TTS_BACKEND=kokoro) if you're on CPU.

---

## Repo layout

```
backend/
  main.py              FastAPI app
  config.py            All config, no magic numbers
  rag/                 ingest, embedder, retriever, reranker, pipeline
  llm/                 SGLang/Sarvam/Groq clients, prompts, extractor
  voice/               stt.py (Whisper), tts.py (Veena/Kokoro), audio_utils.py
  memory/              per-user JSON conversation state
  tools/               eligibility checker
  safety/              input_filter.py, output_validator.py
  observability/       structured logging, metrics, dashboard
  models/              pydantic schemas
whatsapp/              Baileys TypeScript client
data/
  schemes/             50+ Hindi/English source documents
  schemes_structured.json  eligibility rules + benefit amounts
eval/
  test_set.json        50 Hindi Q&A pairs
  evaluate.py          eval pipeline -- recall@5, factual, latency
dashboard/             Chart.js served at /dashboard
docker-compose.yml     SGLang + backend + Weaviate + Redis + WhatsApp
backend/Dockerfile     CUDA 12.4 base (--build-arg CUDA_BASE=cpu for local)
```
