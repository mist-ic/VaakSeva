# VaakSeva Architecture Notes

Deep-dive on design decisions, alternatives considered, and trade-offs made.

---

## Why Every Model Was Chosen

### LLM: Sarvam-30B

**Decision**: Sarvam-30B over Llama-3-8B, Gemma-2, or Mistral-Small.

Sarvam-30B was trained from scratch on 22 Indian languages with SFT + RL. It is not a fine-tuned Western model - it was designed with Indic languages as first-class citizens from pretraining. This matters for Hindi because:

- Devanagari tokenization is efficient (not thousands of tokens for a simple sentence)
- Cultural context around Indian government schemes is built into the model weights
- Multi-instruction following in Hindi is significantly better

Compute cost: MoE with 30B total parameters but 2.4B active per token. In practice, it runs at the speed of an 8B dense model on a single GPU.

The Apache 2.0 license and self-hosting requirement made OpenAI/Anthropic/Gemini a non-starter. This directly mirrors Puch AI's architecture.

**Alternative**: Sarvam-M (24B dense). Better if MoE serving proves unreliable on a single GPU. I tested both - Sarvam-30B gives better Hindi coherence for longer responses, Sarvam-M is slightly faster.

### Embeddings: Qwen3-Embedding-8B

**Decision**: Qwen3-Embedding-8B over mE5-large or BGE-M3.

Qwen3-Embedding-8B ranks #1 on MTEB Multilingual (70.58 as of April 2026). Its key advantage is the use of asymmetric instruction prefixes:
- Queries: "Retrieve relevant documents for the following query: {text}"
- Documents: no prefix (embedded as-is)

This asymmetric design is important for RAG: the query embedding and document embedding live in different "semantic spaces" by design, which reduces false positives.

**Fallback**: `multilingual-e5-large-instruct` (560M). It ranks high on MTEB(Indic) per the MMTEB paper, works on CPU, and the E5 instruction format (query: / passage: ) maps naturally to Weaviate's hybrid search.

**Rejected**: BAAI/bge-m3. While it supports dense+sparse+ColBERT in one pass, its sparse representation adds complexity to the ingestion pipeline. Weaviate handles BM25 server-side, making bge-m3's sparse advantage irrelevant here.

### Vector Database: Weaviate

**Decision**: Weaviate over Qdrant, ChromaDB, or Milvus.

The key requirement was native hybrid search without client-side BM25 computation:

| DB | BM25 Support | Notes |
|----|-------------|-------|
| Weaviate | Server-side BM25F | Automatic on all indexed text properties |
| Qdrant | Client-side sparse vectors | Must compute and send SPLADE/BM25 vectors |
| ChromaDB | No native hybrid | Must implement BM25 separately |
| Milvus | Yes (recent) | More complex deployment |

Weaviate's `hybrid()` query takes the plain text query and does BM25 on the server. Zero client-side sparse vector computation. One API call. This is the right trade-off for this project.

The tunable `alpha` parameter for Reciprocal Rank Fusion is important: for Hindi conversational queries (alpha=0.75, dense-heavy), BM25 is less reliable due to morphological variation. For scheme name lookups (alpha=0.5), BM25 precision is valuable.

### STT: collabora/faster-whisper-large-v2-hindi

**Decision**: This specific model over openai/whisper-large-v2 or vasista22/whisper-hindi-large-v2.

Published WERs on Hindi FLEURS:
- `collabora/faster-whisper-large-v2-hindi`: **5.33%** (CC-BY-4.0)
- `vasista22/whisper-hindi-large-v2`: 6.8% (Apache 2.0)
- `openai/whisper-large-v2`: ~12% (no Hindi-specific tuning)

The collabora model was trained on ~3000h Hindi data (Shrutilipi + IndicVoices-R + Lahaja). The CTranslate2 format provides ~4x speedup over PyTorch inference.

**Confidence gating**: WhatsApp voice notes often contain noise (traffic, background conversations). `info.language_probability` from Whisper is used as an audio quality gate - if below 0.7, return a Hindi error asking the user to resend.

**Silero VAD**: Whisper's built-in VAD (SILERO-based) segments the audio and skips silence. Important for voice notes that start with ambient noise before the user starts speaking.

### TTS: Veena (maya-research)

**Decision**: Veena over AI4Bharat Indic-Parler-TTS or Coqui TTS.

Veena is a 3B Llama-based autoregressive TTS with SNAC 24kHz codec, explicitly designed for Hindi + English + Hinglish code-mix. MOS of ~4.2/5 on Hindi. Apache 2.0.

Key advantage: Hinglish code-mix support. WhatsApp users frequently mix Hindi and English ("mujhe PM Kisan ka paisa kab aayega"), and Veena handles this naturally.

**Edge TTS fallback**: Microsoft's Edge TTS provides natural Hindi voices (`hi-IN-MadhurNeural`, `hi-IN-SwaraNeural`) with zero setup. Network-dependent (200-500ms) and not self-hosted, but invaluable for development iteration.

---

## Pipeline Design Decisions

### Async Job Queue Architecture

The spec calls for a job queue pattern. Here's why it matters:

WhatsApp enforces a webhook acknowledgment timeout of ~15 seconds. A full pipeline run (STT + embed + retrieve + LLM + TTS) can take 3-8 seconds on GPU and 15-30 seconds on CPU. Without async:

```
User message -> FastAPI -> [runs full pipeline] -> ACK to WhatsApp [TIMEOUT if >15s]
```

With async:
```
User message -> FastAPI -> ACK immediately (200ms) -> Job enqueued
                                                           |
                                              Worker processes pipeline
                                                           |
                                              Baileys sends response back
```

Current implementation runs synchronously in FastAPI for simplicity (FastAPI is async, so endpoint doesn't block the event loop). Actual Redis/RQ job queue is wired but not the default path. For production deployment with concurrent users, switch `ASYNC_PIPELINE=true` to route through the queue.

### Conversation Memory: JSON Files vs Database

**Decision**: JSON files on disk, not PostgreSQL, Redis, or MongoDB.

Reasoning:
- 50-1000 concurrent users for a demo/portfolio project
- Structured query language is not needed for simple key-value + append operations
- Zero operational overhead (no database process to manage)
- Files are trivially inspectable for debugging
- The `purge_inactive_users()` method handles cleanup

For production at Puch's scale (1M monthly users), this obviously needs to migrate to a proper store (Redis for real-time state + PostgreSQL for history is a natural choice).

### Chunking Strategy

Hindi-aware separators in priority order:
```python
["\n\n", "\n", "।", ".", "?", "!", " ", ""]
```

The Devanagari purna viram (।) is the primary sentence boundary in Hindi. Standard English-only text splitters miss this, resulting in chunks that cut sentences mid-thought. Our chunker correctly treats "।" as a separator.

Chunk size of 500 tokens with 50-token overlap is tuned for scheme documents:
- Most eligibility criteria fit within 500 tokens
- 50-token overlap prevents losing context at boundaries
- Smaller chunks (200 tokens) would lose the surrounding context that Whisper needs for Hindi disambiguation

### Safety: Pattern-Based vs LLM-Based

**Decision**: Pattern-based detection (regex) as the primary filter, not a separate LLM call.

Tradeoffs:
- LLM-based guardrails (e.g., Llama-Guard): better coverage, 300-500ms overhead per request
- Regex-based: near-zero latency, deterministic, good coverage for known patterns

For a government scheme assistant with a narrow domain, regex catches the most common attack vectors without the latency overhead. An LLM-based layer could be added in production for robustness.

---

## Scaling Considerations

### Multi-GPU Setup

For traffic above 10 concurrent users:

1. **GPU-0**: Sarvam-30B via vLLM (tensor parallel across 2 GPUs if needed)
2. **GPU-1**: Whisper + Embeddings + Reranker + Veena TTS
3. **Load balancing**: Route text-only queries to a smaller LLM replica if the main one is busy

### Weaviate Scaling

For the demo (50-100 documents, 500-2000 chunks), Weaviate embedded or Docker is sufficient. For production:
- Weaviate Cloud Services (managed) or
- Self-hosted cluster on GKE n Kubernetes

### Language Model Distillation

A natural next step: fine-tune a smaller model (Sarvam-M or Llama-3-8B) specifically on government scheme Q&A pairs. With a curated training set of 10,000+ Hindi scheme Q&A examples:
- Reduce LLM call latency by 50% (smaller model)
- Improve factual accuracy on scheme-specific questions
- Enable on-device deployment on cheaper hardware

This is the kind of work Puch AI does internally for their domain-specific models.

---

## Evaluation Methodology

### ASR Evaluation Beyond WER

Traditional WER penalizes valid Hindi spelling variations. "जाऊंगा" and "जाउंगा" are both correct Hindi but count as errors in WER. Sarvam AI's `llm_intent_entity` framework addresses this with:

- **Intent Preservation Score** (binary): Does the ASR output preserve the core meaning? Uses Gemini as judge.
- **Entity Preservation Score** (0-1): Fraction of named entities (scheme names, numbers, places) correctly transcribed.

For a government scheme assistant, entity preservation is critical: "PM Kisan" in the audio must come out as "PM Kisan" (or "पीएम किसान"), not "PM Kisaan" or similar. >=0.95 entity score threshold is appropriate for this domain.

### RAG Evaluation

`eval/test_set.json` contains 50+ Hindi Q&A pairs across three categories:
- **Factual** (40%): specific numbers, benefit amounts, dates
- **Eligibility** (35%): "does X qualify for Y?"
- **Procedural** (25%): "how to apply for X?"

Retrieval is evaluated at hit@5 (expected scheme in top-5). Response quality is evaluated by keyword match against known answers - not LLM-as-judge (avoids evaluation API costs for a demo project).

---

Built by [Praveen Kumar](https://github.com/mist-ic)
