# VaakSeva Architecture Notes

Deep-dive on design decisions, alternatives considered, and trade-offs made.

---

## Why Every Model Was Chosen

### LLM: Sarvam-30B

**Decision**: Sarvam-30B free hosted API over self-hosted vLLM/Ollama.

Sarvam-30B was trained from scratch on 16 trillion tokens across 22 Indian languages. It is not a fine-tuned Western model -- it was designed with Indic languages as first-class citizens from pretraining. This matters for Hindi because:

- Devanagari tokenization is efficient (not thousands of tokens for a simple sentence)
- Cultural context around Indian government schemes is built into the model weights
- Multi-instruction following in Hindi is significantly better

As of April 2026, Sarvam's hosted API offers Sarvam-30B at Rs 0/token (free). There is no operational reason to self-host when the same model -- with better serving infrastructure -- is available at zero cost. TTFT on the hosted API is ~1.2 seconds.

**Speed fallback**: Groq Llama-3.3-70B at 250-500 tokens/second. Groq's LPU gives the fastest token generation available. Llama 3.3-70B handles Hindi well with a strong system prompt. We switch to Groq automatically when Sarvam API latency exceeds the LLM budget (3 seconds).

**Self-hosted**: vLLM + Sarvam-30B (original design) is still supported. Switch `LLM_BACKEND=vllm` if API dependency is not acceptable.

### Embeddings: Qwen3-Embedding-0.6B

**Decision**: Qwen3-Embedding-0.6B over mE5-large or BGE-M3.

Qwen3-Embedding-0.6B outperforms multilingual-e5-large-instruct on MTEB Multilingual despite being much lighter. The 0.6B model produces 1024D embeddings, the same dimension as E5-large, making Weaviate collection schemas compatible between the two models.

At query time (real-time pipeline), Qwen3-0.6B adds ~50-100ms on CPU -- within the 500ms embedding budget. For highest quality offline indexing, the 8B variant (`Qwen/Qwen3-Embedding-8B`, 7168D) can be used by setting the env var during ingestion.

Asymmetric retrieval (the reason instructions matter): query embeddings use the prefix "Retrieve relevant documents for the following query: ". Document embeddings use no prefix. This asymmetric design significantly reduces false positives in RAG retrieval.

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

### STT: Sarvam Saaras V3

**Decision**: Sarvam Saaras V3 hosted API over self-hosted Whisper.

Saaras V3 (released February 2026) achieves ~19% WER on the IndicVoices benchmark, outperforming GPT-4o Transcribe, Deepgram Nova-3, and ElevenLabs Scribe v2 on all top 10 Indian languages. Trained on 1M+ hours of curated multilingual audio with explicit coverage of telephony noise, code-mixed speech, and rural Indian accents.

At Rs 30/hour of audio, a 20-second WhatsApp voice note costs Rs 0.17 -- essentially free at demo volume. This eliminates the CPU bottleneck of running Whisper locally.

**Self-hosted fallback**: faster-whisper large-v3 (upgraded from large-v2). large-v3 improves Hindi WER by 4+ points over large-v2. For best open-source Hindi accuracy: AI4Bharat's IndicWhisper achieves the lowest WER on 39/59 Vistaar benchmarks.

**Rejected for STT**: Groq Whisper (stock large-v3 without Indian fine-tuning). Real-world reports confirm degraded accuracy on Hindi/Hinglish/accented speech.

### TTS: Sarvam Bulbul v3 + Kokoro v1.0

**Decision**: Sarvam Bulbul v3 (API primary) + Kokoro v1.0 (self-hosted fallback) over Veena.

Veena (maya-research) had no publicly available benchmarks and no peer-reviewed evaluation. In the codebase, the Veena synthesise method raised `NotImplementedError` because the actual audio decoding API was never finalized by the model author.

Bulbul v3 (released February 4, 2026): CER of 0.0173 on Sarvam's benchmark, designed for production real-time applications, ~600ms latency. Priced at Rs 30/10,000 characters.

Kokoro v1.0 (hexgrad/Kokoro-82M): 82M parameters, Apache 2.0, ranked #1 on TTS Arena. Runs fast on CPU -- generating 5 minutes of audio in seconds. Hindi voices hf_alpha and hf_omega support correct pronunciation of Hindi vocabulary and technical government scheme terminology.

**Edge TTS**: Microsoft Edge TTS remains as emergency fallback. It is an unofficial endpoint and carries deprecation risk.

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

With the Sarvam API handling STT, LLM, and TTS, GPU is no longer required for the primary pipeline. If self-hosting:

1. **GPU-0**: Sarvam-30B via vLLM (set LLM_BACKEND=vllm)
2. **GPU-1**: Qwen3 embeddings + Qwen3 reranker + faster-whisper
3. **Load balancing**: Route text-only queries to Groq fallback if primary is busy

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
