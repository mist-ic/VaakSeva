"""
Centralized configuration for VaakSeva.

All model paths, thresholds, prompts, and parameters live here.
No magic numbers in the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # Sarvam AI (shared subscription key for STT, LLM, TTS)
    # -------------------------------------------------------------------------
    sarvam_api_key: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai"

    # -------------------------------------------------------------------------
    # Groq (speed fallback for LLM)
    # -------------------------------------------------------------------------
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai"
    groq_model: str = "llama-3.3-70b-versatile"

    # -------------------------------------------------------------------------
    # LLM
    # -------------------------------------------------------------------------
    # Production (GPU): sglang (self-hosted Sarvam-30B via SGLang)
    # Dev (API):        sarvam | groq
    # Dev (CPU):        ollama
    llm_backend: Literal["sglang", "vllm", "ollama", "sarvam", "groq"] = "sglang"

    # SGLang (production primary) — exposes OpenAI-compatible API
    # Start: python -m sglang.launch_server --model-path /models/sarvam-30b-q4_k_m.gguf --port 8080
    sglang_base_url: str = "http://localhost:8080"
    sglang_model: str = "sarvam-30b"   # model name as registered with SGLang

    # Sarvam-30B (Apache 2.0, MoE: 32B total / 2.4B active, Hindi-native)
    sarvam_llm_model: str = "sarvam-30b"

    # Self-hosted fallbacks
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "sarvam-m:latest"
    vllm_base_url: str = "http://localhost:8001"
    vllm_model: str = "sarvamai/sarvam-30b"

    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    # Sarvam-30B is a reasoning model: reasoning chain consumes 500-800 tokens
    # before producing the final content. Set max_tokens high enough for both.
    llm_max_tokens: int = 4096
    llm_timeout_s: int = 60
    llm_max_retries: int = 2

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    # Production (GPU): Qwen3-Embedding-8B (#1 MTEB Multilingual, 16GB FP16 / 8GB INT8)
    # Dev (CPU):        Qwen3-Embedding-0.6B (1.2GB, still outperforms mE5-large)
    # Switch via: QWEN3_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B in .env for local dev
    embedder_backend: Literal["qwen3", "e5"] = "qwen3"
    qwen3_embed_model: str = "Qwen/Qwen3-Embedding-8B"
    e5_embed_model: str = "intfloat/multilingual-e5-large-instruct"

    embed_batch_size: int = 32
    embed_device: str = "auto"   # auto = cuda if available, else cpu

    # -------------------------------------------------------------------------
    # Reranker
    # -------------------------------------------------------------------------
    # Production (GPU): Qwen3-Reranker-8B (#1 MTEB Multilingual reranking, INT8 ~8GB)
    # Dev (CPU):        Qwen3-Reranker-0.6B or noop (noop = no reranking)
    # Switch via: RERANKER_BACKEND=noop or QWEN3_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
    reranker_backend: Literal["qwen3", "noop"] = "qwen3"
    qwen3_reranker_model: str = "Qwen/Qwen3-Reranker-8B"
    reranker_device: str = "auto"  # auto = cuda if available, else cpu

    # -------------------------------------------------------------------------
    # STT
    # -------------------------------------------------------------------------
    # Production (self-hosted): collabora/faster-whisper-large-v2-hindi (5.33% WER on Hindi)
    # Dev (API):                sarvam (Saaras V3, best Hindi STT but external API)
    # Switch via: STT_BACKEND=sarvam in .env for dev mode
    stt_backend: Literal["whisper", "sarvam"] = "whisper"

    # Sarvam STT config (dev/demo fallback)
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_language: str = "hi-IN"

    # Whisper config (production primary)
    # collabora/faster-whisper-large-v2-hindi: 5.33% WER on Hindi FLEURS (best open-source)
    stt_model: str = "large-v3"          # use large-v3 for general; see ARCHITECTURE.md
    stt_device: str = "auto"  # auto = cuda if available, else cpu
    stt_compute_type: str = "int8"
    stt_language: str = "hi"
    stt_language_threshold: float = 0.7

    # -------------------------------------------------------------------------
    # TTS
    # -------------------------------------------------------------------------
    # Production (GPU):   veena  — maya-research/Veena 3B NF4 (self-hosted, Apache 2.0)
    # Dev/Demo (API):     sarvam — Sarvam Bulbul v3 (hosted API, needs SARVAM_API_KEY)
    # CPU fallback:       kokoro — Kokoro-82M (#1 TTS Arena, Apache 2.0, fast on CPU)
    # Emergency fallback: edge   — Microsoft Edge TTS (unofficial endpoint)
    tts_backend: Literal["veena", "kokoro", "sarvam", "edge"] = "veena"

    # Veena 3B config (production primary — self-hosted on GPU)
    # maya-research/Veena: India's first open-source Hindi/English TTS, Apache 2.0
    # Voices: aditi (female), ravi (male), priya (female), arjun (male)
    veena_model: str = "maya-research/Veena"
    veena_voice: str = "aditi"              # aditi = Hindi female (default)

    # Sarvam Bulbul v3 config (dev/demo fallback)
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "priya"       # Hindi female; verified with bulbul:v3

    # Kokoro config (CPU fallback)
    kokoro_voice: str = "hf_alpha"          # hf_alpha = Hindi female, hf_omega = male

    # Edge TTS config (emergency fallback only)
    edge_tts_voice: str = "hi-IN-MadhurNeural"

    tts_device: str = "auto"  # auto = cuda if available, else cpu

    # -------------------------------------------------------------------------
    # Weaviate
    # -------------------------------------------------------------------------
    weaviate_host: str = "localhost"
    weaviate_port: int = 7700
    weaviate_grpc_port: int = 50051
    weaviate_collection: str = "GovernmentSchemes"

    # -------------------------------------------------------------------------
    # Redis / Queue
    # -------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379"

    # -------------------------------------------------------------------------
    # RAG pipeline
    # -------------------------------------------------------------------------
    top_k_retrieval: int = 20    # reduced from 50 (reranker is now enabled by default)
    top_k_rerank: int = 5
    hybrid_alpha: float = 0.75   # 1.0=pure dense, 0.0=pure BM25
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # -------------------------------------------------------------------------
    # Latency budgets (milliseconds)
    # -------------------------------------------------------------------------
    latency_budget_stt_ms: int = 1000
    latency_budget_embedding_ms: int = 500
    latency_budget_retrieval_ms: int = 200
    latency_budget_llm_ms: int = 3000
    latency_budget_tts_ms: int = 1000
    latency_budget_total_ms: int = 6000

    # -------------------------------------------------------------------------
    # Privacy and security
    # -------------------------------------------------------------------------
    phone_hash_salt: str = "change-this-in-production"

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"

    # -------------------------------------------------------------------------
    # Server
    # -------------------------------------------------------------------------
    fastapi_port: int = 8080
    allowed_origins: list[str] = ["*"]

    # -------------------------------------------------------------------------
    # Deployment
    # -------------------------------------------------------------------------
    # GCP project (used for logging, monitoring)
    gcp_project_id: str = "revsight-492123"
    gcp_region: str = "asia-south1"

    # GPU pod model directory (Vast.ai / RunPod / Lambda Labs)
    # Mount point: bind-mount /models volume in docker-compose.yml
    model_dir: str = "/models"

    # -------------------------------------------------------------------------
    # Derived paths (not from env)
    # -------------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def schemes_dir(self) -> Path:
        return self.data_dir / "schemes"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def user_memory_dir(self) -> Path:
        return self.data_dir / "user_memory"

    @property
    def audio_cache_dir(self) -> Path:
        return self.data_dir / "audio_cache"

    @property
    def schemes_structured_path(self) -> Path:
        return self.data_dir / "schemes_structured.json"


# Singleton instance — import this everywhere
settings = Settings()
