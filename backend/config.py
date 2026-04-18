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
    llm_backend: Literal["sarvam", "groq", "ollama", "vllm"] = "sarvam"

    # Sarvam-30B (free hosted API as of April 2026, Hindi-native)
    sarvam_llm_model: str = "sarvam-30b"

    # Self-hosted fallbacks
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "sarvam-m:latest"
    vllm_base_url: str = "http://localhost:8000"
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
    embedder_backend: Literal["qwen3", "e5"] = "qwen3"

    # Qwen3-Embedding-0.6B: outperforms multilingual-e5-large-instruct,
    # fast on CPU (~50-100ms at query time), 1024D embeddings
    qwen3_embed_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # For offline document indexing (better quality, optional override):
    # set QWEN3_EMBED_MODEL=Qwen/Qwen3-Embedding-8B during ingestion
    e5_embed_model: str = "intfloat/multilingual-e5-large-instruct"

    embed_batch_size: int = 32
    embed_device: str = "cpu"

    # -------------------------------------------------------------------------
    # Reranker
    # -------------------------------------------------------------------------
    # Qwen3-Reranker-0.6B: fast on CPU, MTEB multilingual reranking top performer
    # 8B variant (Qwen/Qwen3-Reranker-8B) adds 2-5s on CPU, unacceptable
    reranker_backend: Literal["qwen3", "noop"] = "qwen3"
    qwen3_reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_device: str = "cpu"

    # -------------------------------------------------------------------------
    # STT
    # -------------------------------------------------------------------------
    # Sarvam Saaras V3: #1 on IndicVoices Hindi benchmark, beats GPT-4o
    # Whisper fallback: large-v3 (upgraded from large-v2, 4+ WER points better)
    stt_backend: Literal["sarvam", "whisper"] = "sarvam"

    # Sarvam STT config
    sarvam_stt_model: str = "saaras:v3"
    sarvam_stt_language: str = "hi-IN"

    # Whisper fallback config
    stt_model: str = "large-v3"          # upgraded from large-v2
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str = "hi"
    stt_language_threshold: float = 0.7

    # -------------------------------------------------------------------------
    # TTS
    # -------------------------------------------------------------------------
    # Sarvam Bulbul v3: CER 0.0173, ~600ms latency, production-ready Hindi TTS
    # Kokoro fallback: 82M params, Apache 2.0, #1 TTS Arena, fast on CPU
    # Edge TTS: emergency fallback only (unofficial endpoint)
    tts_backend: Literal["sarvam", "kokoro", "edge"] = "sarvam"

    # Sarvam Bulbul v3 config
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "priya"       # Hindi female; verified with bulbul:v3

    # Kokoro config
    kokoro_voice: str = "hf_alpha"          # hf_alpha = Hindi female, hf_omega = male

    # Edge TTS config (fallback)
    edge_tts_voice: str = "hi-IN-MadhurNeural"

    tts_device: str = "cpu"

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
    # GCP
    # -------------------------------------------------------------------------
    gcp_project_id: str = "revsight-492123"
    gcp_region: str = "asia-south1"

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
