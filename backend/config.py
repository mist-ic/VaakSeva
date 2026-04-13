"""
Centralized configuration for VaakSeva.

All model paths, thresholds, prompts, and parameters live here.
No magic numbers in the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
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
    # LLM
    # -------------------------------------------------------------------------
    llm_backend: Literal["ollama", "vllm"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "sarvam-m:latest"
    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "sarvamai/sarvam-30b"

    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_max_tokens: int = 1024
    llm_timeout_s: int = 60
    llm_max_retries: int = 2

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    embedder_backend: Literal["qwen3", "e5"] = "e5"
    qwen3_embed_model: str = "Qwen/Qwen3-Embedding-8B"
    e5_embed_model: str = "intfloat/multilingual-e5-large-instruct"
    embed_batch_size: int = 32
    embed_device: str = "cpu"

    # -------------------------------------------------------------------------
    # Reranker
    # -------------------------------------------------------------------------
    reranker_backend: Literal["qwen3", "noop"] = "noop"
    qwen3_reranker_model: str = "Qwen/Qwen3-Reranker-8B"
    reranker_device: str = "cpu"

    # -------------------------------------------------------------------------
    # STT
    # -------------------------------------------------------------------------
    stt_backend: Literal["whisper"] = "whisper"
    stt_model: str = "large-v2"          # faster-whisper model size
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_language: str = "hi"
    stt_language_threshold: float = 0.7  # reject if language confidence < this

    # -------------------------------------------------------------------------
    # TTS
    # -------------------------------------------------------------------------
    tts_backend: Literal["veena", "edge"] = "edge"
    edge_tts_voice: str = "hi-IN-MadhurNeural"
    veena_model: str = "maya-research/Veena"
    tts_device: str = "cpu"

    # -------------------------------------------------------------------------
    # Weaviate
    # -------------------------------------------------------------------------
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_grpc_port: int = 50051
    weaviate_collection: str = "GovernmentSchemes"

    # -------------------------------------------------------------------------
    # Redis / Queue
    # -------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379"

    # -------------------------------------------------------------------------
    # RAG pipeline
    # -------------------------------------------------------------------------
    top_k_retrieval: int = 50
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
