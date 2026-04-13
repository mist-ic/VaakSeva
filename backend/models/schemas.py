"""
Pydantic schemas for all VaakSeva request and response types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InputType(str, Enum):
    text = "text"
    voice = "voice"


class QueryCategory(str, Enum):
    factual = "factual"
    eligibility = "eligibility"
    procedural = "procedural"
    general = "general"


# ---------------------------------------------------------------------------
# Pipeline timing
# ---------------------------------------------------------------------------


class PipelineTimings(BaseModel):
    stt_ms: float | None = None
    embedding_ms: float | None = None
    retrieval_ms: float | None = None
    rerank_ms: float | None = None
    llm_ms: float | None = None
    tts_ms: float | None = None
    total_ms: float | None = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    chunk_id: str
    scheme_name: str
    scheme_id: str
    content: str
    language: str
    source_url: str | None = None
    chunk_index: int
    document_section: str | None = None
    score: float


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    num_candidates: int
    retrieval_ms: float
    rerank_ms: float | None = None


# ---------------------------------------------------------------------------
# User profile and memory
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    age: int | None = None
    gender: str | None = None
    state: str | None = None
    district: str | None = None
    occupation: str | None = None
    income: int | None = None
    category: str | None = None       # SC/ST/OBC/General
    education: str | None = None
    has_aadhaar: bool | None = None
    land_holding_acres: float | None = None
    is_bpl: bool | None = None        # Below poverty line


class ConversationTurn(BaseModel):
    role: str                          # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserMemoryState(BaseModel):
    phone_hash: str
    profile: UserProfile = Field(default_factory=UserProfile)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    last_active: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


class EligibilityResult(BaseModel):
    scheme_id: str
    scheme_name: str
    scheme_name_hi: str
    score: float                   # 0.0 to 1.0
    matched_criteria: list[str]
    missing_criteria: list[str]
    benefits_summary_hi: str
    apply_url: str | None = None


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class SafetyCheckResult(BaseModel):
    is_safe: bool
    flagged_patterns: list[str] = Field(default_factory=list)
    sanitized_input: str


class OutputValidationResult(BaseModel):
    is_valid: bool
    flagged_claims: list[str] = Field(default_factory=list)
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    phone_number: str = Field(..., description="User's phone number (used for memory lookup)")
    session_id: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "मेरी उम्र 25 साल है, मैं किसान हूं, कौन सी सरकारी योजनाएं मिल सकती हैं?",
                "phone_number": "+919876543210",
            }
        }


class QueryResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    response_text: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    eligibility_results: list[EligibilityResult] = Field(default_factory=list)
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    language_detected: str = "hi"
    safety: SafetyCheckResult | None = None
    output_validation: OutputValidationResult | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VoiceQueryRequest(BaseModel):
    phone_number: str
    audio_file_path: str | None = None   # path to downloaded OGG
    session_id: str | None = None


class VoiceQueryResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    transcript: str
    transcript_confidence: float
    response_text: str
    audio_response_path: str | None = None   # path to generated TTS audio
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class QueuedJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    phone_number: str
    input_type: InputType
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class RequestLog(BaseModel):
    request_id: str
    timestamp: datetime
    user_phone_hash: str
    input_type: InputType
    pipeline_timings: PipelineTimings
    retrieval: dict[str, Any] = Field(default_factory=dict)
    response_length: int
    language_detected: str
    confidence: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvalQuestion(BaseModel):
    question_hi: str
    expected_answer_contains: list[str]
    expected_scheme: str
    category: QueryCategory


class EvalResult(BaseModel):
    question: EvalQuestion
    retrieved_scheme_ids: list[str]
    response: str
    hit: bool               # expected_scheme in top-5
    factual_accurate: bool  # response contains expected keywords
    latency_ms: float
