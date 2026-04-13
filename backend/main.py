"""
VaakSeva FastAPI application.

Entry point. Registers all routers, middleware, and startup/shutdown events.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models.schemas import (
    QueryRequest,
    QueryResponse,
    VoiceQueryRequest,
    VoiceQueryResponse,
    QueuedJob,
    InputType,
)
from backend.observability.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources at startup, clean up on shutdown."""
    logger.info("VaakSeva starting up...")

    # Create required directories
    for dir_path in [
        settings.log_dir,
        settings.user_memory_dir,
        settings.audio_cache_dir,
        settings.processed_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Lazy-import and warm up pipeline components
    # These are imported here to avoid import-time GPU allocations during tests
    try:
        from backend.rag.pipeline import RAGPipeline
        app.state.rag = RAGPipeline()
        logger.info("RAG pipeline ready")
    except Exception as exc:
        logger.warning("RAG pipeline failed to initialise: %s — running without RAG", exc)
        app.state.rag = None

    try:
        from backend.memory.user_memory import UserMemory
        app.state.memory = UserMemory(settings.user_memory_dir)
        logger.info("User memory ready")
    except Exception as exc:
        logger.warning("User memory failed: %s", exc)
        app.state.memory = None

    yield

    logger.info("VaakSeva shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="VaakSeva",
    description="Indic voice RAG assistant over WhatsApp for government scheme discovery",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "service": "vaakseva",
        "rag_ready": app.state.rag is not None,
        "memory_ready": app.state.memory is not None,
    }


# ---------------------------------------------------------------------------
# Query endpoint (text)
# ---------------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse, tags=["pipeline"])
async def query(request: Request, body: QueryRequest):
    """
    Process a Hindi text query and return a grounded response.

    The pipeline runs synchronously here for simplicity.
    In production, enqueue as a background job via Redis/RQ.
    """
    rag = request.app.state.rag
    memory = request.app.state.memory

    if rag is None:
        raise HTTPException(503, "RAG pipeline not available")

    # Load user context
    user_profile = {}
    conversation_history = []
    if memory:
        user_profile = memory.get_profile(body.phone_number)
        conversation_history = memory.get_conversation_history(body.phone_number)

    try:
        result: QueryResponse = await rag.aquery(
            query_text=body.message,
            user_profile=user_profile,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        logger.exception("Pipeline error for query: %s", body.message[:80])
        raise HTTPException(500, f"Pipeline error: {exc}") from exc

    # Persist turn to memory
    if memory:
        memory.add_turn(body.phone_number, body.message, result.response_text)

    return result


# ---------------------------------------------------------------------------
# Voice query endpoint
# ---------------------------------------------------------------------------


@app.post("/api/voice-query", response_model=VoiceQueryResponse, tags=["pipeline"])
async def voice_query(
    request: Request,
    phone_number: str,
    audio: UploadFile = File(...),
):
    """
    Accept a WhatsApp voice note (OGG/Opus), transcribe it with Whisper,
    run the RAG pipeline, synthesise a Hindi audio response.
    """
    import tempfile
    from backend.voice.stt import get_stt
    from backend.voice.tts import get_tts
    from backend.voice.audio_utils import convert_ogg_to_wav, convert_wav_to_ogg

    stt = get_stt()
    tts = get_tts()
    rag = request.app.state.rag
    memory = request.app.state.memory

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        ogg_path = Path(tmp.name)

    timings = {}

    try:
        # STT
        t0 = time.perf_counter()
        wav_path = convert_ogg_to_wav(ogg_path)
        transcript_result = stt.transcribe(wav_path)
        timings["stt_ms"] = (time.perf_counter() - t0) * 1000

        if transcript_result.confidence < settings.stt_language_threshold:
            return VoiceQueryResponse(
                transcript=transcript_result.text,
                transcript_confidence=transcript_result.confidence,
                response_text="क्षमा करें, मैं आपकी आवाज़ स्पष्ट रूप से नहीं सुन पाया। कृपया फिर से बोलें।",
            )

        # RAG
        user_profile = memory.get_profile(phone_number) if memory else {}
        history = memory.get_conversation_history(phone_number) if memory else []

        rag_response: QueryResponse = await rag.aquery(
            query_text=transcript_result.text,
            user_profile=user_profile,
            conversation_history=history,
        )

        # TTS
        t1 = time.perf_counter()
        tts_wav = await tts.synthesise(rag_response.response_text)
        audio_out_path = settings.audio_cache_dir / f"{rag_response.request_id}.ogg"
        convert_wav_to_ogg(tts_wav, audio_out_path)
        timings["tts_ms"] = (time.perf_counter() - t1) * 1000

        if memory:
            memory.add_turn(phone_number, transcript_result.text, rag_response.response_text)

        from backend.models.schemas import PipelineTimings
        all_timings = rag_response.timings.model_dump()
        all_timings.update(timings)

        return VoiceQueryResponse(
            transcript=transcript_result.text,
            transcript_confidence=transcript_result.confidence,
            response_text=rag_response.response_text,
            audio_response_path=str(audio_out_path),
            timings=PipelineTimings(**all_timings),
        )

    finally:
        ogg_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/dashboard", tags=["system"], include_in_schema=False)
async def dashboard_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/static/index.html")


# Mount dashboard static files
dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")


# ---------------------------------------------------------------------------
# Metrics API (consumed by dashboard)
# ---------------------------------------------------------------------------


@app.get("/api/metrics", tags=["observability"])
async def get_metrics():
    from backend.observability.metrics import compute_metrics
    return compute_metrics(settings.log_dir)


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=settings.fastapi_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
