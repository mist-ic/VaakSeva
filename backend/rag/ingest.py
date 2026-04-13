"""
Document ingestion pipeline for VaakSeva.

Stages:
  1. Load text/PDF from data/schemes/
  2. Clean and normalise Unicode (NFC, virama, nukta)
  3. Detect language (Hindi vs English)
  4. Chunk with Hindi-aware separators
  5. Add metadata to each chunk
  6. Embed with configured embedder
  7. Upsert into Weaviate (BM25F index built automatically)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(path: Path) -> str:
    """Extract raw text from a PDF file using pdfplumber."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("pdfplumber not installed, falling back to PyPDF2")
        return _extract_pdf_fallback(path)


def _extract_pdf_fallback(path: Path) -> str:
    from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def load_document(path: Path) -> str:
    """Load a document (PDF or text) and return raw text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in {".txt", ".md", ".text"}:
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Unicode combining characters common in Devanagari
_DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F\u0966-\u096F]+")

# Common encoding artefacts
_ENCODING_ARTIFACTS = [
    (re.compile(r"â€™"), "'"),
    (re.compile(r"â€œ"), "\u201c"),
    (re.compile(r"â€\x9d"), "\u201d"),
    (re.compile(r"â€""), "\u2013"),
    (re.compile(r"\xa0"), " "),
]


def clean_text(text: str) -> str:
    """
    Normalise Unicode and fix common encoding problems.

    - NFC normalisation (canonical decomposition then canonical composition)
    - Fix known encoding artefacts from PDF extraction
    - Collapse multiple whitespace (but preserve paragraph breaks)
    - Strip leading/trailing whitespace from lines
    """
    # Fix common encoding artefacts
    for pattern, replacement in _ENCODING_ARTIFACTS:
        text = pattern.sub(replacement, text)

    # NFC normalisation: important for Devanagari combining characters
    text = unicodedata.normalize("NFC", text)

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]

    # Collapse 3+ consecutive blank lines to 2
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str:
    """
    Detect whether text is primarily Hindi (hi) or English (en).

    Uses a simple heuristic: count Devanagari characters.
    Falls back to langdetect for ambiguous cases.
    """
    devanagari_chars = len(_DEVANAGARI_RANGE.findall(text))
    total_chars = len(text.replace(" ", ""))

    if total_chars == 0:
        return "en"

    ratio = devanagari_chars / total_chars
    if ratio > 0.15:
        return "hi"

    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return "en"


# ---------------------------------------------------------------------------
# Hindi-aware chunking
# ---------------------------------------------------------------------------

# Hindi full stop (purna viram: ।) and English period
_HINDI_SEPARATORS = ["\n\n", "\n", "।", ".", "?", "!", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursive character-level chunker with Hindi-aware separators.

    Mirrors the behaviour of LangChain's RecursiveCharacterTextSplitter
    without the LangChain dependency.
    """
    if separators is None:
        separators = _HINDI_SEPARATORS

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator in order
    for sep in separators:
        if sep == "":
            # Last resort: split by characters
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i : i + chunk_size])
            return chunks

        if sep in text:
            splits = text.split(sep)
            chunks: list[str] = []
            current = ""

            for split in splits:
                segment = (current + sep + split).strip() if current else split.strip()
                if len(segment) <= chunk_size:
                    current = segment
                else:
                    if current:
                        chunks.append(current)
                    # Recurse into oversized segment
                    sub_chunks = chunk_text(
                        split, chunk_size, chunk_overlap, separators[separators.index(sep) + 1 :]
                    )
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""

            if current:
                chunks.append(current)

            return [c for c in chunks if c.strip()]

    return [text]


# ---------------------------------------------------------------------------
# Metadata extraction from filename
# ---------------------------------------------------------------------------


def parse_scheme_metadata(path: Path) -> dict:
    """
    Extract metadata from filename convention:
      <scheme_id>_<language>.txt
      e.g. pm_kisan_hi.txt, ayushman_bharat_en.txt
    """
    stem = path.stem  # e.g. "pm_kisan_hi"
    parts = stem.rsplit("_", 1)

    if len(parts) == 2 and parts[1] in {"hi", "en"}:
        scheme_id = parts[0]
        language = parts[1]
    else:
        scheme_id = stem
        language = None  # will be detected

    # Human-readable name: replace underscores with spaces, title-case
    scheme_name = scheme_id.replace("_", " ").title()

    return {
        "scheme_id": scheme_id,
        "scheme_name": scheme_name,
        "language": language,
        "source_path": str(path),
    }


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------


def ingest_documents(
    schemes_dir: Path,
    output_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Full ingestion pipeline.

    Returns list of chunk dicts ready for embedding and Weaviate upsert.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []

    doc_paths = sorted(
        [p for p in schemes_dir.iterdir() if p.suffix in {".txt", ".md", ".pdf"}]
    )

    logger.info("Found %d documents in %s", len(doc_paths), schemes_dir)

    for doc_path in doc_paths:
        try:
            # 1. Load
            raw_text = load_document(doc_path)

            # 2. Clean
            text = clean_text(raw_text)

            if len(text) < 50:
                logger.warning("Skipping too-short document: %s", doc_path.name)
                continue

            # 3. Metadata
            meta = parse_scheme_metadata(doc_path)

            # 4. Detect language if not in filename
            if meta["language"] is None:
                meta["language"] = detect_language(text)

            # 5. Chunk
            chunks = chunk_text(text, chunk_size, chunk_overlap)

            # 6. Enrich each chunk with metadata
            for idx, chunk_text_content in enumerate(chunks):
                chunk_id = hashlib.sha256(
                    f"{meta['scheme_id']}_{idx}_{chunk_text_content[:50]}".encode()
                ).hexdigest()[:16]

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "scheme_id": meta["scheme_id"],
                        "scheme_name": meta["scheme_name"],
                        "language": meta["language"],
                        "source_path": meta["source_path"],
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "content": chunk_text_content,
                    }
                )

            logger.info(
                "Ingested %s: %d chunks (lang=%s)",
                doc_path.name,
                len(chunks),
                meta["language"],
            )

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", doc_path.name, exc)

    # Save processed chunks to disk (for inspection / resuming)
    output_file = output_dir / "chunks.json"
    output_file.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d chunks to %s", len(all_chunks), output_file)

    return all_chunks


def ingest_to_weaviate(chunks: list[dict], embedder, weaviate_client, collection_name: str):
    """Embed chunks and upsert into Weaviate."""
    import weaviate

    collection = weaviate_client.collections.get(collection_name)
    texts = [c["content"] for c in chunks]

    logger.info("Embedding %d chunks...", len(texts))
    embeddings = embedder.embed_documents(texts)

    with collection.batch.dynamic() as batch:
        for chunk, embedding in zip(chunks, embeddings):
            batch.add_object(
                properties={
                    "chunk_id": chunk["chunk_id"],
                    "scheme_id": chunk["scheme_id"],
                    "scheme_name": chunk["scheme_name"],
                    "language": chunk["language"],
                    "content": chunk["content"],
                    "chunk_index": chunk["chunk_index"],
                    "source_path": chunk.get("source_path", ""),
                },
                vector=embedding,
            )

    logger.info("Upserted %d chunks into Weaviate collection '%s'", len(chunks), collection_name)

# Chunk overlap default 50 tokens, tuned for scheme documents
