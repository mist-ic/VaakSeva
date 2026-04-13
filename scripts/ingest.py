"""
Ingest script - process all scheme documents into Weaviate.

Usage:
  python scripts/ingest.py
  python scripts/ingest.py --dry-run          # just chunk, no upsert
  python scripts/ingest.py --schemes-dir path/to/custom/schemes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.rag.ingest import ingest_documents, ingest_to_weaviate


def main():
    parser = argparse.ArgumentParser(description="Ingest VaakSeva scheme documents")
    parser.add_argument(
        "--schemes-dir",
        type=Path,
        default=settings.schemes_dir,
        help="Directory containing scheme documents",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.processed_dir,
        help="Directory to write processed chunks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk and print stats without uploading to Weaviate",
    )
    args = parser.parse_args()

    print(f"Ingesting documents from: {args.schemes_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Ingest all documents
    chunks = ingest_documents(
        schemes_dir=args.schemes_dir,
        output_dir=args.output_dir,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )

    print(f"\nTotal chunks: {len(chunks)}")

    if args.dry_run:
        print("Dry run complete - no data uploaded to Weaviate")
        # Print stats
        lang_counts = {}
        scheme_counts = {}
        for chunk in chunks:
            lang = chunk.get("language", "unknown")
            scheme = chunk.get("scheme_id", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            scheme_counts[scheme] = scheme_counts.get(scheme, 0) + 1

        print("\nLanguage breakdown:")
        for lang, count in sorted(lang_counts.items()):
            print(f"  {lang}: {count} chunks")

        print("\nTop schemes by chunk count:")
        for scheme, count in sorted(scheme_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {scheme}: {count} chunks")
        return

    # Upload to Weaviate
    print("\nConnecting to Weaviate...")
    from backend.rag.retriever import get_weaviate_client, ensure_collection
    from backend.rag.embedder import get_embedder

    print("Loading embedder...")
    emb = get_embedder()

    client = get_weaviate_client()
    ensure_collection(client)

    print(f"Uploading {len(chunks)} chunks to Weaviate...")
    ingest_to_weaviate(chunks, emb, client, settings.weaviate_collection)

    client.close()
    print("Done!")


if __name__ == "__main__":
    main()
