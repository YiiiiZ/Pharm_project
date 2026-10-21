#!/usr/bin/env python3
"""Embed rag/prepare_spl.py chunk output into Postgres + pgvector.

Dedupes by content_sha256 before calling the embedding API: the May 2026
prescription corpus has 91,357 chunks but only 70,662 unique content
hashes (repackaged labels repeat identical clinical text), so ~23% of
chunks are skipped and reuse an already-embedded vector instead of
paying to embed the same text twice.

Idempotent: chunk hashes already present in the table are skipped, so
re-running after an interruption only embeds what's missing.

Example:
    python -m rag.embed data/rag/spl_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from openai import OpenAI

from .vector_store import connect, init_schema

MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100


def iter_chunks(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def existing_content_hashes(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT content_sha256 FROM spl_chunk_embeddings")
        return {row[0] for row in cur.fetchall()}


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_corpus(
    *,
    input_path: Path,
    client: OpenAI,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> dict:
    with connect() as conn:
        init_schema(conn)
        done_hashes = existing_content_hashes(conn)

        chunks_by_hash: dict[str, list[dict]] = {}
        for chunk in iter_chunks(input_path):
            chunks_by_hash.setdefault(chunk["content_sha256"], []).append(chunk)

        to_embed = [h for h in chunks_by_hash if h not in done_hashes]
        if limit is not None:
            to_embed = to_embed[:limit]

        stats = {
            "total_chunks": sum(len(v) for v in chunks_by_hash.values()),
            "unique_content_hashes": len(chunks_by_hash),
            "already_embedded": len(done_hashes),
            "newly_embedded": 0,
            "rows_written": 0,
        }

        for batch_hashes in batched(to_embed, batch_size):
            texts = [chunks_by_hash[h][0]["text"] for h in batch_hashes]
            response = client.embeddings.create(model=MODEL, input=texts)

            with conn.cursor() as cur:
                for content_hash, item in zip(batch_hashes, response.data):
                    vector = item.embedding
                    for chunk in chunks_by_hash[content_hash]:
                        meta = chunk["metadata"]
                        cur.execute(
                            """
                            INSERT INTO spl_chunk_embeddings
                                (chunk_id, content_sha256, text, embedding,
                                 product_name, generic_name, section_title,
                                 qualified_section_title, document_type,
                                 source_archive, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                text = EXCLUDED.text,
                                metadata = EXCLUDED.metadata
                            """,
                            (
                                chunk["chunk_id"],
                                content_hash,
                                chunk["text"],
                                vector,
                                meta.get("product_name"),
                                meta.get("generic_name"),
                                meta.get("section_title"),
                                meta.get("qualified_section_title"),
                                meta.get("document_type"),
                                meta.get("source_archive"),
                                json.dumps(meta),
                            ),
                        )
                        stats["rows_written"] += 1
            conn.commit()

            stats["newly_embedded"] += len(batch_hashes)
            if stats["newly_embedded"] % 1000 < batch_size:
                print(
                    f"Embedded {stats['newly_embedded']}/{len(to_embed)} "
                    "unique chunks",
                    file=sys.stderr,
                )

        return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed SPL chunks into pgvector."
    )
    parser.add_argument(
        "input", type=Path, nargs="?", default=Path("data/rag/spl_chunks.jsonl")
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only embed the first N unvisited unique chunks (testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI()
    stats = embed_corpus(
        input_path=args.input,
        client=client,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
