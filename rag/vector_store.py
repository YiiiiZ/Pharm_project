"""Postgres + pgvector connection and schema for SPL chunk embeddings.

This is a dedicated vector store, separate from the Django/SQLite
operational database (Patient/Provider/Order). At the current corpus
scale (~91K chunks) a dedicated vector database isn't needed for
performance; the split exists because the app's operational DB is still
SQLite in this MVP, and pgvector requires Postgres. See README.md for
the migration note.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small


def _database_url() -> str:
    url = os.environ.get("VECTOR_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "VECTOR_DATABASE_URL is not set. Start the vector_db compose "
            "service and set VECTOR_DATABASE_URL (see .env.example)."
        )
    return url


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(_database_url(), connect_timeout=10)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def init_schema(conn: psycopg.Connection) -> None:
    """Create the chunk-embedding table and indexes if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS spl_chunk_embeddings (
                chunk_id TEXT PRIMARY KEY,
                content_sha256 TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL,
                product_name TEXT,
                generic_name TEXT,
                section_title TEXT,
                qualified_section_title TEXT,
                document_type TEXT,
                source_archive TEXT,
                metadata JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS spl_chunk_embeddings_content_sha256_idx "
            "ON spl_chunk_embeddings (content_sha256)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS spl_chunk_embeddings_hnsw "
            "ON spl_chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()
