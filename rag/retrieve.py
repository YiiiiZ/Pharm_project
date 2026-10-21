#!/usr/bin/env python3
"""Query-time retrieval: patient/medication text -> top-k SPL chunks.

Embeds the query with the same model used at index time
(text-embedding-3-small) and runs a cosine-distance nearest-neighbor
search in pgvector. Results are over-fetched and deduplicated by
content_sha256, since repackaged labels can produce near-identical
top hits for the same clinical text (see README.md).

The background worker passes retrieved evidence to the structured generation graph.

Example:
    python -m rag.retrieve "vancomycin renal dosing" --top-k 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from openai import OpenAI

from .embed import MODEL
from .vector_store import connect

DEFAULT_TOP_K = 5
OVERFETCH_FACTOR = 3


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    content_sha256: str
    product_name: str | None
    section_title: str | None
    distance: float
    metadata: dict = field(default_factory=dict)


def retrieve_chunks(
    query: str,
    *,
    client: OpenAI | None = None,
    top_k: int = DEFAULT_TOP_K,
    product_name: str | None = None,
) -> list[RetrievedChunk]:
    client = client or OpenAI(timeout=60, max_retries=1)
    embedding = client.embeddings.create(model=MODEL, input=[query]).data[0].embedding

    filters = []
    params: list = [embedding]
    if product_name:
        filters.append("(strpos(lower(coalesce(product_name, '')), lower(%s)) > 0 OR strpos(lower(coalesce(generic_name, '')), lower(%s)) > 0)")
        params.extend([product_name, product_name])
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(top_k * OVERFETCH_FACTOR)

    sql = f"""
        SELECT chunk_id, text, content_sha256, product_name, section_title,
               embedding <=> %s::vector AS distance, metadata
        FROM spl_chunk_embeddings
        {where_clause}
        ORDER BY distance
        LIMIT %s
    """

    with connect() as conn:
        conn.execute("SET LOCAL statement_timeout = '60s'")
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    seen_hashes: set[str] = set()
    results: list[RetrievedChunk] = []
    for chunk_id, text, content_sha256, product, section, distance, metadata in rows:
        if content_sha256 in seen_hashes:
            continue
        seen_hashes.add(content_sha256)
        results.append(
            RetrievedChunk(chunk_id, text, content_sha256, product, section, distance, metadata)
        )
        if len(results) >= top_k:
            break
    return results


def format_reference_material(chunks: list[RetrievedChunk]) -> str:
    """Join retrieved chunks into the reference_material string expected by
    care.structured_llm.generate_structured_care_plan."""
    return "\n\n---\n\n".join(
        f"[{chunk.chunk_id}] {chunk.product_name or 'Unknown product'} — {chunk.section_title or 'Label section'}\n{chunk.text}"
        for chunk in chunks
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve SPL chunks for a query.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--product-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = retrieve_chunks(
        args.query, top_k=args.top_k, product_name=args.product_name
    )
    for chunk in chunks:
        print(f"[{chunk.distance:.4f}] {chunk.product_name} | {chunk.section_title} ({chunk.chunk_id})")
        print(chunk.text[:200])
        print()


if __name__ == "__main__":
    main()
