"""Round-trip retrieval tests against a live Postgres + pgvector instance.

Skipped automatically unless both OPENAI_API_KEY and VECTOR_DATABASE_URL
are set, matching the pattern used by care/test_query_routing.py.
"""

from __future__ import annotations

import json
import os
import unittest
import uuid

from openai import OpenAI

from .embed import MODEL
from .retrieve import format_reference_material, retrieve_chunks
from .vector_store import connect, init_schema

REQUIRES_LIVE_STACK = not (
    os.environ.get("OPENAI_API_KEY") and os.environ.get("VECTOR_DATABASE_URL")
)


@unittest.skipIf(
    REQUIRES_LIVE_STACK, "requires OPENAI_API_KEY and VECTOR_DATABASE_URL"
)
class RetrievalRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = OpenAI()
        self.chunk_id = f"test-{uuid.uuid4().hex[:12]}"
        self.text = (
            "Vancomycin requires renal dosing and trough-level monitoring."
        )
        embedding = (
            self.client.embeddings.create(model=MODEL, input=[self.text])
            .data[0]
            .embedding
        )
        with connect() as conn:
            init_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO spl_chunk_embeddings
                        (chunk_id, content_sha256, text, embedding,
                         product_name, generic_name, section_title,
                         qualified_section_title, document_type,
                         source_archive, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        self.chunk_id,
                        f"sha-{self.chunk_id}",
                        self.text,
                        embedding,
                        "Vancomycin",
                        "Vancomycin",
                        "Dosage and Administration",
                        "Dosage and Administration",
                        "TEST",
                        "test-archive",
                        json.dumps({}),
                    ),
                )
            conn.commit()

    def tearDown(self) -> None:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM spl_chunk_embeddings WHERE chunk_id = %s",
                    (self.chunk_id,),
                )
            conn.commit()

    def test_retrieves_semantically_close_chunk(self) -> None:
        results = retrieve_chunks(
            "How should vancomycin be dosed for renal patients?",
            client=self.client,
            top_k=3,
        )
        self.assertTrue(any(r.chunk_id == self.chunk_id for r in results))

    def test_format_reference_material_joins_chunk_text(self) -> None:
        results = retrieve_chunks(
            "vancomycin renal dosing", client=self.client, top_k=1
        )
        material = format_reference_material(results)
        self.assertIn("Vancomycin", material)


if __name__ == "__main__":
    unittest.main()
