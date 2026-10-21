"""Offline checks for incremental medication indexing; no API calls."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag.embed import embed_corpus


class MedicationCacheTests(unittest.TestCase):
    def run_index(self, chunks, *, existing_ids=(), cached=None):
        conn = MagicMock()
        cached = cached or {}
        conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [(h,) for h in cached]
        conn.execute.side_effect = [
            SimpleNamespace(fetchall=lambda: [(i,) for i in existing_ids]),
            SimpleNamespace(fetchall=lambda: list(cached.items())),
        ]
        client = MagicMock()
        client.embeddings.create.return_value.data = [SimpleNamespace(index=0, embedding=[0.1, 0.2])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chunks.jsonl'
            path.write_text('\n'.join(json.dumps(chunk) for chunk in chunks))
            with patch('rag.embed.connect') as connect, patch('rag.embed.init_schema'):
                connect.return_value.__enter__.return_value = conn
                stats = embed_corpus(input_path=path, client=client, product_name='metformin')
        return stats, client, conn

    def chunk(self, id='one', hash='hash', generic='metformin'):
        return dict(chunk_id=id, content_sha256=hash, text='label text', metadata={'product_name': 'Brand', 'generic_name': generic})

    def test_generic_match_and_unrelated_products_excluded(self):
        stats, client, _ = self.run_index([self.chunk(), self.chunk('other', 'other-hash', 'vancomycin')])
        self.assertEqual(stats['rows_written'], 1)
        self.assertEqual(stats['newly_embedded'], 1)
        self.assertEqual(client.embeddings.create.call_args.kwargs['input'], ['label text'])

    def test_existing_content_reused_for_new_label_metadata(self):
        stats, client, conn = self.run_index([self.chunk('new-label')], cached={'hash': [0.1, 0.2]})
        client.embeddings.create.assert_not_called()
        self.assertEqual(stats['rows_written'], 1)
        self.assertEqual(stats['newly_embedded'], 0)
        insert = conn.cursor.return_value.__enter__.return_value.execute.call_args.args[1]
        self.assertEqual(insert[0], 'new-label')

    def test_completed_index_does_not_reembed_or_rewrite(self):
        stats, client, _ = self.run_index([self.chunk()], existing_ids=['one'])
        client.embeddings.create.assert_not_called()
        self.assertEqual(stats['rows_written'], 0)
