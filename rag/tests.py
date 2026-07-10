from unittest import TestCase

from rag.prepare_spl import chunk_blocks, estimate_tokens


class ChunkingTests(TestCase):
    def test_short_blocks_remain_together(self):
        chunks = chunk_blocks(
            ["First clinical paragraph.", "Second clinical paragraph."],
            target_tokens=50,
            max_tokens=80,
            overlap_tokens=10,
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("First clinical paragraph", chunks[0])
        self.assertIn("Second clinical paragraph", chunks[0])

    def test_long_section_splits_under_maximum(self):
        blocks = [
            "Renal function should be monitored before treatment. " * 35,
            "Stop treatment if acute kidney injury is suspected. " * 35,
        ]
        chunks = chunk_blocks(
            blocks,
            target_tokens=100,
            max_tokens=140,
            overlap_tokens=20,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(estimate_tokens(chunk) <= 140 for chunk in chunks))

    def test_overlap_repeats_context(self):
        blocks = [
            "Alpha " * 35,
            "Beta " * 35,
            "Gamma " * 35,
        ]
        chunks = chunk_blocks(
            blocks,
            target_tokens=50,
            max_tokens=80,
            overlap_tokens=40,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("Beta", chunks[0])
        self.assertIn("Beta", chunks[1])
