from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from prompts.manager import PromptManager, PromptRenderError


class PromptManagerTests(TestCase):
    def setUp(self):
        self.manager = PromptManager()

    def test_scenario_resolves_and_records_concrete_version(self):
        prompt = self.manager.render(
            workflow="careplan_review",
            scenario="evaluation",
            variables={
                "source_record": "record",
                "care_plan": "plan",
            },
        )

        self.assertEqual(prompt.requested_version, "v1")
        self.assertEqual(prompt.version, "v1")
        self.assertEqual(len(prompt.checksum), 12)
        self.assertIn("record", prompt.content)
        self.assertIn("plan", prompt.content)

    def test_current_alias_records_resolved_version(self):
        template = self.manager.load(
            workflow="careplan_generation",
            version="current",
        )

        self.assertEqual(template.requested_version, "current")
        self.assertEqual(template.resolved_version, "v3")

    def test_missing_variable_raises_clear_error(self):
        with self.assertRaisesRegex(PromptRenderError, "source_record"):
            self.manager.render(
                workflow="careplan_review",
                version="v1",
                variables={"care_plan": "plan"},
            )

    def test_current_copy_matches_aliased_version(self):
        prompts_dir = Path(__file__).resolve().parent
        current = (
            prompts_dir / "careplan_generation" / "current.txt"
        ).read_text()
        version = (
            prompts_dir / "careplan_generation" / "v3.txt"
        ).read_text()

        self.assertEqual(current, version)

    def test_structured_prompt_renders_rag_context(self):
        prompt = self.manager.render(
            workflow="careplan_structured",
            scenario="structured_generation",
            variables={
                "patient_record": "Patient A",
                "reference_material": "[chunk-1] Evidence",
            },
        )

        self.assertEqual(prompt.version, "v2")
        self.assertIn("Patient A", prompt.content)
        self.assertIn("[chunk-1] Evidence", prompt.content)

    def test_structured_current_copy_matches_v2(self):
        prompts_dir = Path(__file__).resolve().parent
        current = (
            prompts_dir / "careplan_structured" / "current.txt"
        ).read_text()
        version = (
            prompts_dir / "careplan_structured" / "v2.txt"
        ).read_text()

        self.assertEqual(current, version)

    def test_repair_prompt_current_matches_v1(self):
        prompts_dir = Path(__file__).resolve().parent
        current = (prompts_dir / "careplan_repair" / "current.txt").read_text()
        version = (prompts_dir / "careplan_repair" / "v1.txt").read_text()

        self.assertEqual(current, version)

    def test_explicit_version_can_be_loaded_from_custom_directory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "example").mkdir()
            (root / "example" / "v1.txt").write_text("Hello $name")
            (root / "config.yaml").write_text(
                """
defaults:
  example: v1
versions:
  example:
    v1: v1.txt
"""
            )

            manager = PromptManager(prompts_dir=root)
            prompt = manager.render("example", {"name": "Ada"})

            self.assertEqual(prompt.content, "Hello Ada")
            self.assertEqual(prompt.version, "v1")
