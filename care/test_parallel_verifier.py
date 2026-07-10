from unittest import TestCase

from care.generation_loop import HallucinationFinding, HallucinationReport
from care.parallel_verifier import (
    RacingVerifier,
    SectioningVerifier,
    VotingVerifier,
    split_care_plan_sections,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLEAN = HallucinationReport(valid=True, hallucination_count=0)
DIRTY_1 = HallucinationReport(
    valid=False,
    hallucination_count=1,
    findings=[HallucinationFinding(path="$.problems[0]", message="bad claim", risk_level="low")],
)
DIRTY_2 = HallucinationReport(
    valid=False,
    hallucination_count=2,
    findings=[
        HallucinationFinding(path="$.goals[0]", message="bad goal", risk_level="low"),
        HallucinationFinding(path="$.interventions[0]", message="bad action", risk_level="medium"),
    ],
)
HIGH_RISK = HallucinationReport(
    valid=False,
    hallucination_count=1,
    findings=[HallucinationFinding(path="$.monitoring[0]", message="invented threshold", risk_level="high")],
)


def make_evaluator(report: HallucinationReport):
    def evaluator(output: str) -> HallucinationReport:
        return report
    return evaluator


SAMPLE_CARE_PLAN = """\
**Problem List**
Active diagnoses relevant to this order.

**Goals**
Measurable therapeutic targets.

**Pharmacist Interventions**
Specific actions pharmacist will take.

**Monitoring Plan**
Parameters and thresholds to monitor.
"""


# ---------------------------------------------------------------------------
# split_care_plan_sections
# ---------------------------------------------------------------------------

class SplitCarePlanSectionsTests(TestCase):
    def test_splits_four_standard_sections(self):
        sections = split_care_plan_sections(SAMPLE_CARE_PLAN)
        self.assertIn("problems", sections)
        self.assertIn("goals", sections)
        self.assertIn("interventions", sections)
        self.assertIn("monitoring", sections)

    def test_section_content_does_not_bleed_into_next(self):
        sections = split_care_plan_sections(SAMPLE_CARE_PLAN)
        self.assertNotIn("Goals", sections["problems"])
        self.assertNotIn("Monitoring", sections["interventions"])

    def test_returns_full_when_no_headers(self):
        plain = "Just some text with no section headers."
        sections = split_care_plan_sections(plain)
        self.assertEqual(sections, {"full": plain})

    def test_handles_plain_text_headers(self):
        plain = "Problem List\nsome problem\nGoals\nsome goal\n"
        sections = split_care_plan_sections(plain)
        self.assertIn("problems", sections)
        self.assertIn("goals", sections)


# ---------------------------------------------------------------------------
# RacingVerifier
# ---------------------------------------------------------------------------

class RacingVerifierTests(TestCase):
    def test_most_findings_returns_worst_report(self):
        racing = RacingVerifier([
            make_evaluator(CLEAN),
            make_evaluator(DIRTY_1),
            make_evaluator(DIRTY_2),
        ], select="most_findings")
        result = racing("some output")
        self.assertEqual(result.hallucination_count, 2)
        self.assertFalse(result.valid)

    def test_most_findings_clean_when_all_pass(self):
        racing = RacingVerifier([
            make_evaluator(CLEAN),
            make_evaluator(CLEAN),
        ], select="most_findings")
        result = racing("some output")
        self.assertTrue(result.valid)

    def test_fastest_returns_a_valid_report(self):
        racing = RacingVerifier([
            make_evaluator(CLEAN),
            make_evaluator(DIRTY_1),
        ], select="fastest")
        result = racing("some output")
        self.assertIsInstance(result, HallucinationReport)

    def test_single_evaluator_passes_through(self):
        racing = RacingVerifier([make_evaluator(DIRTY_1)])
        result = racing("some output")
        self.assertIs(result, DIRTY_1)

    def test_invalid_select_raises(self):
        with self.assertRaises(ValueError):
            RacingVerifier([make_evaluator(CLEAN)], select="random")

    def test_empty_evaluators_raises(self):
        with self.assertRaises(ValueError):
            RacingVerifier([])


# ---------------------------------------------------------------------------
# VotingVerifier
# ---------------------------------------------------------------------------

class VotingVerifierTests(TestCase):
    def test_majority_valid_returns_valid(self):
        voting = VotingVerifier([
            make_evaluator(CLEAN),
            make_evaluator(CLEAN),
            make_evaluator(DIRTY_1),
        ])
        result = voting("some output")
        self.assertTrue(result.valid)

    def test_majority_invalid_returns_invalid(self):
        voting = VotingVerifier([
            make_evaluator(CLEAN),
            make_evaluator(DIRTY_1),
            make_evaluator(DIRTY_2),
        ])
        result = voting("some output")
        self.assertFalse(result.valid)

    def test_tie_resolves_to_invalid(self):
        voting = VotingVerifier([
            make_evaluator(CLEAN),
            make_evaluator(DIRTY_1),
        ])
        result = voting("some output")
        self.assertFalse(result.valid)

    def test_merges_findings_and_deduplicates_by_path(self):
        # DIRTY_1 and DIRTY_2 have different paths — all four should appear
        voting = VotingVerifier([
            make_evaluator(DIRTY_1),
            make_evaluator(DIRTY_2),
        ])
        result = voting("some output")
        paths = {f.path for f in result.findings}
        self.assertEqual(paths, {"$.problems[0]", "$.goals[0]", "$.interventions[0]"})

    def test_total_hallucination_count_is_sum(self):
        voting = VotingVerifier([
            make_evaluator(DIRTY_1),
            make_evaluator(DIRTY_2),
        ])
        result = voting("some output")
        self.assertEqual(result.hallucination_count, 3)  # 1 + 2

    def test_single_evaluator_passes_through(self):
        voting = VotingVerifier([make_evaluator(DIRTY_1)])
        result = voting("some output")
        self.assertIs(result, DIRTY_1)

    def test_empty_evaluators_raises(self):
        with self.assertRaises(ValueError):
            VotingVerifier([])


# ---------------------------------------------------------------------------
# SectioningVerifier
# ---------------------------------------------------------------------------

class SectioningVerifierTests(TestCase):
    def test_all_sections_valid_returns_valid(self):
        verifier = SectioningVerifier(
            {
                "problems": make_evaluator(CLEAN),
                "goals": make_evaluator(CLEAN),
                "interventions": make_evaluator(CLEAN),
                "monitoring": make_evaluator(CLEAN),
            }
        )
        result = verifier(SAMPLE_CARE_PLAN)
        self.assertTrue(result.valid)
        self.assertFalse(result.escalate_to_pharmacist)

    def test_any_section_invalid_fails_overall(self):
        verifier = SectioningVerifier(
            {
                "problems": make_evaluator(CLEAN),
                "goals": make_evaluator(DIRTY_1),
                "interventions": make_evaluator(CLEAN),
                "monitoring": make_evaluator(CLEAN),
            }
        )
        result = verifier(SAMPLE_CARE_PLAN)
        self.assertFalse(result.valid)

    def test_high_risk_finding_sets_escalate_flag(self):
        verifier = SectioningVerifier(
            {
                "problems": make_evaluator(CLEAN),
                "goals": make_evaluator(CLEAN),
                "interventions": make_evaluator(CLEAN),
                "monitoring": make_evaluator(HIGH_RISK),
            }
        )
        result = verifier(SAMPLE_CARE_PLAN)
        self.assertTrue(result.escalate_to_pharmacist)

    def test_low_risk_finding_does_not_escalate(self):
        verifier = SectioningVerifier(
            {"problems": make_evaluator(DIRTY_1)},
            default_evaluator=make_evaluator(CLEAN),
        )
        result = verifier(SAMPLE_CARE_PLAN)
        self.assertFalse(result.escalate_to_pharmacist)

    def test_findings_are_prefixed_with_section_name(self):
        verifier = SectioningVerifier(
            {"goals": make_evaluator(DIRTY_1)},
            default_evaluator=make_evaluator(CLEAN),
        )
        result = verifier(SAMPLE_CARE_PLAN)
        # DIRTY_1 finding path is "$.problems[0]" — prefixed with "goals."
        self.assertTrue(
            any(f.path.startswith("goals.") for f in result.findings)
        )

    def test_default_evaluator_handles_unlisted_sections(self):
        verifier = SectioningVerifier(
            {"problems": make_evaluator(CLEAN)},
            default_evaluator=make_evaluator(DIRTY_1),
        )
        result = verifier(SAMPLE_CARE_PLAN)
        # goals/interventions/monitoring all hit default → invalid
        self.assertFalse(result.valid)

    def test_empty_output_with_no_headers_uses_default(self):
        verifier = SectioningVerifier(
            {},
            default_evaluator=make_evaluator(CLEAN),
        )
        result = verifier("Some unstructured text")
        self.assertTrue(result.valid)

    def test_no_matching_evaluators_returns_clean_pass(self):
        verifier = SectioningVerifier(
            {"problems": make_evaluator(DIRTY_1)},
            # no default_evaluator, so goals/interventions/monitoring are skipped
        )
        plain = "Some text with no section headers"
        # falls back to {"full": ...}; no evaluator for "full" → skipped → valid
        result = verifier(plain)
        self.assertTrue(result.valid)
        self.assertEqual(result.hallucination_count, 0)

    def test_no_evaluators_raises(self):
        with self.assertRaises(ValueError):
            SectioningVerifier({})
