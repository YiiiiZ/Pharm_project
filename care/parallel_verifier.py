"""Parallel verification strategies for care-plan hallucination detection.

Three drop-in Evaluator implementations that conform to generation_loop.Evaluator:

  RacingVerifier     – fan-out to N evaluators, keep the most thorough result
                       (or the fastest, if select="fastest")
  VotingVerifier     – majority vote on validity; ties resolve to invalid;
                       all findings are merged and deduplicated by path
  SectioningVerifier – split output into focused sections, evaluate each in
                       isolation, merge conservatively:
                         any section invalid  → overall invalid
                         any finding high-risk → escalate_to_pharmacist=True

All three are thread-safe and use ThreadPoolExecutor for I/O-bound LLM calls.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Callable, Mapping, Sequence

from .generation_loop import Evaluator, HallucinationFinding, HallucinationReport

SectionSplitter = Callable[[str], dict[str, str]]

_SECTION_PATTERNS: list[tuple[str, str]] = [
    ("problems", r"problem"),
    ("goals", r"goal"),
    ("interventions", r"intervention"),
    ("monitoring", r"monitor"),
]


def split_care_plan_sections(text: str) -> dict[str, str]:
    """Split a plain-text care plan into named sections.

    Sections are found by scanning for lines whose content contains one of the
    four recognised keywords (case-insensitive, handles markdown bold/headers).
    Returns {"full": text} when no headers are found, so callers always get a
    non-empty dict and a default_evaluator can handle the fallback.
    """
    header_re = re.compile(
        r"(?m)^[^\n]*(?:" +
        "|".join(p for _, p in _SECTION_PATTERNS) +
        r")[^\n]*$",
        re.IGNORECASE,
    )

    positions: list[tuple[int, str]] = []
    for m in header_re.finditer(text):
        line = m.group(0).lower()
        for key, pattern in _SECTION_PATTERNS:
            if re.search(pattern, line):
                if not positions or positions[-1][1] != key:
                    positions.append((m.start(), key))
                break

    if not positions:
        return {"full": text}

    sections: dict[str, str] = {}
    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[key] = text[start:end].strip()
    return sections


class RacingVerifier:
    """Fan-out to multiple evaluators; return the result with the most findings.

    With select="most_findings" (default) we take the evaluator that catches
    the most potential hallucinations — the conservative choice for clinical
    verification. With select="fastest" we take whichever evaluator finishes
    first (useful when models have different latency profiles).

    All submitted futures run to completion; 'fastest' means we use the first
    result, not that we abort the others (LLM API calls cannot be cancelled).
    """

    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        *,
        select: str = "most_findings",
        max_workers: int | None = None,
    ) -> None:
        if not evaluators:
            raise ValueError("RacingVerifier requires at least one evaluator")
        if select not in ("most_findings", "fastest"):
            raise ValueError("select must be 'most_findings' or 'fastest'")
        self._evaluators = list(evaluators)
        self._select = select
        self._max_workers = max_workers or len(self._evaluators)

    def __call__(self, output: str) -> HallucinationReport:
        if len(self._evaluators) == 1:
            return self._evaluators[0](output)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(ev, output) for ev in self._evaluators]

            if self._select == "fastest":
                first = next(as_completed(futures))
                result = first.result()
                # remaining futures finish in the background (pool.shutdown waits)
                return result

        # most_findings: collect all results, pick the worst
        reports = [f.result() for f in futures]
        return max(reports, key=lambda r: r.hallucination_count)


class VotingVerifier:
    """Run all evaluators in parallel; validity decided by strict majority vote.

    On a tie (even number of evaluators, split evenly), the result is invalid —
    the conservative choice for a clinical verification tool.
    Findings from all evaluators are merged and deduplicated by path.
    """

    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        *,
        max_workers: int | None = None,
    ) -> None:
        if not evaluators:
            raise ValueError("VotingVerifier requires at least one evaluator")
        self._evaluators = list(evaluators)
        self._max_workers = max_workers or len(self._evaluators)

    def __call__(self, output: str) -> HallucinationReport:
        if len(self._evaluators) == 1:
            return self._evaluators[0](output)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(ev, output) for ev in self._evaluators]
            reports = [f.result() for f in futures]

        valid_votes = sum(1 for r in reports if r.valid)
        valid = valid_votes > len(reports) / 2  # strict majority; ties → invalid

        merged_findings = _deduplicate_findings(
            f for r in reports for f in r.findings
        )
        total_count = sum(r.hallucination_count for r in reports)

        return HallucinationReport(
            valid=valid,
            hallucination_count=total_count,
            findings=merged_findings,
        )


class SectioningVerifier:
    """Evaluate each care-plan section in isolation; merge conservatively.

    section_evaluators maps section names ("problems", "goals",
    "interventions", "monitoring") to dedicated evaluators. Sections not
    listed fall back to default_evaluator; sections with neither are skipped.

    The splitter defaults to split_care_plan_sections. Provide a custom one
    if the care plan uses a non-standard format.

    Conservative merge rules:
      - overall valid only if every evaluated section passed
      - escalate_to_pharmacist=True if any finding carries risk_level="high"
      - each finding's path is prefixed with its section name for traceability
    """

    def __init__(
        self,
        section_evaluators: Mapping[str, Evaluator],
        *,
        default_evaluator: Evaluator | None = None,
        splitter: SectionSplitter | None = None,
        max_workers: int | None = None,
    ) -> None:
        if not section_evaluators and default_evaluator is None:
            raise ValueError(
                "Provide at least one section_evaluator or a default_evaluator"
            )
        self._section_evals = dict(section_evaluators)
        self._default_eval = default_evaluator
        self._splitter = splitter or split_care_plan_sections
        self._max_workers = max_workers

    def __call__(self, output: str) -> HallucinationReport:
        sections = self._splitter(output)

        tasks: dict[str, tuple[str, Evaluator]] = {}
        for section, text in sections.items():
            evaluator = self._section_evals.get(section) or self._default_eval
            if evaluator is not None:
                tasks[section] = (text, evaluator)

        if not tasks:
            return HallucinationReport(valid=True, hallucination_count=0)

        workers = self._max_workers or len(tasks)
        section_reports: dict[str, HallucinationReport] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_section = {
                pool.submit(ev, text): name
                for name, (text, ev) in tasks.items()
            }
            for future in as_completed(future_to_section):
                name = future_to_section[future]
                section_reports[name] = future.result()

        return _merge_section_reports(section_reports)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deduplicate_findings(
    findings: object,
) -> list[HallucinationFinding]:
    seen: set[str] = set()
    result: list[HallucinationFinding] = []
    for finding in findings:  # type: ignore[union-attr]
        if finding.path not in seen:
            seen.add(finding.path)
            result.append(finding)
    return result


def _merge_section_reports(
    section_reports: dict[str, HallucinationReport],
) -> HallucinationReport:
    all_valid = True
    total_count = 0
    all_findings: list[HallucinationFinding] = []
    escalate = False

    for section, report in section_reports.items():
        if not report.valid:
            all_valid = False
        total_count += report.hallucination_count
        for finding in report.findings:
            prefixed = replace(finding, path=f"{section}.{finding.path}")
            all_findings.append(prefixed)
            if finding.risk_level == "high":
                escalate = True

    return HallucinationReport(
        valid=all_valid,
        hallucination_count=total_count,
        findings=all_findings,
        escalate_to_pharmacist=escalate,
    )
