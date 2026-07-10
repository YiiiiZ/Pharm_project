#!/usr/bin/env python3
"""Run and compare versioned structured care-plan prompts.

Examples:
    python -m eval.prompt_benchmark --provider anthropic --prompt-version v1

    python -m eval.prompt_benchmark \
        --provider openai \
        --prompt-version v2 \
        --input-price-per-million 1.25 \
        --output-price-per-million 10

The fixed reference material intentionally isolates prompt-generation quality
from retrieval quality. Retrieval should have its own recall benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from care.care_plan_schema import CarePlan
from eval.score_eval import item_covered
from prompts import PromptManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "eval" / "dataset.json"
CONFIG_PATH = PROJECT_ROOT / "eval" / "prompt_benchmark.yaml"
RESULTS_DIR = PROJECT_ROOT / "eval" / "prompt_benchmark_results"


def load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def load_benchmark_cases(
    *,
    config_path: Path = CONFIG_PATH,
    dataset_path: Path = DATASET_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    config = yaml.safe_load(config_path.read_text())
    dataset = json.loads(dataset_path.read_text())
    cases_by_id = {case["id"]: case for case in dataset["cases"]}

    selected: list[dict[str, Any]] = []
    for selection in config["cases"]:
        case_id = selection["id"]
        if case_id not in cases_by_id:
            raise ValueError(f"Benchmark case not found: {case_id}")
        case = {
            "id": case_id,
            "metadata": cases_by_id[case_id]["metadata"],
            "input": cases_by_id[case_id]["input"],
            "golden": cases_by_id[case_id]["golden"],
            "benchmark_rationale": selection.get("rationale", ""),
        }
        selected.append(case)

    test_set_hash = hashlib.sha256(
        json.dumps(
            selected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return config, selected, test_set_hash


def build_patient_record(case: dict[str, Any]) -> str:
    inp = case["input"]
    patient = inp["patient"]
    provider = inp["provider"]
    order = inp["order"]
    return "\n".join(
        [
            f"Name: {patient.get('first_name', '')} {patient.get('last_name', '')}".strip(),
            f"MRN: {patient.get('mrn', '')}",
            f"Date of birth: {patient.get('dob', 'Not provided')}",
            f"Sex: {patient.get('sex', 'Not provided')}",
            f"Weight: {order.get('weight_kg', 'Not provided')} kg",
            f"Allergies: {order.get('allergies', 'None known')}",
            f"Provider: {provider.get('name', '')} (NPI: {provider.get('npi', '')})",
            f"Medication: {order.get('medication_name', '')}",
            (
                "Primary diagnosis: "
                f"{order.get('primary_diagnosis', '')} — "
                f"{order.get('primary_diagnosis_label', '')}"
            ),
            f"Additional diagnoses: {order.get('additional_diagnoses', 'None')}",
            "Medication history:",
            str(order.get("medication_history", "None provided")),
            "Clinical notes:",
            str(order.get("patient_records", "None provided")),
        ]
    )


def golden_items(case: dict[str, Any]) -> list[str]:
    golden = case["golden"]
    return (
        golden.get("problem_list", [])
        + golden.get("goals", [])
        + golden.get("interventions", [])
        + golden.get("monitoring", [])
    )


def build_reference_material(case: dict[str, Any]) -> str:
    """Create fixed evidence chunks from the frozen expert criteria."""
    sections = [
        ("problem", case["golden"].get("problem_list", [])),
        ("goal", case["golden"].get("goals", [])),
        ("intervention", case["golden"].get("interventions", [])),
        ("monitoring", case["golden"].get("monitoring", [])),
    ]
    chunks: list[str] = []
    index = 1
    for section, items in sections:
        for item in items:
            chunks.append(
                f'<reference id="{case["id"]}_{section}_{index:02d}" '
                f'type="{section}">\n{item}\n</reference>'
            )
            index += 1
    return "\n\n".join(chunks)


def render_prompt(
    case: dict[str, Any],
    *,
    prompt_version: str,
) -> Any:
    return PromptManager().render(
        workflow="careplan_structured",
        version=prompt_version,
        variables={
            "patient_record": build_patient_record(case),
            "reference_material": build_reference_material(case),
        },
    )


def run_anthropic(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    client: Any | None = None,
) -> dict[str, Any]:
    import anthropic

    client = client or anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        output_format=CarePlan,
    )
    parsed = getattr(response, "parsed_output", None)
    raw_output = "".join(
        block.text
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == "text"
    )
    return {
        "parsed": parsed,
        "raw_output": raw_output,
        "response_id": getattr(response, "id", ""),
        "stop_reason": getattr(response, "stop_reason", ""),
        "usage": _usage_dict(getattr(response, "usage", None)),
    }


def run_openai(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    client: Any | None = None,
) -> dict[str, Any]:
    from openai import OpenAI

    client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=CarePlan,
        max_output_tokens=max_tokens,
    )
    return {
        "parsed": getattr(response, "output_parsed", None),
        "raw_output": getattr(response, "output_text", "") or "",
        "response_id": getattr(response, "id", ""),
        "stop_reason": getattr(response, "status", ""),
        "usage": _usage_dict(getattr(response, "usage", None)),
    }


def generated_clinical_items(plan: CarePlan) -> list[str]:
    items: list[str] = []
    for problem in plan.problems:
        items.append(
            " ".join(
                [
                    problem.title,
                    problem.description,
                    *problem.evidence,
                ]
            )
        )
    for goal in plan.goals:
        items.append(
            " ".join(
                value
                for value in [
                    goal.description,
                    goal.target,
                    goal.timeframe,
                    goal.related_problem,
                ]
                if value
            )
        )
    for intervention in plan.interventions:
        items.append(
            " ".join(
                value
                for value in [
                    intervention.action,
                    intervention.rationale,
                    intervention.monitoring,
                ]
                if value
            )
        )
    return items


def score_plan(case: dict[str, Any], plan: CarePlan) -> dict[str, Any]:
    expected = golden_items(case)
    generated = generated_clinical_items(plan)
    plan_text = "\n".join(generated)

    coverage_results = [
        {"item": item, "covered": item_covered(item, plan_text)}
        for item in expected
    ]
    covered_count = sum(result["covered"] for result in coverage_results)

    generated_results = []
    for generated_item in generated:
        matched = any(
            item_covered(expected_item, generated_item)
            for expected_item in expected
        )
        generated_results.append(
            {"item": generated_item, "matches_golden": matched}
        )
    matched_generated = sum(
        result["matches_golden"] for result in generated_results
    )

    accuracy = (
        matched_generated / len(generated) if generated else 0.0
    )
    coverage = covered_count / len(expected) if expected else 0.0
    f1 = (
        2 * accuracy * coverage / (accuracy + coverage)
        if accuracy + coverage
        else 0.0
    )
    return {
        "accuracy": round(accuracy, 4),
        "coverage": round(coverage, 4),
        "f1": round(f1, 4),
        "generated_item_count": len(generated),
        "matched_generated_item_count": matched_generated,
        "golden_item_count": len(expected),
        "covered_golden_item_count": covered_count,
        "coverage_items": coverage_results,
        "generated_items": generated_results,
    }


def aggregate_results(
    case_results: list[dict[str, Any]],
    *,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> dict[str, Any]:
    attempted = len(case_results)
    parsed = [result for result in case_results if result["parse_success"]]
    input_tokens = sum(
        result["usage"].get("input_tokens", 0) or 0
        for result in case_results
    )
    output_tokens = sum(
        result["usage"].get("output_tokens", 0) or 0
        for result in case_results
    )
    aggregate = {
        "case_count": attempted,
        "parse_success_count": len(parsed),
        "parse_success_rate": round(len(parsed) / attempted, 4)
        if attempted
        else 0.0,
        "accuracy_macro": _mean_metric(parsed, "accuracy"),
        "coverage_macro": _mean_metric(parsed, "coverage"),
        "f1_macro": _mean_metric(parsed, "f1"),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "tokens_total": input_tokens + output_tokens,
        "input_tokens_average": round(input_tokens / attempted, 2)
        if attempted
        else 0.0,
        "output_tokens_average": round(output_tokens / attempted, 2)
        if attempted
        else 0.0,
    }
    if (
        input_price_per_million is not None
        and output_price_per_million is not None
    ):
        aggregate["estimated_cost_usd"] = round(
            input_tokens / 1_000_000 * input_price_per_million
            + output_tokens / 1_000_000 * output_price_per_million,
            6,
        )
        aggregate["pricing"] = {
            "input_price_per_million": input_price_per_million,
            "output_price_per_million": output_price_per_million,
        }
    else:
        aggregate["estimated_cost_usd"] = None
    return aggregate


def run_benchmark(
    *,
    provider: str,
    model: str,
    prompt_version: str,
    max_tokens: int,
    input_price_per_million: float | None = None,
    output_price_per_million: float | None = None,
    client: Any | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    config, cases, test_set_hash = load_benchmark_cases()
    started_at = datetime.now(timezone.utc)
    case_results: list[dict[str, Any]] = []
    prompt_checksum = ""

    for case in cases:
        rendered = render_prompt(case, prompt_version=prompt_version)
        prompt_checksum = rendered.checksum
        try:
            if provider == "anthropic":
                response = run_anthropic(
                    prompt=rendered.content,
                    model=model,
                    max_tokens=max_tokens,
                    client=client,
                )
            elif provider == "openai":
                response = run_openai(
                    prompt=rendered.content,
                    model=model,
                    max_tokens=max_tokens,
                    client=client,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            parsed = response["parsed"]
            parse_success = isinstance(parsed, CarePlan)
            score = score_plan(case, parsed) if parse_success else None
            error = None
        except Exception as exc:
            response = {
                "parsed": None,
                "raw_output": "",
                "response_id": "",
                "stop_reason": "",
                "usage": {},
            }
            parse_success = False
            score = None
            error = {"type": type(exc).__name__, "message": str(exc)}

        case_results.append(
            {
                "case_id": case["id"],
                "medication": case["metadata"]["medication"],
                "parse_success": parse_success,
                "score": score,
                "usage": response["usage"],
                "response_id": response["response_id"],
                "stop_reason": response["stop_reason"],
                "raw_output": response["raw_output"],
                "parsed_output": (
                    parsed.model_dump(mode="json")
                    if parse_success
                    else None
                ),
                "error": error,
            }
        )

    result = {
        "benchmark": config["name"],
        "test_set_hash": test_set_hash,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "prompt_workflow": "careplan_structured",
        "prompt_version": prompt_version,
        "prompt_checksum": prompt_checksum,
        "metrics_definition": config["metrics"],
        "aggregate": aggregate_results(
            case_results,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        ),
        "cases": case_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / (
            f"{timestamp}_{provider}_{_slug(model)}_{prompt_version}.json"
        )
    result["result_path"] = str(output_path)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    return result


def find_previous_run(
    current: dict[str, Any],
    *,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in results_dir.glob("*.json"):
        if str(path) == current.get("result_path"):
            continue
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            result.get("test_set_hash") == current.get("test_set_hash")
            and result.get("provider") == current.get("provider")
            and result.get("model") == current.get("model")
        ):
            candidates.append((path.stat().st_mtime, result))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def compare_runs(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline is None:
        return None
    keys = [
        "parse_success_rate",
        "accuracy_macro",
        "coverage_macro",
        "f1_macro",
        "input_tokens_total",
        "output_tokens_total",
        "tokens_total",
        "estimated_cost_usd",
    ]
    comparison = {
        "baseline_prompt_version": baseline["prompt_version"],
        "current_prompt_version": current["prompt_version"],
        "baseline_tokens_total": baseline["aggregate"].get("tokens_total"),
        "current_tokens_total": current["aggregate"].get("tokens_total"),
        "deltas": {},
    }
    for key in keys:
        old = baseline["aggregate"].get(key)
        new = current["aggregate"].get(key)
        comparison["deltas"][key] = (
            round(new - old, 6)
            if isinstance(old, (int, float))
            and isinstance(new, (int, float))
            else None
        )
    return comparison


def assess_comparison_gates(
    comparison: dict[str, Any] | None,
    gates: dict[str, Any],
) -> dict[str, Any] | None:
    if comparison is None:
        return None
    failures: list[str] = []
    deltas = comparison["deltas"]
    checks = [
        (
            "parse_success_rate",
            float(gates["parse_success_rate_max_regression"]),
        ),
        ("accuracy_macro", float(gates["accuracy_max_regression"])),
        ("coverage_macro", float(gates["coverage_max_regression"])),
    ]
    for metric, allowed_regression in checks:
        delta = deltas.get(metric)
        if delta is not None and delta < -allowed_regression:
            failures.append(
                f"{metric} regressed by {abs(delta):.4f}; "
                f"allowed {allowed_regression:.4f}"
            )

    baseline_tokens = comparison.get("baseline_tokens_total")
    current_tokens = comparison.get("current_tokens_total")
    if baseline_tokens and current_tokens is not None:
        increase_ratio = (current_tokens - baseline_tokens) / baseline_tokens
        allowed_increase = float(gates["token_total_max_increase_ratio"])
        if increase_ratio > allowed_increase:
            failures.append(
                f"tokens_total increased by {increase_ratio:.1%}; "
                f"allowed {allowed_increase:.1%}"
            )
    return {"passed": not failures, "failures": failures}


def print_summary(
    result: dict[str, Any],
    comparison: dict[str, Any] | None,
    gate_result: dict[str, Any] | None = None,
) -> None:
    aggregate = result["aggregate"]
    print(
        f"\nPrompt {result['prompt_version']} | "
        f"{result['provider']} / {result['model']}"
    )
    print("-" * 72)
    print(
        f"Parse success: {aggregate['parse_success_rate']:.1%} "
        f"({aggregate['parse_success_count']}/{aggregate['case_count']})"
    )
    print(f"Accuracy:      {aggregate['accuracy_macro']:.1%}")
    print(f"Coverage:      {aggregate['coverage_macro']:.1%}")
    print(f"F1:            {aggregate['f1_macro']:.1%}")
    print(
        f"Tokens:        {aggregate['input_tokens_total']} input + "
        f"{aggregate['output_tokens_total']} output = "
        f"{aggregate['tokens_total']}"
    )
    if aggregate["estimated_cost_usd"] is not None:
        print(f"Est. cost:     ${aggregate['estimated_cost_usd']:.6f}")

    if comparison:
        print(
            f"\nCompared with prompt "
            f"{comparison['baseline_prompt_version']}:"
        )
        for key, delta in comparison["deltas"].items():
            if delta is not None:
                print(f"  {key}: {delta:+.4f}")
    else:
        print("\nNo comparable previous run found.")
    if gate_result:
        print(
            "\nComparison gate: "
            + ("PASS" if gate_result["passed"] else "FAIL")
        )
        for failure in gate_result["failures"]:
            print(f"  - {failure}")
    print(f"\nSaved: {result['result_path']}")


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("input_tokens", "output_tokens", "total_tokens")
        if getattr(usage, key, None) is not None
    }


def _mean_metric(results: list[dict[str, Any]], metric: str) -> float:
    values = [result["score"][metric] for result in results if result["score"]]
    return round(statistics.mean(values), 4) if values else 0.0


def _slug(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in value
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one structured care-plan prompt version."
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
    )
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when configured comparison gates fail",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    model = args.model or (
        os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        if args.provider == "anthropic"
        else os.environ.get("OPENAI_MODEL", "gpt-5.5")
    )
    result = run_benchmark(
        provider=args.provider,
        model=model,
        prompt_version=args.prompt_version,
        max_tokens=args.max_tokens,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
        output_path=args.output,
    )
    baseline = (
        json.loads(args.baseline.read_text())
        if args.baseline
        else find_previous_run(result)
    )
    comparison = compare_runs(result, baseline)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    gate_result = assess_comparison_gates(
        comparison,
        config["comparison_gates"],
    )
    result["comparison"] = comparison
    result["comparison_gate"] = gate_result
    Path(result["result_path"]).write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    print_summary(result, comparison, gate_result)
    if (
        args.fail_on_regression
        and gate_result is not None
        and not gate_result["passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
