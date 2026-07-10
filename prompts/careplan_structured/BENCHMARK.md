# Structured Care-Plan Prompt Benchmark

Current promoted version: `v2`

Benchmark:

- Test set: `careplan_prompt_benchmark_v1`
- Cases: IVIG, vancomycin, methotrexate, omalizumab, warfarin
- Provider/model: Anthropic `claude-sonnet-4-6`
- Date: June 24, 2026

| Version | Parse success | Accuracy | Coverage | F1 | Input tokens | Output tokens | Total tokens | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `v1` | 100.0% | 97.9% | 99.0% | 98.4% | 14,267 | 11,413 | 25,680 | Baseline |
| `v2` | 100.0% | 97.8% | 99.0% | 98.4% | 15,147 | 7,646 | 22,793 | Promoted |
| `v3` | 100.0% | 97.9% | 98.0% | 98.0% | 15,662 | 8,131 | 23,793 | Rejected |

`v2` was promoted because it preserved parse success and coverage with
effectively unchanged F1 while reducing total token use by 2,887 tokens
(11.2%) relative to `v1`.

`v3` was rejected because its additional prioritization instructions reduced
coverage and F1 while increasing token use relative to `v2`.

Detailed run reports are generated under `eval/prompt_benchmark_results/` and
are intentionally ignored by Git.
