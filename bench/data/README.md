# Benchmark datasets

All subsets are drawn with a fixed seed (42) from public, well-known benchmarks
for reproducibility. Each item has `id`, `domain`, and `expected_specialist`
(the node that should handle it, per the v0.1 router).

| File | Source | Size | Scoring |
|---|---|---|---|
| `code_humaneval.jsonl` | [openai/human-eval](https://github.com/openai/human-eval), `HumanEval.jsonl` | 60 / 164 | unit-test execution (pass@1) |
| `math_gsm8k.jsonl` | [openai/grade-school-math](https://github.com/openai/grade-school-math), `test.jsonl` | 60 / 1319 | numeric exact match |
| `general_mmlu.jsonl` | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu), `all/test` split | 70 across all 57 subjects | exact match (letter) |

## Regenerating

```bash
cd bench
.venv/bin/python3 -c "..."  # see git history for the exact fetch scripts used;
                             # re-run with the same seed (42) for identical subsets.
```
