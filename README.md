# Context-Memory-Management-for-Agents

## To be completed (for other parts)

## Healthy Diet Agent Benchmark (JSONL) + Grading Helpers

This pack contains:
- `healthy_diet_benchmark.jsonl` — 15 multi-turn conversations across 4 users (short/medium/long). Each object includes:
  - `id`, `user_id`, `session_id`, `length`, `required_tools`, `rotation_policy`,
  - `memory_dependencies` (intra/inter-session),
  - `turns` (assistant messages left blank for evaluation),
  - `ground_truth` (pass/fail criteria),
  - `notes` (what this test stresses).
- `grading_helpers.py` — tiny helpers to map ambiguous phrases (e.g., “my usual”) to hard anchors and verify variety and macro rules.

### “My usual” → anchors
Use these anchors when a user says “my usual”:
- u01 → 1800 kcal/day; ≥140 g protein; ≥30 g fiber; US units; peanut allergy; stove/oven only; Med/Mex; no blender.
- u02 → 1600 kcal/day; ≥110 g protein; metric; vegetarian; lactose-free; 12–20 fasting; microwave + rice cooker only.
- u03 → 2000 kcal/day; ≥150 g protein; US units; halal; low-glycemic; grill + air fryer; (fiber ≥30 g where specified).
- u04 → 2200 kcal/day; ≥130 g protein; fiber ≥30 g; US units; pescatarian; no tuna; Japanese/Thai.

### Variety rules
Every test object has a `rotation_policy`. Enforce:
- `no_repeat_days`: no exact dish repeats within that window.
- `max_same_primary_protein_per_week`: cap per primary protein across the plan.
- `ingredient_jaccard_max`: keep day-to-day ingredient overlap below this threshold.

### Tool usage
All tests require both tools. Log tool usage (e.g., `["search","calculator"]`) and validate with `check_tool_usage`.

### Running
- Load JSONL line-by-line. For each test, run your agent over the `turns` and capture outputs.
- Compare outputs against `ground_truth` using the helpers or your own evaluator.
- Provide any `menu_history` and prior-session context your harness maintains to check variety.