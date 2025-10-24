# Context-Memory-Management-for-Agents

A comprehensive framework with context and memory management capabilities in AI agents using a healthy diet planning benchmark.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Context-Memory-Management Framework              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Discord Bot   │    │   Core Agent    │    │  Evaluation  │ │
│  │   (discordBot)  │    │    (agent.py)   │    │  Framework   │ │
│  │                 │    │                 │    │              │ │
│  │ • User Input    │───►│ • GeminiClient  │◄───│ • Baseline   │ │
│  │ • Mention       │    │ • Calculator    │    │   Evaluation │ │
│  │ • Logging       │    │ • Web Search    │    │ • Analysis   │ │
│  │                 │    │ • Rate Limiting │    │ • Reporting  │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                     │ │                       ▲     │
│           │                     │ │                       │     │
│           │ ┌───────────────────┘ │                       │     │
│           │ │                     │                       │     │
│           │ ▼                     ▼                       │     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   External      │    │   Tools &       │    │  Benchmark   │ │
│  │   APIs          │    │   Helpers       │    │  Dataset     │ │
│  │                 │    │                 │    │              │ │
│  │ • Gemini API    │◄───│ • Calculator    │◄───│ • 15 Tests   │ │
│  │ • Tavily Search │    │ • Search Tool   │    │ • 4 Users    │ │
│  │ • Discord API   │    │ • Grading Help  │    │ • Multi-turn │ │
│  │                 │    │ • Validation    │    │ • Memory Dep │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                        Data Flow                                │
│                                                                 │
│  User Input → Discord Bot → Core Agent → Response               │
│       ↓                                                         │
│  Benchmark → Evaluation → Analysis → Reports                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────────────────────┐
|                                Core Agent Data Flow                                   |
├───────────────────────────────────────────────────────────────────────────────────────┤
|                                                                                       |
|  ┌──────────────────────────────────────┐                                             |
|  | (1) Input                            |                                             |
|  |──────────────────────────────────────┤                                             |
|  | ─ Agent.chat_with_tools()            |                                             |
|  └──────────────────────────────────────┘                                             |
|                                                                                       |
|                                                                                       |
|  ┌─────────────────────────────────────┐            ┌──────────────────────────────┐  |
|  | (2) Tool Planning Loop              |            | (3) Tool Execution           |  |
|  |─────────────────────────────────────┤ tool call  |──────────────────────────────┤  |
|  | - Agent.determine_if_calc_needed()  |  request   | - CalculatorTool.calculate() |  |
|  | - Agent.refine_calc_expression()    |----------->| - TavilyClient.search()      |  |
|  | - Agent.determine_if_search_needed()|            | - RateLimiter.acquire()      |  |
|  | - Agent.refine_search_term()        |            | - _retry_with_backoff()      |  |
|  └─────────────────────────────────────┘            └──────────────────────────────┘  |
|      |         ^                                                             |        |
|      |         |  tool result & updated history                              |        |
|      |         └─────────────────────────────────────────────────────────────┘        |
|      |                                                                                |
|      | no more tool calls needed                                                      |
|      v                                                                                |
|  ┌─────────────────────────────────────┐                                              |
|  | (4) Generate Final Answer & Output  |                                              |
|  |─────────────────────────────────────┤                                              |
|  | - Agent.generate_response()         |                                              |
|  | - GeminiClient.infer()              |                                              |
|  └─────────────────────────────────────┘                                              |
|                                                                                       |
└───────────────────────────────────────────────────────────────────────────────────────┘






```

## Components

### Core Agent (`agent.py`)
- **GeminiClient**: Wrapper for Google's Gemini API with rate limiting and retry logic
- **CalculatorTool**: Safe arithmetic expression evaluator for nutrition calculations
- **Agent**: Main agent class with tool-calling capabilities (search + calculator)
- **RateLimiter**: Prevents API rate limit violations
- **SimpleLogger**: Structured logging for debugging and analysis

### Discord Bot (`discordBot.py`)
- Discord integration for real-time agent interaction
- Responds to mentions in "general" channel
- Logs user interactions with timestamps

### Evaluation Framework
- **`baseline_evaluation.py`**: Comprehensive baseline evaluation without context management
- **`analyze_results.py`**: Analysis tools for evaluation results with trend analysis
- **`grading_helpers.py`**: Helper functions for nutrition validation and user requirement checking

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

#### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` file with API keys:
   ```
   GEMINI_API_KEY=your_gemini_key
   TAVILY_API_KEY=your_tavily_key
   DISCORD_TOKEN=your_discord_token
   ```

#### Evaluation
- **Baseline evaluation**: `python baseline_evaluation.py`
- **Results analysis**: `python analyze_results.py`
- **Discord bot**: `python discordBot.py`

#### Benchmark Usage
- Load JSONL line-by-line. For each test, run your agent over the `turns` and capture outputs.
- Compare outputs against `ground_truth` using the helpers or your own evaluator.
- Provide any `menu_history` and prior-session context your harness maintains to check variety.

## Evaluation Metrics

### Core Metrics
- **Task Completion Rate**: Overall pass/fail rate across all tests
- **Nutrition Validation**: Proper macro/micro nutrient calculations
- **User Requirements**: Adherence to dietary restrictions and preferences
- **Context Handling**: Memory dependency resolution

### Advanced Metrics
- **Variety Rules**: No repeats, protein rotation, ingredient diversity
- **Timing Constraints**: Fasting windows, meal timing
- **Tool Usage**: Calculator and search tool utilization
- **Inter-session Memory**: Cross-session context retention

### Analysis Features
- Length-based performance analysis (short/medium/long conversations)
- User-specific requirement tracking
- Context weakness identification
- Trend analysis across conversation types