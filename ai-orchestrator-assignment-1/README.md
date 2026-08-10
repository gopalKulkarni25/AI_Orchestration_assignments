# Assignment 1 — AI Orchestrator Code Pipeline

A LangGraph-based multi-agent pipeline that takes Jira-style tickets and automatically generates, compiles, tests, and self-corrects Python code.

---

## Architecture

The pipeline is a directed graph of 5 nodes:

```
Planner → Coder → Compiler → Test Runner
                     ↑              ↓
              Error Handler ←───────┘ (on failure, up to 3 retries)
```

| Node | Role |
|---|---|
| **Planner** | Produces a structured JSON implementation plan from the ticket |
| **Coder** | Writes Python code from the plan; on retry, incorporates the error |
| **Compiler** | Runs the generated code in an isolated temp directory |
| **Test Runner** | Generates and runs pytest tests with coverage reporting |
| **Error Handler** | Increments retry count and routes back to Coder |

---

## Stretch Goals

- **Structured Planner output** — Planner uses `response_format={"type": "json_object"}` and returns `{"summary": "...", "steps": [...]}` stored in `plan_json` state field
- **LangSmith tracing** — auto-enabled when `LANGSMITH_API_KEY` is set; traces appear under project `ai-orchestrator-assignment-1`
- **Coverage reporting** — Test Runner runs `pytest --cov=solution --cov-report=term-missing` and reports per-line coverage

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this directory:

```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_pt_...   # optional — enables LangSmith tracing
```

---

## Usage

Edit the `__main__` block in `code_pipeline.py` to select a ticket:

```python
ticket = TICKETS[0]   # DATA-001: ISO date parser
ticket = TICKETS[1]   # DATA-002: Binary search
ticket = TICKETS[2]   # DATA-003: CSV row counter
```

Set `simulate_failure: True` to force a first-attempt compiler error and demo the retry loop.

Run:

```bash
python code_pipeline.py
```

Results are written to `output/<TICKET-ID>_result.txt` containing the plan (JSON + formatted), generated code, compiler output, and test output with coverage.

---

## Sample Tickets

| ID | Description |
|---|---|
| DATA-001 | ISO 8601 date parser returning a structured dict |
| DATA-002 | Iterative binary search on a sorted list |
| DATA-003 | CSV row counter with header-skip and error handling |

---

## Project Structure

```
ai-orchestrator-assignment-1/
├── code_pipeline.py        # Main pipeline
├── requirements.txt
├── output/
│   ├── DATA-002_result.txt
│   ├── DATA-003_result.txt
│   └── LangSmithOutput/    # LangSmith trace screenshots
```
