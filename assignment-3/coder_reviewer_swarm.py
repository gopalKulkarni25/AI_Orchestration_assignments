import ast
import os
import pathlib
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

load_dotenv()

_langsmith_key = os.getenv("LANGSMITH_API_KEY")
if _langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = _langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = "ai-orchestrator-assignment-3"
    print("[LANGSMITH] Tracing enabled")

REPO_DIR = pathlib.Path(__file__).parent / "mock_buggy_repo"
MAX_ITERATIONS = 3

ARCHITECTURE_GUIDELINES = (
    "snake_case names; type hints on every parameter AND return value; "
    "docstring on every public function; no global mutable state; "
    "max 30 lines per function body; no bare except clauses; "
    "all imports at the top of the file."
)

FEATURE_REQUEST = (pathlib.Path(__file__).parent / "feature_request.txt").read_text(encoding="utf-8")


# ── State ────────────────────────────────────────────────────────────────────

class SwarmState(TypedDict):
    feature_request: str
    codebase_context: str
    generated_code: str
    lint_findings: str      # populated by linter_agent before reviewer sees the code
    review_feedback: str
    review_status: str
    iteration_count: int


# ── RAG ──────────────────────────────────────────────────────────────────────

def _extract_chunks(filepath: str) -> list[dict]:
    """Parse a Python file into one chunk per top-level function/class via AST."""
    source = pathlib.Path(filepath).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"  [AST] SyntaxError in {pathlib.Path(filepath).name}: {exc} — indexing as raw chunk")
        return [{"name": pathlib.Path(filepath).stem, "source": source[:2000], "file": filepath, "line": 0}]

    lines = source.splitlines()
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 20)
            chunks.append({
                "name": node.name,
                "source": "\n".join(lines[start:end]),
                "file": filepath,
                "line": node.lineno,
            })
    return chunks


def _build_chroma_index() -> chromadb.Collection:
    """Index every Python file in mock_buggy_repo into an in-memory ChromaDB collection."""
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.create_collection("codebase", embedding_function=ef)

    docs, ids, metas = [], [], []
    for py_file in sorted(REPO_DIR.glob("*.py")):
        for chunk in _extract_chunks(str(py_file)):
            chunk_id = f"{pathlib.Path(chunk['file']).stem}__{chunk['name']}__{chunk['line']}"
            docs.append(f"{chunk['name']}\n{chunk['source']}")
            ids.append(chunk_id)
            metas.append({"file": chunk["file"], "line": chunk["line"]})
            print(f"  [RAG] indexed: {pathlib.Path(chunk['file']).name}:{chunk['name']}")

    collection.add(documents=docs, ids=ids, metadatas=metas)
    print(f"[RAG] ChromaDB index ready — {len(docs)} chunks from {REPO_DIR.name}/\n")
    return collection


def retrieve_context(collection: chromadb.Collection, query: str, top_k: int = 3) -> str:
    """Return the top-k most relevant code chunks formatted for prompt injection."""
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    parts = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        parts.append(f"Chunk {i + 1} ({pathlib.Path(meta['file']).name}:{meta['line']})\n{doc}")
    return "=== Retrieved Codebase Context (RAG) ===\n\n" + "\n\n".join(parts)


# ── LLM ──────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ── Agent nodes ──────────────────────────────────────────────────────────────

def coder_agent(state: SwarmState) -> SwarmState:
    """Write (iteration 0) or revise (>0) Python code using codebase context and reviewer feedback."""
    it = state["iteration_count"]
    if it == 0:
        print("\n[CODER] Writing initial implementation...")
        user_content = (
            f"Codebase context:\n{state['codebase_context']}\n\n"
            f"Feature request:\n{state['feature_request']}"
        )
    else:
        print(f"\n[CODER] Iteration {it + 1}: revising based on feedback...")
        lint = f"\nLinter violations to fix:\n{state['lint_findings']}" if state["lint_findings"] else ""
        print(f"    review feedback : {state['review_feedback'][:120]}")
        user_content = (
            f"Feature request:\n{state['feature_request']}\n\n"
            f"Codebase context:\n{state['codebase_context']}\n\n"
            f"Your previous code:\n{state['generated_code']}\n\n"
            f"Fix ALL of this feedback:\n{state['review_feedback']}{lint}"
        )

    messages = [
        SystemMessage(
            content=(
                "You are an expert Python developer. "
                "Return ONLY raw Python code with no markdown fences or explanation. "
                f"Your code must follow: {ARCHITECTURE_GUIDELINES}"
            )
        ),
        HumanMessage(content=user_content),
    ]
    result = _llm.invoke(messages)
    code = result.content.strip()
    print(f"[CODER] Code written ({len(code.splitlines())} lines)")
    return {**state, "generated_code": code, "lint_findings": "", "review_status": "PENDING"}


def linter_agent(state: SwarmState) -> SwarmState:
    """Run ruff on the generated code and store structured violations in lint_findings."""
    print("\n[LINTER] Running ruff on generated code...")
    findings = ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(state["generated_code"])
            tmp_path = f.name

        result = subprocess.run(
            ["ruff", "check", "--output-format=text", tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        findings = result.stdout.strip().replace(tmp_path, "<generated_code>")
        if findings:
            print(f"[LINTER] {len(findings.splitlines())} violation(s) found")
        else:
            print("[LINTER] No violations — code is clean")
    except FileNotFoundError:
        print("[LINTER] ruff not installed — skipping lint check")
    except subprocess.TimeoutExpired:
        print("[LINTER] ruff timed out — skipping lint check")
    finally:
        if tmp_path:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    return {**state, "lint_findings": findings}


def _run_focused_review(args: tuple[str, str]) -> dict:
    """Run one focused sub-review (style or security) and return role/verdict/feedback."""
    role, code_block = args
    focus = {
        "style": (
            "code style and conventions only: naming (snake_case), type hints on every "
            "parameter and return value, docstrings on public functions, and imports at the top"
        ),
        "security": (
            "correctness and safety only: input validation (ValueError on empty input), "
            "no bare except clauses, no global mutable state, function body ≤ 30 lines"
        ),
    }[role]

    messages = [
        SystemMessage(
            content=(
                f"You are a strict code reviewer. Review ONLY for {focus}. "
                "List every violation found. "
                "End your response with EXACTLY one of:\nVERDICT: APPROVED\nVERDICT: REJECTED"
            )
        ),
        HumanMessage(content=f"Code to review:\n{code_block}"),
    ]
    result = _llm.invoke(messages)
    feedback = result.content.strip()
    verdict = "APPROVED" if "VERDICT: APPROVED" in feedback else "REJECTED"
    print(f"  [{role:>8} reviewer] {verdict}")
    return {"role": role, "verdict": verdict, "feedback": feedback}


def reviewer_agent(state: SwarmState) -> SwarmState:
    """Fan out to style + security reviewers in parallel; APPROVE only if both agree."""
    it = state["iteration_count"]
    print(f"\n[REVIEWER] Parallel review — style + security (iteration {it + 1})...")

    lint_note = f"\n\nLinter findings:\n{state['lint_findings']}" if state["lint_findings"] else ""
    code_block = state["generated_code"] + lint_note

    print("[FAN OUT] dispatching style + security reviewers concurrently...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_run_focused_review, [("style", code_block), ("security", code_block)]))

    style_r, security_r = results[0], results[1]
    rejections = [r for r in results if r["verdict"] == "REJECTED"]
    status = "REJECTED" if rejections else "APPROVED"
    combined_feedback = (
        " | ".join(f"[{r['role']}] {r['feedback']}" for r in rejections)
        if rejections
        else f"[style] {style_r['feedback']}"
    )

    print(
        f"[AGGREGATE] style={style_r['verdict']} | security={security_r['verdict']} -> {status}"
    )
    return {**state, "review_feedback": combined_feedback, "review_status": status, "iteration_count": it + 1}


def graceful_degrader(state: SwarmState) -> SwarmState:
    """Escalate to human review after MAX_ITERATIONS without approval."""
    print("\n[GRACEFUL DEGRADER] Max iterations reached — escalating to human review.")
    print(f"  iterations    : {state['iteration_count']}")
    print(f"  last feedback : {state['review_feedback'][:120]}")
    preview = state["generated_code"][:200].replace("\n", " ")
    print(f"  code preview  : {preview}...")
    return {**state, "review_status": "ESCALATED"}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_review(state: SwarmState) -> str:
    """3-way conditional edge: END (approved) / coder_agent (retry) / graceful_degrader (cap)."""
    if state["review_status"] == "APPROVED":
        print("\n[ROUTER] APPROVED -> END")
        return "end"
    if state["iteration_count"] < MAX_ITERATIONS:
        print(f"\n[ROUTER] REJECTED (iter {state['iteration_count']}/{MAX_ITERATIONS}) -> retry coder")
        return "coder_agent"
    print(f"\n[ROUTER] REJECTED after {state['iteration_count']} iters -> graceful_degrader")
    return "graceful_degrader"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    """Compile the coder → linter → reviewer(parallel) → {END | retry | escalate} LangGraph."""
    g = StateGraph(SwarmState)
    g.add_node("coder_agent", coder_agent)
    g.add_node("linter_agent", linter_agent)
    g.add_node("reviewer_agent", reviewer_agent)
    g.add_node("graceful_degrader", graceful_degrader)
    g.set_entry_point("coder_agent")
    g.add_edge("coder_agent", "linter_agent")       # linter runs before reviewer
    g.add_edge("linter_agent", "reviewer_agent")
    g.add_edge("graceful_degrader", END)
    g.add_conditional_edges(
        "reviewer_agent",
        route_after_review,
        {"end": END, "coder_agent": "coder_agent", "graceful_degrader": "graceful_degrader"},
    )
    return g.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  CODER-REVIEWER SWARM  (Assignment 3)")
    print("=" * 60 + "\n")

    collection = _build_chroma_index()
    rag_query = "process transactions list dict amount currency timestamp summary"
    codebase_context = retrieve_context(collection, rag_query)

    initial_state: SwarmState = {
        "feature_request": FEATURE_REQUEST,
        "codebase_context": codebase_context,
        "generated_code": "",
        "lint_findings": "",
        "review_feedback": "",
        "review_status": "PENDING",
        "iteration_count": 0,
    }

    app = build_graph()
    final = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print(f"Final status : {final['review_status']}")
    print(f"Iterations   : {final['iteration_count']}")
    if final["review_status"] == "APPROVED":
        print("\n--- Approved code ---\n" + final["generated_code"])
    else:
        print("\n--- Escalation report ---")
        print(f"Last feedback:\n{final['review_feedback']}")


if __name__ == "__main__":
    main()
