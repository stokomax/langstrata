# Worker Server Architecture Options

> Design exploration: running subagent workers in their own LangGraph Agent Server instance(s) instead of in-process `worker.ainvoke()` calls.

## Current Architecture (In-Process)

Today all worker graphs are built and invoked **in the same Python process** as the supervisor `StateGraph`:

```
supervisor_graph.py build_supervisor_graph()
  │
  ├─ supervisor_node  (LLM call)
  ├─ router           (deterministic, returns Send[])
  └─ wrapper nodes    (one per worker type)
       │
       ├─ _get_or_build_worker("macro_analysis")
       │    └─ imports macro_analysis._build_agent()
       │    └─ calls worker.ainvoke(input, config)   ← in-process
       │
       └─ extract structured_response → WorkerEnvelope
            └─ return {"collected_outputs": [envelope]}
```

Key properties:
- **Single process, single event loop:** workers share the same heap, import cache, connection pool
- **Zero serialization:** state passes as live Python objects (Pydantic models, BaseMessage instances)
- **`_worker_cache` dict:** compiled graphs cached after first build (~1-2s each)
- **No separate checkpoint lineage:** worker runs are not persisted as independent LangGraph threads
- **All 9 workers are already individually registrable** as LangGraph Server graphs (each `workers/*.py` exports `graph = make_graph(...)`) but they are only used as importable libraries

## Approach 1 — Remote Workers on the Same Server

Register all 9 workers as separate assistants in `langgraph.json`, then replace in-process `.ainvoke()` with HTTP calls via `langgraph_sdk`.

### `langgraph.json` Changes

```json
{
  "python_version": "3.13",
  "dependencies": ["."],
  "graphs": {
    "supervisor": "./src/driftline/core/agents/supervisor_graph.py:make_graph",
    "daily_signal_analysis": "./src/driftline/core/agents/workers/daily_signal_analysis.py:graph",
    "daily_signal_news":     "./src/driftline/core/agents/workers/daily_signal_news.py:graph",
    "macro_analysis":        "./src/driftline/core/agents/workers/macro_analysis.py:graph",
    "cross_asset":           "./src/driftline/core/agents/workers/cross_asset.py:graph",
    "macro_brief":           "./src/driftline/core/agents/workers/macro_brief.py:graph",
    "macro_chart":           "./src/driftline/core/agents/workers/macro_chart.py:graph",
    "correlation_analysis":  "./src/driftline/core/agents/workers/correlation_analysis.py:graph",
    "correlation_brief":     "./src/driftline/core/agents/workers/correlation_brief.py:graph",
    "correlation_chart":     "./src/driftline/core/agents/workers/correlation_chart.py:graph"
  },
  "env": ".env",
  "api_version": "~=0.11.1",
  "dockerfile_lines": ["ENV UV_INDEX=pypiserver=http://host.docker.internal:8082"]
}
```
### `config.py` Toggle

```python
class Settings(BaseSettings):
    worker_mode: Literal["in_process", "remote"] = "in_process"
    # Already configurable via LANGGRAPH_API_URL env var:
    langgraph_api_url: str = Field(
        default="http://localhost:2024",
        validation_alias=AliasChoices("LANGGRAPH_API_URL", "langgraph_api_url"),
    )
```

### `supervisor_graph.py` — Remote Invocation

Replace `_get_or_build_worker()` + `_invoke_worker_traced()` with:

```python
from langgraph_sdk import get_client

async def _invoke_worker_remote(
    agent_name: str,
    as_of: str,
    task_id: str,
    description: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Invoke a worker via the LangGraph SDK HTTP API (stateless run)."""
    settings = get_settings()
    client = get_client(url=settings.langgraph_api_url)

    input_data = {
        "messages": [{"role": "user", "content": description}],
    }
    run_config = {"configurable": {}}
    as_of_date = config.get("configurable", {}).get("as_of_date")
    if as_of_date:
        run_config["configurable"]["as_of_date"] = as_of_date
    scan_run_id = config.get("configurable", {}).get("scan_run_id")
    if scan_run_id is not None:
        run_config["configurable"]["scan_run_id"] = scan_run_id

    # POST /runs/wait — stateless, single request/response
    return await client.runs.wait(
        thread_id=None,
        assistant_id=agent_name,   # matches langgraph.json graph name
        input=input_data,
        config=run_config,
    )
```

Branch in `_make_wrapper()`:

```python
if settings.worker_mode == "remote":
    result = await _invoke_worker_remote(agent_name, as_of, task_id, description, config)
else:
    result = await _invoke_worker_traced(agent_name, as_of, task_id, description, config)
```

The rest of the wrapper — envelope construction, persist handlers, `_summarize_structured` — stays **identical**.
### What Changes vs What Stays

| Module | Change |
|---|---|
| `langgraph.json` | Add 9 worker graph registrations |
| `config.py` | Add `worker_mode` setting |
| `pyproject.toml` | Promote `langgraph-sdk` from dev to runtime dep |
| `supervisor_graph.py` | Add `_invoke_worker_remote()` + toggle branch |
| `docker-compose.langgraph.yml` | No change (already has `LANGGRAPH_API_URL=http://localhost:8000`) |
| All `workers/*.py` | **Unchanged** — same `_build_agent()`, same middleware, same prompts |
| `_PERSIST_HANDLERS` | **Unchanged** — persist still runs inline in wrapper |
| `_worker_schema()`, `_WORKER_NODE_MAP` | **Unchanged** |
| Router logic | **Unchanged** — still reads `pending_tasks`, returns `Send[]` |
| `SupervisorState`, `WorkerEnvelope` | **Unchanged** |

### The RemoteGraph Warning

LangChain docs warn against using `RemoteGraph` on the **same deployment** due to deadlock risk. Approach 1 uses the **raw SDK `client.runs.wait()`**, not `RemoteGraph`, and the server is configured with `N_JOBS_PER_WORKER=40`, providing ample concurrent process slots. The supervisor run and worker runs execute in separate process pool slots, not blocking each other.

## Approach 2 — Separate Worker Server

A second LangGraph Server deployment (separate Docker compose) dedicated to worker graphs:

- Adds `DRIFTLINE_WORKER_API_URL` config setting
- Worker server gets its own `langgraph.json` (workers only, no supervisor)
- The `_invoke_worker_remote()` function targets the worker server URL instead of the same server
- Advantages: independent scaling, no same-deployment concerns, can update worker code independently
- Disadvantages: more infrastructure (second postgres/redis if dedicated), more Docker complexity
## Nodes, Graphs, and Assistants — The Layered Model

```
  Assistant  ───  A named, configured, HTTP-addressable *instance* of a graph
       ↑
   Graph  ───────  A compiled state machine (nodes + edges + state + reducers)
       ↑
   Node  ────────  A single function or runnable, reading/writing one slice of state
```

| Concept | What it is | In Driftline | Analogy |
|---|---|---|---|
| **Node** | A function that reads/writes state | `_supervisor_node`, `_make_wrapper("macro_analysis")` | A single step in a recipe |
| **Graph** | Compiled state machine of nodes + edges | `build_supervisor_graph()` → `CompiledGraph` | The full recipe book |
| **Assistant** | A named, configured, HTTP-addressable instance of a graph | Default assistant created by Agent Server for each `langgraph.json` graph entry | A specific chef following the recipe with specific ingredients |

When you register a graph in `langgraph.json`, the Agent Server creates a **default assistant** with `assistant_id` = graph name. You can create additional assistants from the same graph with different config (models, prompts) via `client.assistants.create()`.

## Message Passing: In-Process vs Remote

### Boundary 1: Supervisor → Router → Wrapper Nodes (via `Send`)

**Identical in both modes.** The router returns `Send("macro_wrapper", {"as_of": str, "description": str})`. Only primitive strings. No serialization.

### Boundary 2: Wrapper → Worker Graph

**In-process:** `await worker.ainvoke({"messages": [...], config=...})` — passes live Python dicts and `RunnableConfig` directly. No serialization.

**Remote:** `await client.runs.wait(assistant_id=..., input={"messages": [...]}, config=...)` — input is serialized to JSON for the HTTP request, deserialized on the server. The worker graph receives the same dict structure.

### Boundary 3: Worker → Wrapper

**In-process:** Worker returns final state dict containing live objects:
- `messages`: list of `HumanMessage`, `AIMessage`, `ToolMessage` (Pydantic models with `.content`, `.tool_calls`, `.response_metadata`)
- `structured_response`: live `WorkerOutput` / `BriefContent` / `DailySignalNewsContent` Pydantic model

**Remote:** Worker returns final state dict with **already-deserialized** values:
- `messages`: list of **dicts** (not live `BaseMessage` objects)
- `structured_response`: plain **dict** (not a Pydantic model) — needs `model_validate()` to convert back

### Boundary 4: Wrapper → Supervisor State (collected_outputs)

**Identical in both modes.** The wrapper extracts `structured_response`, calls `model_dump(mode="json")`, wraps in `WorkerEnvelope`, returns `{"collected_outputs": [envelope.model_dump(mode="json")]}`.

### Key Observation

The wrapper **does not use** the full `messages` list in either mode. It only inspects `structured_response` (for status/summary) and `messages[-1].content` (fallback summary). Everything else is discarded. So the remote path loses nothing of value.
