
![langstrata fin logo](docs/screenshots/fin_logo_v2.png)

# Langstrata

Langstrata is a blueprint for deploying multi-agent solutions
with Langhchain that scale from day-one. It has three design pillars:

* The Supervisor-worker pattern is all you need.
* A curated subset of Langhain/Langgraph features is all you need.
* The supervisor and workers run as independent Agent Servers. 

The included demonstration software and scripts walk you through three deployment
scenarios for langstrata. The first is a canonical single agent server running _langgraph dev_. Its there to establish the graph coding pattern that spans all three deployment scenarios. The second deployment scenario is dual _langgaph dev_ servers running the same two graphs but as independent agent servers. It's purley for instructional and prototyping purposes; A learning crutch before graduating to the self-hosted dual-agent server scenario. 

The demonstration software and docker compose files can serve as templates to build your own dual-agent server solutions. They are more complicated, as a result, than your typical demonstration software. 

## The Building Blocks

LangChain and LangGraph already provide all the tools needed to build
this pattern. langstrata is a curated subset of those tools and shows a
deployment topology where the supervisor and workers run on separate
servers. The databases(PostgreSQL for checkpoints, Redis for streaming) are shared. For the supervisor-worker pattern: a single supervisor agent decides which worker to invoke and synthesizes their results.  Workers do not make routing decisions.

Agents are comprised of three building blocks:

- `create_agent()` to build individual workers.
- `create_deep_agent()` with `AsyncSubAgent` specs to build a
  supervisor that delegates tasks.
- `StateGraph` with `Send` to fan work out within a single graph.

The supervisor API (`create_deep_agent` with `AsyncSubAgent`) routes
work to worker graphs. The data transport between agent servers is either ASGI or HTTP. ASGI is exclusively for running a single _langgraph dev_ server. The agent code is the same in both topologies. Only the URL passed to
`AsyncSubAgent` changes.  The infrastructure (two `langgraph.json`
files, two server processes) absorbs the difference.

| Element | Role | Reference |
|---|---|---|
| `create_agent()` | A single LLM-powered agent with tools and a system prompt. | [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents) |
| `create_deep_agent()` + `AsyncSubAgent(name=..., url=?)` | A coordinating supervisor that delegates tasks to subagents.  The `url` field is optional.  When you omit it, the subagent runs on the same server (ASGI transport).  When you set it to an HTTP URL, the subagent runs on a remote server via the Agent Protocol. | [Async Subagents docs](https://docs.langchain.com/oss/python/deepagents/async-subagents) |
| `StateGraph` + `Send` | Parallel fan-out within a single graph.  A dispatch function returns one `Send` per item, and the graph executor runs them concurrently.  Results accumulate via a reducer such as `operator.add`. | [Map-Reduce and the Send API](https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api) |

These three building blocks are enough. Start here; Specialized variants that leverage all the optimizations Langchain/Langgraph supports can be reached for later.

## Two Agent Patterns

langstrata demonstrates two agent patterns built from the three primitives.

### Pattern A: Individual Worker Delegation

The supervisor launches one worker at a time. The LLM decides which worker fits each task.

```
create_deep_agent(subagents=[
    AsyncSubAgent(name="researcher", graph_id="researcher"),
    AsyncSubAgent(name="coder",      graph_id="coder"),
    AsyncSubAgent(name="analyst",    graph_id="analyst"),
])
```

In ASGI mode, each worker runs on its own thread within the single
server.  In HTTP mode, the supervisor runs on one server and all
workers run together on a separate worker server.  This is the
canonical async subagents pattern from the Deep Agents documentation.

### Pattern B: Send API Map-Reduce on the Worker

:sparkles: Coming Soon!

A reference implementation on the worker server is the canonical
LangGraph example from the [Map-Reduce and the Send API
page](https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api).
The worker wraps each `Send` target to call the same compiled
`create_agent()` graphs that Pattern A uses.
The supervisor launches a single subagent called `map_reduce_worker`.
That subagent points at a `StateGraph` on the worker server that
implements the canonical map-reduce pattern from the LangGraph docs.

```
create_deep_agent(subagents=[
    AsyncSubAgent(name="SendAPI", graph_id="map_reduce_worker"),
])
```

The `map_reduce_worker` graph on the worker server receives a list of tasks,
fans them out via `Send` to per-worker wrapper nodes, collects the
results with a reducer, and returns a consolidated batch summary.
The internal fan-out is deterministic: no LLM is involved in the
routing, only the graph executor.

This pattern is useful when you know ahead of time that a set of
tasks should run in parallel and you want the fan-out to happen in a
single super-step on the worker server, without the supervisor's LLM
making individual launch decisions for each one.

## Why Two Agent Servers?

On a single server, supervisor turns and worker runs share the same
pool of reources.  When the supervisor launches several workers,
those workers consume resources that the supervisor potentially needs for its next turn. Under load, workers can delay the supervisor indefinitely.

langstrata's two-server topology gives each server its own execution
pool. Guarding the supervisor from unintended worker agent activity. The supervisor
proceeds on its own server regardless of how
many workers are running on the worker server.  

The following _langgraph.json_ configuration files are included:

| File | Graphs registered | Server |
|---|---|---|
| `langgraph.json` | supervisor, researcher, coder, analyst | Single server & supervisor (port 2024). Start here. |
| `langgraph.workers.json` | researcher, coder, analyst | Worker server (port 2025) |

An included _justfile_ defines recipes for building and deploying the three deployment
scenarios. The [Deployment Examples](#deployment-examples) section provides a tutorial demonstrating usage of the justfile
recipes.

Below is a sample of what the justfile recipes are actually executing. The agent code remains the same. Environment variables and multiple langgraph.json files define the deployment.

```bash
# Start: everything on one server
AGENT_SERVER_MODE=asgi langgraph dev --config langgraph.json --port 2024

# Graduate: split into two servers
AGENT_SERVER_MODE=http langgraph dev --config langgraph.workers.json --port 2025
AGENT_SERVER_MODE=http AGENT_SERVER_WORKER_API_URL=http://localhost:2025 \
  langgraph dev --config langgraph.json --port 2024
```

When you run the dual-agent server, the supervisor server serves the
`supervisor` agent server on port 2024 and the worker server serves the
`researcher`, `coder`, and `analyst` subagents on port
2025.  The supervisor launches subagents by making HTTP calls to the
worker server.  Each subagent run on the worker server creates its own
checkpoint lineage, independent of the supervisor's thread.

The use of two _langgraph dev_ servers is to help you become familiar with the infrastructure changes before graduating to the self-hosted Docker deployment shown in [Example 3](#example-3-self-hosted-docker-production). The _docker-compose.yml_ provides a complete working example — it launches two independent agent servers, resolves port number conflicts and creates a shared database.

## Deployment Examples

The following examples graduate from a single server to a self-hosted Docker
deployment.  The same agent code powers all three — only the infrastructure
changes.  Every command is a `just` recipe defined in the included
[justfile](https://github.com/casey/just).

### Prerequisites

**Platform support.** langstrata is right at home on Linux, and it also
runs great inside the Windows Subsystem for Linux 2 (WSL2). macOS is
likely fine too — it just hasn't been tested here yet. Native Windows
(running directly on the Windows host) is not currently supported, so
if you're on Windows, WSL2 is the way to go.

Start by copying the example environment file and setting the required variables:

```bash
cp .env.example .env
```

The `.env.example` file documents every variable the project uses:

```ini
# Required: LLM provider API keys
# ANTHROPIC_API_KEY=sk-ant-...
# DEEPSEEK_API_KEY=sk-...

# LLM model for all agents (supervisor + workers).
# Default from config.py: anthropic:claude-sonnet-5
AGENT_SERVER_MODEL=anthropic:claude-sonnet-4-6

# Optional: LangSmith tracing
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_...
# LANGSMITH_PROJECT=langstrata
```

**Required variables:**

- `ANTHROPIC_API_KEY` — API key for Anthropic Claude models (needed unless you use DeepSeek)
- `DEEPSEEK_API_KEY` — API key for DeepSeek models (needed unless you use Anthropic)
- `AGENT_SERVER_MODEL` — Model identifier, e.g. `anthropic:claude-sonnet-5`, `deepseek:reasoner`, or `claude-haiku-4-5-20251001`

**Optional variables (LangSmith tracing is opt-in — not required by LangChain or LangGraph):**

- `LANGSMITH_TRACING` — set to `true` to enable tracing (the opt-in switch; leave unset to run without tracing).
- `LANGSMITH_API_KEY` — API key for [LangSmith](https://smith.langchain.com) tracing. Set this only if you want to inspect traces (see [LangChain observability docs](https://docs.langchain.com/oss/python/langchain/observability#enable-tracing)).
- `LANGSMITH_PROJECT` — LangSmith project name (default: `langstrata`)

Once your `.env` is ready, install dependencies and verify the configuration:

```bash
uv sync
uv run python main.py    # verify config loads
```

### Example 1: Single Server (ASGI)

Start here.  A single `langgraph dev` server runs the supervisor and all
workers co-located via in-process ASGI transport.  This establishes the
graph coding pattern that spans all three deployment scenarios.

```bash
# Terminal 1
# Start the combined server
just supervisor
```

Once the server is running on port 2024, run the demo in another terminal:

```bash
# Terminal 2
just run-demo
```

Optionaly monitor with Langshark:

```bash
# Terminal 3
just langshark-supervisor
```

### Example 2: Dual Server (HTTP Split)

The supervisor and workers run as independent Agent Servers
communicating over HTTP.  Each server has its own execution pool
and checkpoint lineage.  This scenario is for instructional and
prototyping purposes — a learning crutch before graduating to
the self-hosted deployment.

Open **two terminals**:

```bash
# Terminal 1 — start the worker server (port 2025)
just worker

# Terminal 2 — start the supervisor server (port 2024)
just supervisor-http
```
> **Terminals 1 and 2** - Two langgraph dev servers
>
> ![Supervisor and Worker servers](docs/screenshots/langstrata-supervisor-http-and-worker-terminals.png)


In a **third terminal**, run the same demo (no code changes needed):

```bash
# Terminal 3
just run-demo
```

> Terminal 3 — Run demo
>
> ![Run demo start](docs/screenshots/just-run-demo-begin.png)
> ![Run demo end](docs/screenshots/just-run-demo-end.png)


Optionally monitor each server with its own Langshark instance:

```bash
just langshark-supervisor   # supervisor orchestration view
just langshark-worker       # worker execution view
```

> Langshark TUI showing the supervisor's orchestration view
>
> ![Langshark supervisor view](docs/screenshots/langshark-supervisor-thread.png)

> Langshark TUI showing worker execution threads on the worker server
>
> ![Langshark worker view](docs/screenshots/langshark-worker-thread.png)


### Example 3: Self-Hosted Docker (Production)

The Docker Compose stack launches two independent agent servers
using the production `langchain/langgraph-api` image, with shared
PostgreSQL and Redis backing stores.  The supervisor listens on
port **8123**.

```bash
# Terminal 1
# Build and start the full stack (Postgres, Redis, supervisor, worker)
just docker-up --detach
docker compose ps
```

![docker compose ps](docs/screenshots/langstrata-docker-compose.png)

Run the demo against the Docker supervisor:

```bash
# Terminal 1
just docker-run-demo
```

Optionally monitor with Langshark (connecting to Docker-exposed ports):

```bash
# Terminal 3
just docker-langshark-supervisor
# Terminal 4
just docker-langshark-worker
```

Tear down when finished:

```bash
# Terminal 1
# If the docker session is running detached. Run just docker-down otherwise enter Ctrl-C.
just docker-down
```

![docker compose ps](docs/screenshots/langstrata-docker-compose-down.png)

### Building the Images

The committed Dockerfiles are **generated artifacts** of the langgraph
CLI — they are not hand-written.  Each `langgraph.*.json` config file
declares which graphs its server hosts, and `langgraph dockerfile`
translates that config into a production Dockerfile:

```bash
# Supervisor image (all 4 graphs: supervisor + researcher, coder, analyst)
uv run langgraph dockerfile -c langgraph.json supervisor.Dockerfile

# Worker image (3 graphs: researcher, coder, analyst)
uv run langgraph dockerfile -c langgraph.workers.json worker.Dockerfile
```

> **Regenerate after config changes.**  Because the Dockerfiles are
> generated, you must re-run the appropriate command above whenever you
> change a `langgraph.*.json` file — otherwise the built image will not
> reflect your changes.

If you don't need to check a Dockerfile in, `langgraph build` builds the
image directly from the config (handy when driving `langgraph up`
yourself):

```bash
uv run langgraph build -c langgraph.json -t langstrata-supervisor
```

And `langgraph validate` sanity-checks a config file without building
anything:

```bash
uv run langgraph validate -c langgraph.workers.json
```

### Docker Compose File

`docker-compose.yml` launches the entire stack with a single command,
picking up the images from [Building the Images](#building-the-images)
in two different ways:

- **The agent-server images are built from the Dockerfiles.**  The
  `supervisor` and `worker` services use a `build:` block, which tells
  `docker compose` to build an image from a Dockerfile on disk — it does
  *not* pull a prebuilt image from a container registry:

  ```yaml
  supervisor:
    build:
      context: .                    # project root is the build context
      dockerfile: supervisor.Dockerfile
  ```

  Docker runs a standard build from `supervisor.Dockerfile` /
  `worker.Dockerfile`.  Each starts `FROM langchain/langgraph-api:3.14`
  (the official LangGraph agent-server base image, pulled from Docker
  Hub), adds the project code with `ADD . /deps/langstrata`, installs it,
  and records which graphs the server serves in the `LANGSERVE_GRAPHS`
  environment variable.  This is why no `langgraph build` step is needed
  before `just docker-up` — compose does the building for you.

- **Postgres and Redis are pulled from Docker Hub.**  Those services use
  an `image:` block, not `build:`, so compose pulls `postgres:16` and
  `redis:7` straight from the registry.

Here is what each project file contributes:

| File | How docker-compose.yml uses it |
|---|---|
| `langgraph.json` | `langgraph dockerfile` bakes its graphs into `supervisor.Dockerfile`, which compose builds into the `supervisor` image |
| `langgraph.workers.json` | Same, into `worker.Dockerfile` → the `worker` image |
| `supervisor.Dockerfile`, `worker.Dockerfile` | Named by each service's `build.dockerfile` (build context `.`) |
| `.env` | `env_file` on both services — injects `ANTHROPIC_API_KEY`, `AGENT_SERVER_MODEL`, etc. |
| `docker-init/01-create-worker-db.sql` | Mounted at Postgres's `/docker-entrypoint-initdb.d/`; creates the `langgraph_workers` database on first boot |
| `.dockerignore` | Trims the build context that `ADD . /deps/langstrata` copies into the image |

The `ENV LANGSERVE_GRAPHS=...` line inside the generated Dockerfiles is
what tells each server *which* graphs it serves.  It is a JSON mapping
of **graph name → graph definition** (a module path pointing at the
`graph` variable or a `create_*` factory function).  The supervisor
image's line resolves to:

```json
{
  "supervisor": "/deps/langstrata/src/langstrata/supervisor/agent.py:create_supervisor_agent",
  "researcher": "/deps/langstrata/src/langstrata/workers/researcher.py:graph",
  "coder":      "/deps/langstrata/src/langstrata/workers/coder.py:graph",
  "analyst":    "/deps/langstrata/src/langstrata/workers/analyst.py:graph"
}
```

The `environment:` block then wires the pieces together over Docker's
internal network — service names, not `localhost`, so there are no port
conflicts to manage by hand:

- `DATABASE_URI` and `REDIS_URI` point at the `postgres` and `redis`
  service hostnames.  Both servers share one Postgres, but the worker
  uses its own `langgraph_workers` database and Redis DB 1.
- `AGENT_SERVER_MODE: "http"` with `AGENT_SERVER_WORKER_API_URL:
  http://worker:8124` makes the supervisor reach the worker over the
  compose network.
- `depends_on: condition: service_healthy` waits for the Postgres and
  Redis healthchecks before the agent servers start.
- `ports` expose `8123` (supervisor) and `8124` (worker) to the host.

#### `just docker-up` parameters

`just docker-up` wraps `docker compose up`:

- **`--build` is always on** — compose rebuilds the `supervisor` and
  `worker` images on every run, so a freshly regenerated Dockerfile, or
  source and `.env` changes, are picked up automatically.
- **`DETACH` is the only parameter** — `just docker-up` runs in the
  foreground (Ctrl-C stops the stack); pass the flag through to run in
  the background, e.g. `just docker-up --detach`.
- The justfile loads `.env` via `set dotenv-load := true`, so secrets
  are available to both containers through `env_file`.

## Project Layout

```
src/langstrata/
├── __init__.py                  # Package root
├── config.py                    # Settings (env-driven topology toggle)
├── schemas.py                   # Shared Pydantic models
├── supervisor/
│   ├── agent.py                 # Pattern A: create_supervisor_agent()
│   └── send_supervisor.py       # Pattern B: Send API supervisor (planned)
└── workers/
│   ├── __init__.py
│   ├── base.py                  # create_worker_agent() helper
│   ├── researcher.py            # graph = ... (web search)
│   ├── coder.py                 # graph = ... (code generation)
│   ├── analyst.py               # graph = ... (data analysis)
│   └── send_worker.py           # Pattern B: Send API map-reduce graph

scripts/
├── demo.py                      # SDK client demo
└── switch-mode.sh               # Topology toggle helper

justfile                           # Command runner (just, not make)
langgraph.json                     # All graphs (supervisor + workers); start here
langgraph.workers.json             # Worker-only config (researcher, coder, analyst)
supervisor.Dockerfile              # Supervisor container image (generated)
worker.Dockerfile                  # Worker container image (generated)
docker-compose.yml                 # Full-stack deployment
```

## Next Steps

- **Make the demo agents your own.**  Replace the `researcher`, `coder`,
  and `analyst` workers with graphs and tools that fit your domain,
  then regenerate the config files and Dockerfiles as shown above.
- **Harden your workers.**  As your workers fan out and call more
  external APIs, the [langshark-bites](https://github.com/stokomax/langshark-bites)
  collection addresses exactly those problems — rate limiting, visible
  backoff, provider failover, tolerant JSON output parsing, and reducers
  that merge parallel `Send` results without duplicates.
- **Make the supervisor event-driven.**  Today the supervisor *polls*
  for worker completion — the deep-agent flow launches a subagent with
  `start_async_task` and then repeatedly checks `check_async_task` until
  it finishes, which burns supervisor turns and adds latency under load.
  The hardening step is to be *notified* when a worker completes instead
  of asking.  That event-driven completion is the motivation behind the
  planned `completion_notify` bite; it isn't available yet, so it remains
  a known gap in this template.
- **[Observe with LangSmith](#observing-with-langsmith).**  Turn on
  LangSmith tracing to capture every supervisor and worker run in one
  place — see the appendix for the exact environment variables and how
  to read the traces.

## Appendix

### Monitoring with Langshark

[Langshark](https://github.com/stokomax/langshark) is a local-first LangGraph inspector TUI
that connects to a running LangGraph Server.  With a dual-server deployment, you run
**one Langshark per server** — each shows a different view of the same workflow.

#### Connecting

```bash
# Supervisor server (shows orchestration)
just langshark-supervisor        # langshark -c http://localhost:2024 --graph supervisor

# Worker server (shows individual worker runs)
just langshark-worker            # langshark -c http://localhost:2025

# Or use the web UI
just langshark-web-supervisor    # open http://localhost:8000
just langshark-web-worker
```

> :bulb: Not all features are available when connected to a langgraph dev server

#### What To Observe

##### 1. Server Stats (`s` key) — Confirm Which Server Is Which

| Stat | Supervisor Server | Worker Server |
|---|---|---|
| Graphs registered | `supervisor` | `researcher`, `coder`, `analyst` |
| Thread count | 1 (the conversation) | N (one per delegated task) |
| Status | `active` or `idle` | `active` when a worker is processing |

**Why it matters:** Each server has its own graph set, its own thread list,
its own lifecycle.  They are independent.

##### 2. Thread Browser (`t` key) — Independent Checkpoint Lineages

| Server | Threads visible |
|---|---|
| Supervisor | One thread (the conversation) — full message history |
| Worker | N threads — each worker invocation has its own checkpoint sequence |

**Why it matters:** Each worker run is a first-class LangGraph thread with its
own checkpoint history, independent of the supervisor's state.  This is
impossible in the in-process model.

##### 3. Graph Filtering (`--graph` flag)

```bash
langshark -c http://localhost:2025 --graph researcher
```

Focus on a single worker's behavior without noise from other workers or the
supervisor.  Each graph is independently addressable.

### Observing with LangSmith

[LangSmith](https://smith.langchain.com) is LangChain's hosted observability
platform.  Unlike Langshark — which inspects a single *running* server from a
terminal — LangSmith records every run to the cloud, giving one unified view of
the whole supervisor → worker workflow across both servers.

LangChain agents built with `create_agent` (the workers) and `create_deep_agent`
(the supervisor) already support LangSmith tracing.  No code changes are
required — you only enable it with environment variables.

#### Prerequisites

1. Sign up for a free account at [smith.langchain.com](https://smith.langchain.com).
2. Create an API key (Settings → API Keys → Create API Key) and copy it.

#### Enable tracing

Add the three `LANGSMITH_*` variables to your `.env` file (they are read by
`just` via `set dotenv-load := true` and passed to both agent servers):

```ini
# LangSmith observability (optional — leave unset to disable tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=langstrata
```

- `LANGSMITH_TRACING=true` — the opt-in switch.  Unset (or `false`) and
  langstrata runs exactly as before, with no tracing.
- `LANGSMITH_API_KEY` — authenticates traces to your LangSmith account.
- `LANGSMITH_PROJECT` — groups all langstrata runs under one project (LangSmith
  defaults to `default` when unset).

> :bulb: Set `LANGSMITH_PROJECT=langstrata` so every supervisor and worker run
> lands in one place.

#### Run and view traces

Start the servers and run the demo exactly as before:

```bash
just supervisor                 # single server — or the dual-server pair:
just worker                     # terminal 2
just supervisor-http            # terminal 1

just run-demo                   # another terminal
```

Then open [smith.langchain.com](https://smith.langchain.com) and select the
`langstrata` project.  Each `just run-demo` produces a trace whose root is the
supervisor (DeepAgent) graph, with each worker delegation linked as its own
sub-run/thread.  This is the key difference from Langshark: **one** LangSmith
project captures the entire split-server workflow, whereas Langshark needs a
separate TUI per server.

For finer control — tracing only part of your app, attaching tags/metadata, or
using a self-hosted endpoint — see the [LangChain observability guide](https://docs.langchain.com/oss/python/langchain/observability)
and the [Trace with LangGraph](https://docs.langchain.com/langsmith/trace-with-langgraph) guide.

### Why the demo probes port 2025

The transport mode (ASGI vs HTTP) is a **compile-time decision** — it is set when
`deepagents` constructs the graph and wires up the `AsyncSubAgent` specs.
Specifically, `AsyncSubAgent.url` controls the transport:

- **`url=None`** (omitted) → the LangGraph SDK creates an in-process
  **ASGI transport** loopback. No HTTP connection is made. The subagent runs
  in the same process as the supervisor.
- **`url="http://..."`** → the LangGraph SDK creates a standard HTTP client
  that connects to a remote Agent Protocol server. The subagent runs on a
  separate process.

The `Settings` class reads `AGENT_SERVER_MODE` from the environment
(`"asgi"` or `"http"`) and passes the appropriate URL to each `AsyncSubAgent`.
This is the single toggle that switches between the two topologies.

The name server itself does not know whether a subagent is co-deployed or remote;
it only knows how to run the compiled graph it was given.  So the demo's graph
topology display runs `_detect_mode()` — a function that probes
`http://localhost:2025`. If a worker server responds, it concludes dual-server
mode; otherwise, single-server.

This is a **post-hoc inspection**, not a query of the server's internals.
The LangGraph Server API does not expose whether a subagent was configured
with a URL. The only way to confirm the deployment mode is to observe the
runtime behavior: which ports respond, where threads are created, and whether
the worker server has its own set of assistants.

In production, the mode should be set explicitly and not left to
auto-detection. The demo's probe exists only as a convenience for
tutorial use.

### Prompt Configuration

The demonstration script (`scripts/demo.py`) ships with two built-in prompts
that control what the supervisor is asked to do.  The active prompt is chosen
at runtime via the `DEMO_PROMPT_MODE` environment variable — set it in your
`.env` file or pass it inline.  The demo defaults to the **abridged** prompt.

#### Original Prompt

Set `DEMO_PROMPT_MODE=original` to use the full research-then-code prompt:

```
Research the latest trends in LangGraph agent architectures
and then write a Python script demonstrating a simple state machine.
```

This prompt exercises **two worker types** — the supervisor delegates the
research phase to the `researcher` subagent, and the coding phase to the
`coder` subagent — before synthesising the final answer.

This prompt is taken from the [LangChain DeepAgent demo](https://docs.langchain.com/oss/python/deepagents/async-subagents).

#### Abridged Prompt

The default (`DEMO_PROMPT_MODE=abridged`, or unset) uses a shorter prompt
that exercises a single worker delegation:

```
You MUST delegate the research task to the researcher subagent.
Wait for it to complete, then write a one-sentence summary of LangGraph's purpose.
```

This is the quickest way to verify that supervisor → worker routing works
end-to-end without waiting for a multi-step sequence.

#### Usage

Set `DEMO_PROMPT_MODE` inline:

```bash
# Use the abridged prompt (default — no env var needed)
uv run python scripts/demo.py

# Use the original prompt
DEMO_PROMPT_MODE=original uv run python scripts/demo.py
```

Or add it to your `.env` file so `just run-demo` picks it up automatically
(the `justfile` loads `.env` with `set dotenv-load := true`):

```
DEMO_PROMPT_MODE=original
```

Then:

```bash
just run-demo   # uses the prompt from .env
```

### Inspecting Configuration

The `main.py` entrypoint prints the **effective runtime configuration**
resolved from environment variables and defaults, and warns about any
missing API keys.  Run it via `just` or directly:

```bash
# Via the justfile recipe
just check-config

# Or directly
uv run python main.py
```

Example output:

```
Hello from langstrata!
  mode=http              # set AGENT_SERVER_MODE=http|asgi
  model=claude-haiku-4-5-20251001            # set AGENT_SERVER_MODEL=provider:model
  worker_api_url=http://worker:2024  # set AGENT_SERVER_WORKER_API_URL=<url>
  env prefix: AGENT_SERVER_          # all config vars use this prefix
  demo_prompt_mode=abridged  # set DEMO_PROMPT_MODE=abridged|full
```

Each value is annotated with the environment variable you would set to
change it.  This is the quickest way to see what your deployment will
actually do before starting any servers, and to catch missing
credentials early.

**Note about recipes that set env vars.**  Recipes such as
`just supervisor`, `just supervisor-http`, and `just worker` set
`AGENT_SERVER_MODE` and sometimes `AGENT_SERVER_WORKER_API_URL` inline.
Running `just check-config` reads from your current shell environment
and your `.env` file — it does not inherit the env vars that those
recipes set.  To preview the exact command a recipe would run, use
`just --dry-run <recipe>` (short form: `just -n <recipe>`).  This
prints every shell command the recipe executes, including its inline
environment variables:

```bash
# Preview what `just supervisor` will execute
just --dry-run supervisor

# Preview what `just supervisor-http` will execute
just --dry-run supervisor-http

# Preview the worker recipe
just --dry-run worker
```

The output shows the full invocation with all env vars set by the
recipe, so you can see exactly what mode and URL each recipe uses.  For
example, `just --dry-run supervisor` outputs
`AGENT_SERVER_MODE=asgi uv run langgraph dev ...`.
