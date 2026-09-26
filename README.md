
![langstrata fin logo](docs/screenshots/fin_logo_v2.png)

# Langstrata

Langstrata is a blueprint for building mult-agent Langgraph/Langchain systems that are performative from first prototype through production. This blueprint has three design pillars:

1. **Prefer supervisor–worker delegation patterns.** Work is run under a
   supervising agent that delegates to workers using asynchronous subagents today (Pattern A) and,
   later where appropriate, a map-reduce worker that employs the Send API
   (Pattern B).
2. **Run the supervisor and the workers as separate Agent Servers.** The
   **Split Agent Server architecture** assigns each server its own process,
   resource pool, and checkpoint lineage, so that worker load does not contend
   with the supervisor's turn.
3. **Prefer push completion to poll-based waiting.** If the supervisor learns of completion only by polling, model turns
   are consumed by status traffic (i.e. turn erosion), crowding out planning and
   synthesis. Turn erosion also appears when the supervisor **stops prematurely** and must later be **nudged**
   with human input to resume. This blueprint therefore delivers worker termination as a runtime push notification. See [Push completion](#push-completion) for more information.

 Langstrata is built with langshark-bites. Langshark-bites provides reusable primitivies (called "bites") that langstrata integrates.  There are bites for push completion, external API rate limiting, JSON parsing, and more. You can adopt langshark-bites in any LangGraph project without adopting the langstrata blueprint. See the [langshark-bites documentation](https://github.com/stokomax/langshark-bites) for details.

The lab tutorials show how this repository runs LangGraph and illustrate the design pillars in practice. Demonstration software and Compose files are deliberately richer than a minimal sample: they serve as templates for dual-server solutions. The labs progressively introduce infrastructure and agent configuration changes.

 The first lab demonstrates a standlone langgraph dev server to establish a baseline. The fourth lab, the final lab, demonstrates a complete self-hosted Docker deployment with all the blueprint features enabled. Labs 3 and 4 describe, in detail, the infrastructure, configuration changes and tools needed. Once you are familiar with the blueprint, the same infrastructure can be applied to your mulit-agent projects; Learn it once, here, and apply it everywhere is the goal. 

## Quick Start (TL;DR)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY or DEEPSEEK_API_KEY

# 2. Install dependencies
uv sync

# 3. Start single-server mode (supervisor + workers on one process)
just standalone

# 4. Run the demo (in another terminal)
just run-demo

# 5. (Optional) Graduate to dual-server mode
# Terminal 1: just worker
# Terminal 2: just supervisor
# Terminal 3: just run-demo
```

## Table of Contents

- [Quick Start](#quick-start-tldr)
- [The Building Blocks](#the-building-blocks)
- [Supervisor-Worker Delegation](#supervisor-worker-delegation)
- [Split-Agent Server Architecture](#split-agent-server-architecture)
- [Push Completion](#push-completion)
- [Labs](#labs)
  - [Prerequisites](#prerequisites)
  - [Lab 1: Single Server (Pattern A)](#lab-1-pattern-a-on-a-single-server)
  - [Lab 2: Split Server Architecture](#lab-2-split-agent-server-architecture)
  - [Lab 3: Split Server with Push Notifications](#lab-3-split-agent-server-architecture-with-push-notifications)
  - [Lab 4: Docker Compose Stack](#lab-4-self-hosted-dual-langgraph-servers-deployed-with-docker-compose)
    - [Building the Images — `langgraph` CLI Explained](#building-the-images-langgraph-cli-explained)
    - [Manual Image Building](#manual-image-building-without-just-or-docker-compose)
    - [Database Initialization Details](#database-initialization-details)
    - [Docker Compose File — What to Borrow for Production](#docker-compose-file-what-to-borrow-for-production)
    - [Self-Hosted Single LangGraph Server](#self-hosted-single-langgraph-server-production-pattern)
    - [`just docker-up` Parameters Reference](#just-docker-up-parameters-reference)
- [Next Steps](#next-steps)
- [Summary](#summary)
- [Appendix](#appendix)
  - [Project Layout](#project-layout)
  - [Demo Prompt Configuration](#prompt-configuration)
  - [Deployment Configuration Reference](#configuration-reference-1)
  - [Deployment Configuration Diagnostics](#configuration-diagnostics)
  - [Monitoring with Langshark](#monitoring-with-langshark)
  - [Monitoring with LangSmith](#monitoring-with-langsmith)
  - [Troubleshooting the laboratories](#troubleshooting-the-laboratories)
  - [Why the Demo Probes Port 2025](#why-the-demo-probes-port-2025)

## The Building Blocks

LangChain and LangGraph already supply the necessary primitives. Langstrata is a
curated subset of those primitives. 

Agents are built with three primitives sourced from Langchain, Langgraph
and Deepagents APIs:
- `create_agent()`: individual workers
- `create_deep_agent()` with `AsyncSubAgent`: a supervisor that delegates
- `StateGraph` with `Send`: fan-out within a single graph (Pattern B)

Transport between agent servers is either ASGI (a single `langgraph dev` process) or
HTTP (a split deployment). **Agent code is unchanged**; only the URL supplied to
each `AsyncSubAgent` differs.

| Element | Role | Reference |
|---|---|---|
| `create_agent()` | A single LLM-powered agent with tools and a system prompt. | [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents) |
| `create_deep_agent()` + `AsyncSubAgent(name=..., url=?)` | A coordinating supervisor that delegates. Omit `url` for same-server ASGI transport; set an HTTP URL for a remote Agent Protocol server. | [Async Subagents](https://docs.langchain.com/oss/python/deepagents/async-subagents) |
| `StateGraph` + `Send` | Parallel fan-out within one graph; results accumulate via a reducer (for example `operator.add`). | [Map-Reduce and the Send API](https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api) |

These three primitives suffice for the agent design patterns described here. More
specialized LangChain and LangGraph facilities may be introduced later without
altering the architecture.

## Supervisor-Worker Delegation

The work delegation is always from supervisor to workers. The number of subagent patterns is limited to two: asynchronous subagents and map-reduce with send.

**Pattern A is implemented in this blueprint.** Pattern B (map-reduce on the
worker) is planned.

### Pattern A: Asynchronous subagents

The supervisor launches workers one at a time, or a small number at a time. The
model selects which worker is appropriate for each task.

```python
create_deep_agent(subagents=[
    AsyncSubAgent(name="researcher", graph_id="researcher"),
    AsyncSubAgent(name="coder",      graph_id="coder"),
    AsyncSubAgent(name="analyst",    graph_id="analyst"),
])
```

This is the canonical asynchronous-subagents pattern from the Deep Agents
documentation. Co-location versus remote placement of those graphs is configured
separately via `AsyncSubAgent.url` and `AGENT_SERVER_MODE`.

### Pattern B: Map-reduce via `Send` (planned)

> **Not yet implemented in this blueprint.**

A future worker-facing subagent would host a `StateGraph` that fans a known batch
of tasks out with `Send` to the same `create_agent()` workers used by Pattern A,
then reduces results in a single super-step. Routing in that fan-out is
deterministic; no model participates in the dispatch. See the LangGraph
[Map-Reduce and the Send API](https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api)
documentation for the underlying graph structure.

## Split-Agent Server Architecture

The split-agent architecture gives each server its own resource pool. On a single server, supervisor turns and worker runs share one execution pool.
Under load, workers may delay the supervisor’s next turn indefinitely.

By manipulating `langgraph.json` and deepagents features the split-agent is realized. A single-server configuration is built entirely from the `langgraph.json` file. It includes the supervisor and worker defintions. In a split-agent architecture, a second `langgraph.json` file that includes only worker agents is required. A second requirement is the worker definitions in both files need to match. The role of the `langgraph.json` changes from configuring the workers to defining how a supervisor discovers the workers. The`langgraph.workers.json` role is to configure the workers. 

A deepagents middleware feature is then used by the langstrata code to establish the supervisor and worker runtimes. The presence of a URL is all that's required to change the runtime to expect a single server or split-agent architecture.

## Push completion

This section concerns how the supervisor learns that a subagent has finished. When many **agent** instances run asynchronously—especially under **horizontal
scaling** (several instances of the same agent, each with a different initial
prompt)—**waiting by status poll** exposes the supervisor to **turn erosion**
(see the third pillar): model turns spent on “done yet?”, and runs that stop
early until a human nudges them. **Push completion** removes that dependency.
Completion arrives as an event, so polling-induced turn erosion no longer 
governs whether the supervisor can proceed.

Push completion replaces waiting by status poll with an event path from worker to
supervisor:

1. When a worker agent reaches a terminal state, it **emits** a completion notification
   toward the supervisor.
2. A **receiver** on the supervisor side accepts that notification and verifies it.
3. A **mailbox** holds pending notices for the relevant supervisor thread.
4. **Drain** middleware, on the supervisor’s next model call, reads the mailbox,
   filters and deduplicates notices, and **summarizes** them into context the
   model can use, then clears what it has delivered.
5. Optionally, a **wake** starts a follow-on supervisor run so an idle or stopped
   supervisor processes the mailbox without a human nudge.

Steps 1–2 follow **A2A roles** (emitter on the worker, receiver on the
supervisor, push-style delivery of “this work finished”). Steps 3–5 are
**blueprint machinery**, not A2A. The blueprint machinery injects the notifications into the supervisor's prompt history. 


| Mechanism | Behaviour | Cost while workers run |
|---|---|---|
| **Poll tools** | The model repeatedly asks whether a task has finished (`start_async_task` / `check_async_task`). | Often many supervisor model turns |
| **Push completion** | Worker emits; supervisor receives; mailbox and drain filter, deduplicate, and summarize into the next model call; optional wake. | Worker wall-clock dominates; supervisor status turns and manual resume recede |

| Role | Where | Responsibility |
|---|---|---|
| **Emitter** (A2A) | Worker process | On terminal state, send a completion notification to the configured supervisor endpoint. |
| **Receiver** (A2A) | Supervisor process | Accept the notification, verify delivery credentials, and pass it into local delivery. |
| **Mailbox** (blueprint) | Supervisor Store | Buffer pending notices per thread so completions survive until the supervisor can read them. |
| **Drain** (blueprint) | Supervisor middleware | Before a model call: filter, deduplicate, and summarize mailbox notices into context; clear delivered items. |
| **Wake** (blueprint) | Receiver follow-on | Optionally start a supervisor run so mailbox contents are processed without a human message. |

## Labs

The following tutorials describe how this repository hosts LangGraph. Each
tutorial states which design pillars it illustrates. The agent graphs remain the same; only host configuration changes. Commands are `just` recipes defined in the project [justfile](https://github.com/casey/just).

### Prerequisites

**Platform support.** langstrata is right at home on Linux, and it also
runs great inside the Windows Subsystem for Linux 2 (WSL2). macOS is
likely fine too; it just hasn't been tested here yet. Native Windows
(running directly on the Windows host) is not currently supported, so
if you're on Windows, WSL2 is the way to go.

Start by copying the example environment file and setting the required variables:

```bash
cp .env.example .env
```

The `.env.example` file documents every variable the project uses.
At minimum set a provider key and model; A2A defaults match the embedded
receiver (supervisor origin) used by the `just` recipes:

```ini
# Required: LLM provider API keys
# ANTHROPIC_API_KEY=sk-ant-...
# DEEPSEEK_API_KEY=sk-...

# LLM model for all agents (supervisor + workers).
AGENT_SERVER_MODEL=anthropic:claude-sonnet-4-6

# Optional: LangSmith tracing
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=lsv2_...
# LANGSMITH_PROJECT=langstrata

# A2A completion notifier (optional but on by default in just dual-server recipes)
A2A_COMPLETION_NOTIFIER_ENABLED=true
A2A_VERIFY_MODE=dev
A2A_RECEIVER_URL=http://localhost:2024
A2A_SUPERVISOR_URL=http://localhost:2024
```

**Required variables:**

- `ANTHROPIC_API_KEY` :  API key for Anthropic Claude models (needed unless you use DeepSeek)
- `DEEPSEEK_API_KEY` :  API key for DeepSeek models (needed unless you use Anthropic)
- `AGENT_SERVER_MODEL` :  Model identifier, e.g. `anthropic:claude-sonnet-4-6`, `deepseek:reasoner`, or `anthropic:claude-haiku-3.5`

**Optional variables (LangSmith tracing is opt-in, not required by LangChain or LangGraph):**

- `LANGSMITH_TRACING` :  set to `true` to enable tracing (the opt-in switch; leave unset to run without tracing).
- `LANGSMITH_API_KEY` :  API key for [LangSmith](https://smith.langchain.com) tracing. Set this only if you want to inspect traces (see [LangChain observability docs](https://docs.langchain.com/oss/python/langchain/observability#enable-tracing)).
- `LANGSMITH_PROJECT` :  LangSmith project name (default: `langstrata`)

Once your `.env` is ready, install dependencies and verify the configuration:

```bash
uv sync
uv run python main.py    # verify config loads
```

> :bulb: **Recipe lists.** Run bare `just` for a curated quick-start (the few commands
> you need day-to-day); run `just list` (or `just --list`) for every recipe.

### Lab 1: Pattern A on a Single Server

Begin here. A single `langgraph dev` server runs the supervisor and all workers  co-located via in-process ASGI transport. The same graph coding pattern is used in every subsequent laboratory.

```bash
# Terminal 1
# Start the combined server
just standalone
```

Once the server is running on port 2024, run the demo in another terminal:

```bash
# Terminal 2
just run-demo
```

The `scripts/demo.py`, which `just run-demo` invokes, creates a supervisor Langgraph thread. When in ASGI mode
`AsyncSubAgent.url` is omitted, so worker delegation stays in-process. `AsyncSubAgent` constitutes part of the deepagents middleware and serves as a building block for langstrata.

Optionally monitor with Langshark:

```bash
# Terminal 3
just langshark-supervisor
```

### Lab 2: Split Agent Server Architecture

The supervisor and workers run as independent Agent Servers communicating over HTTP. Each has
its own execution pool and checkpoint lineage. 

> :warning: The dual `langgraph dev` example that follows is for instructional purposes only. Infrastructure limitations with langgraph dev server make it not suitable for split-agent deployments. Langstrata employs workarounds to ease the transition from running a single server to implementing a self-hosted dual-server langgraph deployment.

Open **two terminals**:

```bash
# Terminal 1: start the worker server (port 2025)
just worker

# Terminal 2: start the supervisor server (port 2024)
just supervisor
```
The screenshots below are representative of what you see in each terminal window. The red arrows highlight the URL's for the two servers.

> **Terminals 1 and 2** - Two langgraph dev servers
>
> ![Supervisor and Worker servers](docs/screenshots/langstrata-supervisor-http-and-worker-terminals.png)


In a **third terminal**, run the same demo (no code changes needed):

```bash
# Terminal 3
just run-demo
```

The demo retrieves and displays the current server topology. The mode should indicated HTTP which is highlighted by the red arrow on the screenshot below. The orange arrows highlight the Langgraph thread ID's which are formatted as UUID's.

> Terminal 3: Run demo
>
> ![Run demo start](docs/screenshots/just-run-demo-begin.png)
> ![Run demo end](docs/screenshots/just-run-demo-end.png)


Optionally monitor each server with its own Langshark instance. Look for the Langgraph thread ID's similar to how the orange underlines highlight the thread data in the screenshots below.

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

At this point in the tutorial, the role of the langgraph.json file and deepagents middleware has been demonstrated. Two langgragh dev servers were used to demonstrate the dual-server operation, plus an introduction on how to observe the execution of the deployment. 

The next lab extends the dual langgraph dev server example by adding push completion to eliminate supervisor turn erosion. 

### Lab 3: Split Agent Server Architecture with Push Notifications

To highlight and confirm the operational differences between polling and push completion a revised version of the demo is used. The `a2a-demo` command demonstrates push completion in action: it requires
that emit → mailbox → drain succeed (see
[A2A Completion Notifications](#a2a-completion-notifications-langshark-bites)
below for the full mechanism). The `a2a-demo` manipulates environment variables to 
activate the infrastructure associated with the push completion. 

| | `just run-demo` | `just a2a-demo` |
|---|---|---|
| Program | `scripts/demo.py` | `scripts/a2a_demo.py` |
| Question | Did Pattern A succeed on the split architecture? | Was at least one completion notice delivered and drained? |
| Success | The supervisor answers; workers executed | `/health` succeeds and the demo exits 0 after a drained notice |
| Pillar | Split Agent Server architecture | Push completion (avoid supervisor turn erosion and premature stop when peer instances finish out of order) |

The a2a-demo internally provisions push complection at runtime over a client connection to the supervisor langgraph server. The provisioning is handled dynamically by a2-demo. A factory function creates the push configuration metadata (`a2a_push_config`)

#### Push Config Structure and Flow

When `a2a-demo` runs, it mints an `a2a_push_config` — a dictionary with five fields — and passes it as the `config` parameter to the supervisor's `runs.stream()` call. This config travels with the dispatch and tells the worker where and how to send its completion notification.

The five fields:

- **url** — the webhook endpoint (`{receiver_url}/a2a/notifications`) where the worker POSTs its completion
- **token** — an opaque JWT signed with the supervisor's callback secret; the worker echoes it back, and the receiver uses it to route the completion to the correct supervisor thread
- **context_id** — the supervisor's thread ID, used as the A2A Task.contextId so completions correlate with the originating conversation
- **authentication** — always `{"schemes": ["Bearer"]}` for this integration
- **mode** — the `A2A_VERIFY_MODE` value (`dev`, `verify`, or `strict`), so the worker's emitter can fail fast if strict mode requires signing but no key is configured

This config crosses the supervisor→worker boundary through two pieces of langstrata glue:

**Supervisor side** (`a2a_dispatch_forwarder.py`): Replaces deepagents' built-in `start_async_task` tool. When the supervisor's LLM invokes that tool to launch a worker, the custom tool reads the push config from the parent run's config (or mints one from `A2A_*` env vars) and passes it in the `config=` argument when calling `client.runs.create()` on the worker server.

**Worker side** (`worker_emitter.py`): Each worker graph factory receives that config and calls `build_worker_middleware(config)`. That function extracts the push config and, if present, attaches the langshark-bites `A2APushNotifierMiddleware` to the worker graph. If the config is absent (plain invocation, not a dispatched subagent), the worker builds without the emitter and behaves exactly as before.

When the worker reaches a terminal state, the emitter middleware POSTs a signed completion notification to the webhook URL, including the token in the payload and a Bearer JWT in the Authorization header when a signing key is configured.

#### The `http.app` Configuration

Lab 3 is the first lab where the `http.app` setting in `langgraph.json` becomes active. In Labs 1 and 2, the `http` block exists but is latent — it points to `./src/langstrata/a2a/webapp.py:app` but the A2A completion notifier is disabled by default (`A2A_COMPLETION_NOTIFIER_ENABLED=false`), so the webapp returns an empty FastAPI app that does nothing.

```json
{
  "graphs": {
    "supervisor": "./src/langstrata/supervisor/agent.py:create_supervisor_agent",
    "researcher": "./src/langstrata/workers/researcher.py:create_researcher_agent",
    "coder": "./src/langstrata/workers/coder.py:create_coder_agent",
    "analyst": "./src/langstrata/workers/analyst.py:create_analyst_agent"
  },
  "http": {
    "app": "./src/langstrata/a2a/webapp.py:app"
  }
}
```

When you run Lab 3 with `A2A_COMPLETION_NOTIFIER_ENABLED=true`, the webapp activates and mounts the langshark-bites completion receiver on the **same process and port as the supervisor graphs**:

| Route | Purpose |
|-------|---------|
| `POST /a2a/notifications` | Completion webhook (worker → supervisor) |
| `POST /a2a/result` | Full task result fetch (on-demand) |
| `GET /health` | Liveness probe |

This is why `A2A_RECEIVER_URL` and `A2A_SUPERVISOR_URL` both point to the supervisor origin — the receiver shares the supervisor's Agent Server. The bite appends `/a2a/notifications` to the base URL when minting the push config. Reserved paths under `/a2a/` are `notifications` and `result`; Google's built-in A2A remains at `/a2a/{assistant_id}`.

The webapp (`webapp.py`) returns an empty FastAPI app when A2A is disabled (so the supervisor boots without A2A in earlier labs), and returns the full receiver when enabled, wired to the supervisor's Store via `runtime.store`.


### Lab 4: Self-hosted dual langgraph servers deployed with Docker Compose.

Replaces the langgraph dev servers with containerized langgraph servers. Compose starts two independent agent servers with Postgres and Redis. The Postgres and Redis infrastructure is shared.

This lab is the **production deployment template**. It demonstrates a complete self-hosted LangGraph stack: two Agent Servers (supervisor + workers), PostgreSQL for checkpoint persistence, Redis for pub/sub and cross-replica coordination, and an embedded A2A completion receiver on the supervisor.

#### Quick Start (using the `just` recipes)

```bash
# Build and start the full stack (Postgres, Redis, supervisor, worker)
just docker-up --detach
docker compose ps
```

![docker compose ps](docs/screenshots/langstrata-docker-compose.png)

The supervisor listens on port 8123, the worker on port 8124. Run the demo against the Docker supervisor:

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
# If running detached
just docker-down
# Or Ctrl-C if running in foreground
```

![docker compose ps](docs/screenshots/langstrata-docker-compose-down.png)

---

### Building the Images — `langgraph` CLI Explained

The `langgraph` CLI provides three commands that translate your `langgraph.json` configuration into production Docker images. These are generated artifacts — you must regenerate them whenever you change a `langgraph.*.json` file.

#### 1. `langgraph dockerfile` — Generate a Dockerfile from config

```bash
# Supervisor image (all 4 graphs: supervisor + researcher, coder, analyst)
uv run langgraph dockerfile -c langgraph.json supervisor.Dockerfile

# Worker image (3 graphs: researcher, coder, analyst)
uv run langgraph dockerfile -c langgraph.workers.json worker.Dockerfile
```

| Flag | Purpose |
|------|---------|
| `-c`, `--config` | Path to the `langgraph.json` config file |
| (positional) | Output Dockerfile path |

**What it does:** Reads the `graphs` and `http.app` entries from the config, resolves the base image (`langchain/langgraph-api:3.14`), and emits a Dockerfile that:
- Copies the project into `/deps/langstrata`
- Installs dependencies via `uv pip install -e .` with constraints from the base image
- Sets `ENV LANGSERVE_GRAPHS=<json>` mapping graph names to their factory functions
- Strips build tools (`pip`, `setuptools`, `wheel`, `uv`) to minimize attack surface

> **Regenerate after config changes.** Because the Dockerfiles are generated, you must re-run the appropriate command above whenever you change a `langgraph.*.json` file (including the `http.app` entry for the A2A receiver); otherwise the built image will not reflect your changes.

#### 2. `langgraph build` — Build the Docker image locally

```bash
# Build supervisor image and tag it
uv run langgraph build -c langgraph.json -t langstrata-supervisor

# Build worker image and tag it
uv run langgraph build -c langgraph.workers.json -t langstrata-worker
```

| Flag | Purpose |
|------|---------|
| `-c`, `--config` | Path to the `langgraph.json` config file |
| `-t`, `--tag` | Docker image tag (e.g., `langstrata-supervisor:latest`) |

**What it does:** Internally runs `langgraph dockerfile` to generate a temporary Dockerfile, then executes `docker build` with the project root as context. This is equivalent to running the two-step generate + `docker build` manually, but in one command.

**When to use:** Use `langgraph build` for quick local iteration. For CI/CD or registry pushes, prefer the two-step approach (generate Dockerfile → `docker build` → `docker push`) so the Dockerfile is auditable and reproducible.

#### 3. `langgraph validate` — Validate config without building

```bash
uv run langgraph validate -c langgraph.workers.json
```

| Flag | Purpose |
|------|---------|
| `-c`, `--config` | Path to the `langgraph.json` config file |

**What it does:** Checks that the config file is syntactically valid, all graph entry points resolve to importable Python objects, and dependencies are declared. Returns exit code 0 on success, non-zero on failure. Useful in pre-commit hooks or CI pipelines.

---

### Manual Image Building (without `just` or `docker compose`)

For environments where you need full control over the build process (e.g., pushing to a private registry, multi-arch builds, SBOM generation):

```bash
# 1. Generate Dockerfiles (auditable, commit-able artifacts)
uv run langgraph dockerfile -c langgraph.json supervisor.Dockerfile
uv run langgraph dockerfile -c langgraph.workers.json worker.Dockerfile

# 2. Build with docker (or buildx for multi-arch)
docker build -f supervisor.Dockerfile -t your-registry/langstrata-supervisor:v1.0.0 .
docker build -f worker.Dockerfile -t your-registry/langstrata-worker:v1.0.0 .

# 3. Push to registry
docker push your-registry/langstrata-supervisor:v1.0.0
docker push your-registry/langstrata-worker:v1.0.0

# 4. Update docker-compose.yml to use prebuilt images:
#    supervisor:
#      image: your-registry/langstrata-supervisor:v1.0.0
#    worker:
#      image: your-registry/langstrata-worker:v1.0.0
```

**Base image:** Both generated Dockerfiles start from `FROM langchain/langgraph-api:3.14` — the official LangGraph Agent Server base image. This image includes:
- Python 3.14 runtime
- LangGraph API server (`langgraph-api` package)
- Runtime dependencies and constraints (`/api/constraints.txt`)
- Non-root user for security

---

### Database Initialization Details

The stack uses **one PostgreSQL instance with two databases** and **one Redis instance with three logical databases**:

#### PostgreSQL
| Database | Owner | Purpose | Created by |
|----------|-------|---------|------------|
| `langgraph` | `langgraph` | Supervisor checkpoints, threads, runs | Postgres default `POSTGRES_DB` |
| `langgraph_workers` | `langgraph` | Worker checkpoints, threads, runs | `docker-init/01-create-worker-db.sql` |

The file `docker-init/01-create-worker-db.sql` contains a single statement:
```sql
CREATE DATABASE langgraph_workers;
```

**How it works:** The `postgres` service mounts `./docker-init/` at `/docker-entrypoint-initdb.d/`. PostgreSQL's official entrypoint automatically executes all `.sql`/`.sh` files in that directory **alphabetically** on first initialization (when the data directory is empty). This is a standard PostgreSQL Docker pattern — no custom entrypoint needed.

#### Redis
| Database | Purpose |
|----------|---------|
| DB 0 | Supervisor: checkpoints, pub/sub, internal state |
| DB 1 | Worker: checkpoints, pub/sub, internal state |
| DB 2 | A2A completion notifier: cross-replica JTI deduplication, dead-letter queue |

The `REDIS_URI` environment variables encode the DB number in the path: `redis://redis:6379/0`, `redis://redis:6379/1`, `redis://redis:6379/2`.

---

### Docker Compose File — What to Borrow for Production

The `docker-compose.yml` is a **complete, working reference** for a dual-server LangGraph deployment. Here is what each service contributes and what you can adapt:

#### Service Breakdown

| Service | Type | Reusable for Production? | Notes |
|---------|------|--------------------------|-------|
| `postgres` | `image:` (pulled) | **Yes** — replace with managed PostgreSQL (RDS, Cloud SQL, etc.) | Healthcheck ensures readiness. Mount `docker-init/` for worker DB creation. |
| `redis` | `image:` (pulled) | **Yes** — replace with managed Redis (ElastiCache, etc.) | Healthcheck ensures readiness. Logical DB separation (0/1/2) via `REDIS_URI` paths. |
| `supervisor` | `build:` (from Dockerfile) | **Yes** — use prebuilt image from registry | Embeds A2A receiver via `http.app`. Requires `AGENT_SERVER_MODE=http`. |
| `worker` | `build:` (from Dockerfile) | **Yes** — use prebuilt image from registry | Worker-only graphs. No `http.app` (receiver lives on supervisor). |

#### Key Patterns to Borrow

1. **Service hostnames for internal networking** — Use `postgres`, `redis`, `worker`, `supervisor` as hostnames in `DATABASE_URI`, `REDIS_URI`, `AGENT_SERVER_WORKER_API_URL`. No `localhost` or hardcoded IPs.

2. **Healthcheck-gated startup** — `depends_on: condition: service_healthy` on both agent servers prevents startup races. Copy the healthcheck definitions for your infrastructure.

3. **Environment variable wiring** — The `environment:` blocks show the exact variables each server needs. For production, move secrets to a secrets manager and inject via `env_file` or orchestration platform secrets.

4. **Port exposure** — Only `ports: ["8123:8123"]` and `["8124:8124"]` are exposed to the host. Internal communication uses container ports on the Docker network.

5. **Build context trimming** — `.dockerignore` excludes `.git`, `__pycache__`, `*.pyc`, `tests/`, `docs/`, `.venv`, etc. from the `ADD . /deps/langstrata` copy. Essential for reasonable image sizes.

---

### Self-Hosted Single LangGraph Server (Production Pattern)

If you only need one Agent Server (e.g., supervisor + workers co-located, or just workers), the pattern simplifies:

```yaml
# docker-compose.single.yml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: langgraph
      POSTGRES_USER: langgraph
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langgraph"]
      interval: 5s
      timeout: 5s
      retries: 10

  redis:
    image: redis:7
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 10

  langgraph-server:
    image: your-registry/langstrata-supervisor:v1.0.0  # or worker image
    ports:
      - "8123:8123"
    env_file:
      - .env
    environment:
      PORT: "8123"
      DATABASE_URI: "postgres://langgraph:${POSTGRES_PASSWORD}@postgres:5432/langgraph?sslmode=disable"
      REDIS_URI: "redis://redis:6379/0"
      AGENT_SERVER_MODE: "asgi"   # or "http" if workers are remote
      # AGENT_SERVER_WORKER_API_URL: "http://worker:8124"  # only for http mode
      A2A_COMPLETION_NOTIFIER_ENABLED: "true"
      A2A_RECEIVER_URL: "http://langgraph-server:8123"
      A2A_SUPERVISOR_URL: "http://langgraph-server:8123"
      A2A_REDIS_URL: "redis://redis:6379/2"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  postgres_data:
```

**Key differences from dual-server:**
- Single `langgraph-server` service (use supervisor image for ASGI mode, or worker image for worker-only)
- `AGENT_SERVER_MODE=asgi` for co-located supervisor+workers (no worker URL needed)
- `AGENT_SERVER_MODE=http` + `AGENT_SERVER_WORKER_API_URL` if workers are on a separate server
- A2A receiver still embeds on the same server via `http.app` in `langgraph.json`

---

### `just docker-up` Parameters Reference

`just docker-up` wraps `docker compose up` with sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--build` | **Always enabled** | Rebuilds `supervisor` and `worker` images on every run. Picks up regenerated Dockerfiles, source changes, and `.env` changes automatically. |
| `DETACH` | `""` (foreground) | Pass `--detach` to run in background: `just docker-up --detach`. Foreground mode: Ctrl-C stops the stack. |
| `.env` loading | Via `set dotenv-load := true` | Secrets (`ANTHROPIC_API_KEY`, etc.) are available to both containers through `env_file: .env` in compose. |

**Equivalent manual command:**
```bash
docker compose up --build --detach   # background
docker compose up --build            # foreground
```

---

## Next Steps

- **Make the demo agents your own.**  Replace the `researcher`, `coder`,
  and `analyst` workers with graphs and tools that fit your domain,
  then regenerate the config files and Dockerfiles as shown above.
- **Harden your workers.**  As your workers fan out and call more
  external APIs, the [langshark-bites](https://github.com/stokomax/langshark-bites)
  collection addresses exactly those problems: rate limiting, visible
  backoff, provider failover, tolerant JSON output parsing, and reducers
  that merge parallel `Send` results without duplicates.
- **Pattern B (map-reduce via Send)** is planned. Extend Pattern A workers first.
- **[Observe with LangSmith](#observing-with-langsmith).**  Turn on
  LangSmith tracing to capture every supervisor and worker run in one
  place. See the appendix for the exact environment variables and how
  to read the traces.


## Summary

langstrata is a **blueprint** — an opinionated architecture for split-server multi-agent systems. langshark-bites is a **library** of reusable bites. This section clarifies the boundary so you know what to adopt.

### langstrata Provides (Architecture & Wiring)

| Area | What it is | Reusable? |
|------|------------|-----------|
| Split-server topology | Supervisor + workers as separate Agent Servers with separate checkpoint lineages | Yes — pattern applies to any domain |
| Supervisor factory | `create_supervisor_agent()` with deepagents + AsyncSubAgent delegation | Yes — adapt worker list for your domain |
| Worker factories | `create_researcher_agent`, `create_coder_agent`, `create_analyst_agent` — per-run graph factories that accept `config` | Yes — replace tools/system prompts for your workers |
| Configuration system | `Settings` class with `AGENT_SERVER_*` prefix, mode toggle (asgi/http), per-worker URL overrides | Yes — extend with your own env vars |
| A2A dispatch forwarder | `a2a_dispatch_forwarder.py` — replaces deepagents' `start_async_task` to forward `a2a_push_config` to worker runs | Yes — if you use deepagents + A2A |
| Worker emittr wiring | `worker_emitter.py` — `build_worker_middleware(config)` attaches emitter only when dispatched with push config | Yes — if you use per-run graph factories |
| Embedded receiver | `webapp.py` — mounts langshark-bites receiver on supervisor via `langgraph.json` `http.app` | Yes — if you want receiver on supervisor origin |
| MCP diagnostics | `supervisor_mcp.py` — optional diagnostic tools via MCP | Optional — if you want MCP observability |
| Docker/Compose | Split-server compose with PostgreSQL + Redis, generated Dockerfiles | Yes — adapt for your services |

### langshark-bites Provides (Library Primitives)

The `a2a_completion_notifier` bite supplies the **mechanics** langstrata wires together:

| Primitive | Purpose |
|-----------|---------|
| `A2APushNotifierMiddleware` | Worker middleware that POSTs completion webhook on terminal state |
| `build_a2a_notifier_from_config` | Factory that reads `config.configurable["a2a_push_config"]` and builds the middleware |
| `A2ASigner` | RS256 JWT signing for emitter (optional, needed for `verify`/`strict` modes) |
| `PushClient` | HTTP POST with retry/backoff for webhook delivery |
| `build_push_config` | Supervisor-side: mints callback token, builds `a2a_push_config` dict |
| `create_receiver_app` | FastAPI receiver: verifies, deduplicates, writes to Store, triggers wake |
| `ReceiverSettings` / `EmitterSettings` | Configuration dataclasses (env-driven) |
| `MailboxDrainMiddleware` | Supervisor middleware: reads Store mailbox, injects notices into model call |
| `NotificationMailbox` | Store write + wake logic |
| `build_notification` | A2A Task wire format (sparse profile) |
| `mint_callback_token` / `unseal_callback_token` | Opaque routing tokens (supervisor→worker→receiver) |
| `JWKSClient` / `verify_sender_jwt` | Sender auth verification |
| `RedisJtiStore` / `InMemoryJtiStore` | Deduplication by JWT `jti` |
| `create_a2a_bridge_server` | MCP server hosting receiver + Store SDK + diagnostic tools |
| `create_supervisor_tools` | Reconnectable LangChain tools for supervisor (list_pending, health, get_result, DLQ) |

### What to Adopt for Your Project

| If you want... | Adopt from langstrata | Adopt from langshark-bites |
|----------------|----------------------|---------------------------|
| Split-server supervisor/worker pattern | `create_supervisor_agent`, worker factories, config system | — |
| A2A push completion | `a2a_dispatch_forwarder`, `worker_emitter`, `webapp` | `a2a_completion_notifier` bite (all primitives) |
| Per-run graph factories with config | `create_researcher_agent(config)` pattern | — |
| Deepagents + A2A integration | `a2a_dispatch_forwarder` | `A2APushNotifierMiddleware`, `build_push_config` |
| MCP diagnostics on supervisor | `supervisor_mcp.py` | `create_supervisor_tools` |
| Embedded receiver on supervisor | `webapp.py` + `langgraph.json` `http.app` | `create_receiver_app` |

### What's langstrata-Specific

- The three example workers (researcher, coder, analyst) and their tools
- The demo scripts (`demo.py`, `a2a_demo.py`)
- The specific `langgraph.json` / `langgraph.workers.json` graph registrations
- The Docker Compose service names and network topology


## Appendix

### Project Layout

```text
src/langstrata/
├── __init__.py
├── config.py                    # Settings (AGENT_SERVER_* topology toggle)
├── schemas.py                   # Shared Pydantic models
├── py.typed
├── a2a/                         # langshark-bites a2a_completion_notifier wiring
│   ├── webapp.py                # DEFAULT: receiver via langgraph.json http.app
│   ├── a2a_dispatch_forwarder.py  # deepagents dispatch-config forwarding
│   ├── worker_emitter.py        # worker attach-or-skip emitter
│   └── supervisor_mcp.py        # optional native MCP tools on supervisor
├── supervisor/
│   └── agent.py                 # Pattern A: create_supervisor_agent()
└── workers/
    ├── base.py                  # create_worker_agent() helper
    ├── researcher.py            # create_researcher_agent(config) + module graph
    ├── coder.py                 # create_coder_agent(config)
    └── analyst.py               # create_analyst_agent(config)

scripts/
├── demo.py                      # SDK client demo (Pattern A)
├── a2a_demo.py                  # live A2A push demo
├── a2a_mcp_check.py             # receiver /health (+ optional MCP tools)
├── gen_a2a_keys.py              # RSA/JWKS for verify/strict
└── switch-mode.sh

justfile                           # lab recipes
langgraph.json                     # graphs + http.app (A2A webapp); start here
langgraph.workers.json             # worker-only graphs
supervisor.Dockerfile / worker.Dockerfile   # generated (regenerate after json changes)
docker-compose.yml                 # supervisor + worker + Postgres + Redis
tests/                             # unit tests (incl. A2A attach-or-skip)
```

### Demo Prompt Configuration

The demonstration script (`scripts/demo.py`) ships with two built-in prompts
that control what the supervisor is asked to do.  The active prompt is chosen
at runtime via the `DEMO_PROMPT_MODE` environment variable. Set it in your
`.env` file or pass it inline.  The demo defaults to the **abridged** prompt.

#### Original Prompt

Set `DEMO_PROMPT_MODE=original` to use the full research-then-code prompt:

```text
Research the latest trends in LangGraph agent architectures
and then write a Python script demonstrating a simple state machine.
```

This prompt exercises **two worker types.** The supervisor delegates the
research phase to the `researcher` subagent, and the coding phase to the
`coder` subagent, before synthesising the final answer.

This prompt is taken from the [LangChain DeepAgent demo](https://docs.langchain.com/oss/python/deepagents/async-subagents).

#### Abridged Prompt

The default (`DEMO_PROMPT_MODE=abridged`, or unset) uses a shorter prompt
that exercises a single worker delegation:

```text
You MUST delegate the research task to the researcher subagent.
Wait for it to complete, then write a one-sentence summary of LangGraph's purpose.
```

This is the quickest way to verify that supervisor → worker routing works
end-to-end without waiting for a multi-step sequence.

#### Usage

Set `DEMO_PROMPT_MODE` inline:

```bash
# Use the abridged prompt (default; no env var needed)
uv run python scripts/demo.py

# Use the original prompt
DEMO_PROMPT_MODE=original uv run python scripts/demo.py
```

Or add it to your `.env` file so `just run-demo` picks it up automatically
(the `justfile` loads `.env` with `set dotenv-load := true`):

```ini
DEMO_PROMPT_MODE=original
```

Then:

```bash
just run-demo   # uses the prompt from .env
```

### Deployment Configuration Reference

All configuration is via environment variables with the `AGENT_SERVER_` prefix (loaded from `.env` by the justfile). Run `just check-config` to see the effective resolved values.


#### Topology & Model

| Variable | Default | Description |
|---|---|---|
| `AGENT_SERVER_MODE` | `http` | `asgi` = co-deployed supervisor+workers (ASGI transport); `http` = split servers (Agent Protocol) |
| `AGENT_SERVER_MODEL` | `anthropic:claude-sonnet-4-6` | LLM for all agents in `provider:model` format (e.g. `anthropic:claude-sonnet-4-6`, `deepseek:reasoner`, `openai:gpt-4o`) |
| `AGENT_SERVER_WORKER_API_URL` | `http://worker:2025` | Base URL of worker Agent Protocol server (used in `http` mode) |
| `AGENT_SERVER_SYSTEM_PROMPT` | (built-in) | Custom supervisor system prompt |

#### Per-Worker URL Overrides (Optional)

For hybrid deployments where specific workers live on different servers:

| Variable | Default | Description |
|---|---|---|
| `AGENT_SERVER_RESEARCHER_URL` | (none) | Override worker URL for researcher only |
| `AGENT_SERVER_CODER_URL` | (none) | Override worker URL for coder only |
| `AGENT_SERVER_ANALYST_URL` | (none) | Override worker URL for analyst only |

#### A2A Completion Notifier

Enable on **both** servers to activate push completion:

| Variable | Default | Description |
|---|---|---|
| `A2A_COMPLETION_NOTIFIER_ENABLED` | `false` | Set `true` to enable push completion loop |
| `A2A_VERIFY_MODE` | `dev` | `dev` (no signing), `verify` (verify signatures), `strict` (require valid signatures) |
| `A2A_CALLBACK_TOKEN_SECRET` | `change-me...` | Shared secret for callback token HMAC |
| `A2A_RECEIVER_URL` | `http://localhost:2024` | Supervisor origin (bite appends `/a2a/notifications`) |
| `A2A_SUPERVISOR_URL` | `http://localhost:2024` | Supervisor origin for StoreClient fallback |
| `A2A_MCP_TRANSPORT` | `stdio` | MCP bridge transport: `stdio` or `streamable-http` |
| `A2A_SUPERVISOR_MCP_URL` | (none) | External MCP bridge URL (e.g. `http://localhost:8001/mcp`); attaches the four diagnostic tools |
| `A2A_REDIS_URL` | (none) | Redis for cross-replica JTI dedup + dead-letter queue |
| `A2A_PRIVATE_KEY_FILE` | (none) | RSA private key for emitter signing (`verify`/`strict` modes) |
| `A2A_KID` | (none) | Key ID for JWKS |
| `A2A_EMITTER_ISSUER` | (none) | Emitter issuer URL (worker origin) |
| `A2A_EMITTER_AUDIENCE` | (none) | Emitter audience (supervisor origin) |

#### LangSmith Tracing

| Variable | Default | Description |
|---|---|---|
| `LANGSMITH_TRACING` | (unset) | Set `true` to enable |
| `LANGSMITH_API_KEY` | (none) | LangSmith API key |
| `LANGSMITH_PROJECT` | `langstrata` | Project name in LangSmith |

### Deployment Configuration Diagnostics

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

```text
Hello from langstrata!
  mode=http              # set AGENT_SERVER_MODE=http|asgi
  model=anthropic:claude-sonnet-4-6            # set AGENT_SERVER_MODEL=provider:model
  worker_api_url=http://worker:2025  # set AGENT_SERVER_WORKER_API_URL=<url>
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
and your `.env` file; it does not inherit the env vars that those
recipes set.  To preview the exact command a recipe would run, use
`just --dry-run <recipe>` (short form: `just -n <recipe>`).  This
prints every shell command the recipe executes, including its inline
environment variables:

```bash
# Preview what `just supervisor` will execute
just --dry-run standalone

# Preview what `just supervisor-http` will execute
just --dry-run supervisor

# Preview the worker recipe
just --dry-run worker
```

The output shows the full invocation with all env vars set by the
recipe, so you can see exactly what mode and URL each recipe uses.  For
example, `just --dry-run supervisor` outputs
`AGENT_SERVER_MODE=asgi uv run langgraph dev ...`.

### Monitoring with Langshark

[Langshark](https://github.com/stokomax/langshark) is a local-first LangGraph inspector TUI
that connects to a running LangGraph Server.  With a dual-server deployment, you run
**one Langshark per server:** each shows a different view of the same workflow.

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

##### 1. Server Stats (`s` key): Confirm Which Server Is Which

| Stat | Supervisor Server | Worker Server |
|---|---|---|
| Graphs registered | `supervisor` | `researcher`, `coder`, `analyst` |
| Thread count | 1 (the conversation) | N (one per delegated task) |
| Status | `active` or `idle` | `active` when a worker is processing |

**Why it matters:** Each server has its own graph set, its own thread list,
its own lifecycle.  They are independent.

##### 2. Thread Browser (`t` key): Independent Checkpoint Lineages

| Server | Threads visible |
|---|---|
| Supervisor | One thread (the conversation): full message history |
| Worker | N threads: each worker invocation has its own checkpoint sequence |

**Why it matters:** Each worker run is a first-class LangGraph thread with its
own checkpoint history, independent of the supervisor's state.  This is
impossible in the in-process model.

##### 3. Graph Filtering (`--graph` flag)

```bash
langshark -c http://localhost:2025 --graph researcher
```

Focus on a single worker's behavior without noise from other workers or the
supervisor.  Each graph is independently addressable.

### Monitoring with LangSmith

[LangSmith](https://smith.langchain.com) is LangChain's hosted observability
platform.  Unlike Langshark, which inspects a single *running* server from a
terminal, LangSmith records every run to the cloud, giving one unified view of
the whole supervisor → worker workflow across both servers.

LangChain agents built with `create_agent` (the workers) and `create_deep_agent`
(the supervisor) already support LangSmith tracing.  No code changes are
required; you only enable it with environment variables.

#### Prerequisites

1. Sign up for a free account at [smith.langchain.com](https://smith.langchain.com).
2. Create an API key (Settings → API Keys → Create API Key) and copy it.

#### Enable tracing

Add the three `LANGSMITH_*` variables to your `.env` file (they are read by
`just` via `set dotenv-load := true` and passed to both agent servers):

```ini
# LangSmith observability (optional; leave unset to disable tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=langstrata
```

- `LANGSMITH_TRACING=true`: the opt-in switch.  Unset (or `false`) and
  langstrata runs exactly as before, with no tracing.
- `LANGSMITH_API_KEY`: authenticates traces to your LangSmith account.
- `LANGSMITH_PROJECT`: groups all langstrata runs under one project (LangSmith
  defaults to `default` when unset).

> :bulb: Set `LANGSMITH_PROJECT=langstrata` so every supervisor and worker run
> lands in one place.

#### Run and view traces

Start the servers and run the demo exactly as before:

```bash
just standalone        # single server, or the dual-server pair:
just worker            # terminal 2
just supervisor        # terminal 1

just run-demo          # another terminal
```

Then open [smith.langchain.com](https://smith.langchain.com) and select the
`langstrata` project.  Each `just run-demo` produces a trace whose root is the
supervisor (DeepAgent) graph, with each worker delegation linked as its own
sub-run/thread.  This is the key difference from Langshark: **one** LangSmith
project captures the entire split-server workflow, whereas Langshark needs a
separate TUI per server.

For finer control (tracing only part of your app, attaching tags/metadata, or
using a self-hosted endpoint), see the [LangChain observability guide](https://docs.langchain.com/oss/python/langchain/observability)
and the [Trace with LangGraph](https://docs.langchain.com/langsmith/trace-with-langgraph) guide.


### Troubleshooting the laboratories

| Symptom | Likely cause | Fix |
|---|---|---|
| `just run-demo` cannot connect | Supervisor not up on 2024 | `just standalone` or `just supervisor` |
| Dual-server demo never finishes workers | Worker not on 2025 | Start `just worker` first |
| Port already in use | Leftover `langgraph dev` | Stop the old process or change ports |
| Missing model / auth errors | Empty `.env` keys | `cp .env.example .env` and set a provider key |
| `just a2a-mcp-check` fails `/health` | Receiver not mounted or wrong origin | Confirm `langgraph.json` has `http.app`; `A2A_RECEIVER_URL` = supervisor origin |
| Compose demo fails | Stack not healthy | `docker compose ps` / `just docker-down` then `just docker-up --detach` |

Preview recipe env with `just --dry-run supervisor` (or `supervisor-http` / `worker`).

### Why the demo probes port 2025

The transport mode (ASGI vs HTTP) is a **compile-time decision:** it is set when
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
topology display runs `_detect_mode()`, a function that probes
`http://localhost:2025`. If a worker server responds, it concludes dual-server
mode; otherwise, single-server.

This is a **post-hoc inspection** of runtime behaviour. The LangGraph Server API
does not expose whether a subagent was configured with a URL, so mode is inferred
from which ports respond, where threads are created, and whether the worker server
has its own set of assistants.

In production, the mode should be set explicitly and not left to
auto-detection. The demo's probe exists only as a convenience for
tutorial use.
