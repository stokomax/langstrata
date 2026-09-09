# justfile — langstrata dual-server workflow
#
# The justfile mirrors the README graduation path:
#   just supervisor      -> single-server mode (ASGI, all graphs on one process)
#   just supervisor-http -> supervisor half of dual-server (HTTP to worker)
#   just worker          -> worker half of dual-server (runs on separate port)
#
# Start with `just supervisor` (single server).  Graduate to running
# `just worker` + `just supervisor-http` in two terminals (dual server).

# Use bash so PATH includes ~/.local/bin and other user-level additions
set shell := ["bash", "-l", "-c"]
# Load .env so secrets (ANTHROPIC_API_KEY, etc.) are available to recipes
set dotenv-load := true

# Ensure we're in the project root
guard:
  @test -f pyproject.toml || { echo "error: Run this recipe from the project root (cd langstrata/)"; exit 1; }

# Show available recipes
default: guard
  @just --list

# ── Local dev servers ───────────────────────────────────────────
#
# ⚠  Symlink hack for `.langgraph_api`
#
#   `langgraph dev` always writes checkpoint/memory state to
#   `.langgraph_api` in the current directory with no option to
#   change the path.  When running two servers (supervisor-http +
#   worker) they would trample each other's data.
#
#   The workaround: each recipe removes the old dir/symlink,
#   creates a dedicated per-server directory, and symlinks
#   `.langgraph_api` → that directory so `langgraph dev` writes
#   into the isolated directory.
#
#   This is a demonstration / prototyping convenience only.  For
#   durable per-server state use `langgraph up` with PostgreSQL.
#   The `.langgraph_api_*` directories are gitignored by convention.

# Single-server: supervisor + workers co-deployed on one process (ASGI).  Start here.
supervisor: guard
  rm -rf .langgraph_api
  mkdir -p .langgraph_api_supervisor
  ln -sfn .langgraph_api_supervisor .langgraph_api
  AGENT_SERVER_MODE=asgi uv run langgraph dev --config langgraph.json --port 2024 --no-reload --no-browser --n-jobs-per-worker 10

# Supervisor half of dual-server: delegates to worker server via HTTP.  Run alongside `just worker`.
supervisor-http: guard
  rm -rf .langgraph_api
  mkdir -p .langgraph_api_supervisor_http
  ln -sfn .langgraph_api_supervisor_http .langgraph_api
  AGENT_SERVER_MODE=http AGENT_SERVER_WORKER_API_URL=http://localhost:2025 \
    uv run langgraph dev --config langgraph.json --port 2024 --no-reload --no-browser --n-jobs-per-worker 10

# Worker half of dual-server: hosts researcher, coder, analyst graphs.  Run alongside `just supervisor-http`.
worker: guard
  rm -rf .langgraph_api
  mkdir -p .langgraph_api_worker
  ln -sfn .langgraph_api_worker .langgraph_api
  AGENT_SERVER_MODE=http uv run langgraph dev --config langgraph.workers.json --port 2025 --no-reload --no-browser --n-jobs-per-worker 10

# ── Docker full stack ──────────────────────────────────────────

# Build and start Docker full stack (Postgres, Redis, supervisor, worker).
# Pass --detach to run in the background.
docker-up DETACH="": guard
  docker compose up --build $DETACH

# ── Docker stack ────────────────────────────────────────────

# Run the demo against the Docker stack (supervisor on port 8123).
# Assumes `just docker-up` (or `just docker-up --detach`) is already running.
docker-run-demo: guard
  SUPERVISOR_URL=http://localhost:8123 uv run python scripts/demo.py

# Tear down Docker full stack
docker-down: guard
  docker compose down

# Tail supervisor logs
logs-supervisor: guard
  docker compose logs -f supervisor

# Tail worker logs
logs-worker: guard
  docker compose logs -f worker

# ── Langshark monitoring ───────────────────────────────────────

# Langshark TUI pointed at supervisor server (Docker, port 8123)
docker-langshark-supervisor: guard
  ./scripts/langshark.sh -c http://localhost:8123 --graph supervisor

# Langshark TUI pointed at worker server (Docker, port 8124)
docker-langshark-worker: guard
  ./scripts/langshark.sh -c http://localhost:8124

# Langshark TUI pointed at supervisor server
langshark-supervisor: guard
  ./scripts/langshark.sh -c http://localhost:2024 --graph supervisor

# Langshark TUI pointed at worker server
langshark-worker: guard
  ./scripts/langshark.sh -c http://localhost:2025

# Langshark web UI pointed at supervisor server
langshark-web-supervisor: guard
  ./scripts/langshark.sh web -c http://localhost:2024 --graph supervisor

# Langshark web UI pointed at worker server
langshark-web-worker: guard
  ./scripts/langshark.sh web -c http://localhost:2025

# ── Demo script ────────────────────────────────────────────────

# Run the SDK demo script against local servers
run-demo: guard
  uv run python scripts/demo.py

# ── Configuration ─────────────────────────────────────────────────

# Print effective configuration (model, mode, worker API URL, etc.)
check-config: guard
  uv run python main.py

# ── Code quality ───────────────────────────────────────────────

# Run ruff linter
lint: guard
  uv run ruff check src/

# Run ruff formatter
format: guard
  uv run ruff format --check src/
