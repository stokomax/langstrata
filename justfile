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

# A2A environment defaults for every recipe.  Each ${VAR:-default} expansion
# picks up an already-set value (e.g. from .env via dotenv-load) and falls
# back to the project default -- so the a2a loop works even without .env.
# Default: receiver embedded on supervisor (http.app).  A2A_RECEIVER_URL is the
# supervisor origin; the bite appends /a2a/notifications.  AUTO_SPAWN off.
A2A_ENV := 'A2A_COMPLETION_NOTIFIER_ENABLED="${A2A_COMPLETION_NOTIFIER_ENABLED:-true}" A2A_VERIFY_MODE="${A2A_VERIFY_MODE:-dev}" A2A_CALLBACK_TOKEN_SECRET="${A2A_CALLBACK_TOKEN_SECRET:-change-me-callback-token-secret-0123456789}" A2A_RECEIVER_URL="${A2A_RECEIVER_URL:-http://localhost:2024}" A2A_SUPERVISOR_URL="${A2A_SUPERVISOR_URL:-http://localhost:2024}" A2A_RECEIVER_PORT="${A2A_RECEIVER_PORT:-8001}" A2A_RECEIVER_HOST="${A2A_RECEIVER_HOST:-0.0.0.0}" A2A_MCP_TRANSPORT="${A2A_MCP_TRANSPORT:-stdio}"'

# ── Help ─────────────────────────────────────────────────────────
# Bare `just` (no recipe) shows a curated quick-start.  `just list` (or
# `just --list`) shows every recipe.
default: guard
  @echo ""
  @echo "langstrata — quick start"
  @echo "========================"
  @echo ""
  @echo "  just standalone            Single server (ASGI): all graphs on one process.  Start here."
  @echo "  just worker                Worker half of the dual-server stack (separate port)."
  @echo "  just supervisor-http       Supervisor half of the dual-server stack (HTTP to worker)."
  @echo "  just run-demo              Run the SDK demo against the running servers."
  @echo "  just a2a-demo              Live A2A push demo against running supervisor+worker."
  @echo "  just check-config          Print effective configuration (model, mode, worker URL, etc.)."
  @echo "  just lint / format         Ruff linter / formatter."
  @echo ""
  @echo "  just list                  Show ALL available recipes (same as: just --list)"
  @echo ""

# Show ALL available recipes (the full list)
list: guard
  @just --list


# Ensure we're in the project root (also the bare-`just` default's dependency)
guard:
  @test -f pyproject.toml || { echo "error: Run this recipe from the project root (cd langstrata/)"; exit 1; }

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
standalone: guard
  rm -rf .langgraph_api
  mkdir -p .langgraph_api_supervisor
  ln -sfn .langgraph_api_supervisor .langgraph_api
  {{A2A_ENV}} AGENT_SERVER_MODE=asgi uv run langgraph dev --config langgraph.json --port 2024 --no-reload --no-browser --n-jobs-per-worker 10

# Supervisor half of dual-server: delegates to worker server via HTTP.  Run alongside `just worker`.
supervisor: guard
  rm -rf .langgraph_api
  mkdir -p .langgraph_api_supervisor_http
  ln -sfn .langgraph_api_supervisor_http .langgraph_api
  {{A2A_ENV}} AGENT_SERVER_MODE=http AGENT_SERVER_WORKER_API_URL=http://localhost:2025 \
    uv run langgraph dev --config langgraph.json --port 2024 --no-reload --no-browser --n-jobs-per-worker 10

# Worker half of dual-server: hosts researcher, coder, analyst graphs.  Run alongside `just supervisor-http`.
worker: guard
  rm -rf .langgraph_api
  mkdir -p .langgraph_api_worker
  ln -sfn .langgraph_api_worker .langgraph_api
  {{A2A_ENV}} AGENT_SERVER_MODE=http uv run langgraph dev --config langgraph.workers.json --port 2025 --no-reload --no-browser --n-jobs-per-worker 10

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

# ── A2A completion notifications (langshark-bites) ─────────────────

# Generate RSA keys + JWKS for the a2a emitter (verify/strict modes only; dev mode needs none)
a2a-keys: guard
  uv run python scripts/gen_a2a_keys.py

# Legacy: standalone FastAPI sidecar on :8001.  Prefer embedded http.app
# (default): webhook lives on the supervisor — no separate receiver process.
# a2a-receiver: guard
#   {{A2A_ENV}} A2A_RECEIVER_URL=http://localhost:8001 uv run uvicorn langstrata.a2a.receiver_server:app --host 0.0.0.0 --port 8001

# Verify A2A: GET /health on A2A_RECEIVER_URL (supervisor origin when embedded)
# and optional MCP tools if a bridge is configured.
a2a-mcp-check: guard
  {{A2A_ENV}} uv run python scripts/a2a_mcp_check.py

# Live A2A push demo against the running split-server stack.  Requires
# `just supervisor` + `just worker` (receiver is on the supervisor via
# http.app — no :8001 process).  This recipe never starts a server.
a2a-demo: guard
  {{A2A_ENV}} uv run python scripts/a2a_demo.py

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
