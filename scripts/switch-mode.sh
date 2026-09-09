#!/usr/bin/env bash
# switch-mode.sh — Toggle langstrata between ASGI and HTTP topology.
#
# Usage:
#   ./scripts/switch-mode.sh asgi          # co-deployed mode
#   ./scripts/switch-mode.sh http          # remote worker mode (default URL)
#   ./scripts/switch-mode.sh http :2025    # remote worker mode (custom URL)
#   ./scripts/switch-mode.sh status        # show current mode

set -euo pipefail

MODE="${1:-status}"
WORKER_URL="${2:-http://localhost:2025}"

case "$MODE" in
  asgi)
    echo "=== ASGI mode (co-deployed) ==="
    echo ""
    echo "  export AGENT_SERVER_MODE=asgi"
    echo ""
    echo "  Workers will run on the SAME server as the supervisor."
    echo ""
    echo "  Start the supervisor:"
    echo "    just supervisor"
    echo ""
    echo "  Run the demo:"
    echo "    just run-demo"
    echo ""
    echo "  Observe with Langshark:"
    echo "    just langshark-supervisor"
    ;;

  http)
    echo "=== HTTP mode (remote worker server) ==="
    echo ""
    echo "  export AGENT_SERVER_MODE=http"
    echo "  export AGENT_SERVER_WORKER_API_URL=$WORKER_URL"
    echo ""
    echo "  Workers will run on a SEPARATE server at $WORKER_URL."
    echo ""
    echo "  Start the servers (two terminals):"
    echo "    Terminal 1: just worker"
    echo "    Terminal 2: AGENT_SERVER_MODE=http \\"
    echo "                AGENT_SERVER_WORKER_API_URL=$WORKER_URL \\"
    echo "                langgraph dev --config langgraph.json --port 2024"
    echo ""
    echo "  Run the demo:"
    echo "    just run-demo"
    echo ""
    echo "  Observe with Langshark (two terminals):"
    echo "    Terminal 3: just langshark-supervisor"
    echo "    Terminal 4: just langshark-worker"
    ;;

  status)
    echo "=== Current topology ==="
    echo "  AGENT_SERVER_MODE=${AGENT_SERVER_MODE:-<not set> (default: http)}"
    echo "  AGENT_SERVER_WORKER_API_URL=${AGENT_SERVER_WORKER_API_URL:-http://worker:2024}"
    echo ""
    echo "Usage: $0 {asgi|http|status} [worker_url]"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {asgi|http|status} [worker_url]"
    exit 1
    ;;
esac