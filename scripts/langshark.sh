#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# langshark.sh — smart langshark launcher
#
# Uses the local `langshark` CLI if available; otherwise falls back to
# Docker (ghcr.io/stokomax/langshark).  When falling back to Docker the
# script automatically:
#   - translates http://localhost:PORT → http://host.docker.internal:PORT
#   - adds --add-host host.docker.internal:host-gateway (required on Linux)
#   - maps port 8000 for the `web` sub-command
#
# Override the Docker image with the LANGSHARK_IMAGE env variable.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

DEFAULT_IMAGE="ghcr.io/stokomax/langshark"
IMAGE="${LANGSHARK_IMAGE:-$DEFAULT_IMAGE}"

# ── Detect local install ────────────────────────────────────────────
if command -v langshark &>/dev/null; then
    exec langshark "$@"
fi

# ── Docker fallback ─────────────────────────────────────────────────
ARGS=()
IS_WEB=false

for arg in "$@"; do
    # Translate localhost → host.docker.internal so the container can
    # reach the host machine.  Use sed with | delimiter because the
    # URLs contain forward slashes.
    arg=$(echo "$arg" | sed 's|http://localhost:|http://host.docker.internal:|g')
    ARGS+=("$arg")
done

# Detect web sub-command
for arg in "$@"; do
    if [[ "$arg" == "web" ]]; then
        IS_WEB=true
        break
    fi
done

DOCKER_OPTS=(
    -it --rm
    --add-host host.docker.internal:host-gateway
)

if [[ "$IS_WEB" == true ]]; then
    DOCKER_OPTS+=(-p 8000:8000)
fi

exec docker run "${DOCKER_OPTS[@]}" "$IMAGE" "${ARGS[@]}"