#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run.sh — langstrata entrypoint
#
# Checks for `just` and delegates to it.  This handles the case where
# a new developer clones the repo without `just` installed and gets a
# helpful message instead of a cryptic "command not found" error.
#
# Usage:  ./run.sh <recipe> [args...]
#
# Examples
# --------
#   ./run.sh --list          # show available recipes
#   ./run.sh supervisor      # start single-server mode
#   ./run.sh supervisor-http  # start supervisor (dual-server)
#   ./run.sh worker           # start worker (dual-server)
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── CWD check ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$PWD" != "$SCRIPT_DIR" ]; then
    echo "error: Run run.sh from the project root" >&2
    echo "  cd $SCRIPT_DIR" >&2
    exit 1
fi

# ── Windows check ─────────────────────────────────────────────
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "langstrata is designed for Unix-like environments." >&2
        echo "It has not been tested on Windows / GitBash." >&2
        echo "" >&2
        echo "  Recommended: Install WSL2 and clone the repo inside WSL." >&2
        echo "  See https://learn.microsoft.com/en-us/windows/wsl/install" >&2
        echo "" >&2
        echo "  GitBash may work but is not yet tested or supported." >&2
        exit 1
        ;;
esac

# ── just check ─────────────────────────────────────────────────
if ! command -v just &>/dev/null; then
    echo "langstrata requires 'just' (a command runner)." >&2
    echo "" >&2
    echo "  Install it with one of:" >&2
    echo "    macOS:  brew install just" >&2
    echo "    Linux:  cargo install just" >&2
    echo "    or see https://github.com/casey/just#installation" >&2
    echo "" >&2
    echo "  Pre-built binaries:  https://github.com/casey/just/releases" >&2
    exit 1
fi

exec just "$@"