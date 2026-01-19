#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

MODEL="${IOPAINT_MODEL:-lama}"
PORT="${IOPAINT_PORT:-8080}"
VERBOSE="${IOPAINT_VERBOSE:-}"
FRONTEND_PORT="${IOPAINT_FRONTEND_PORT:-5173}"
NO_SYNC=0
NO_NPM=0
FORCE_NPM=0

usage() {
    cat <<'USAGE'
Usage: ./run.sh dev [--model MODEL] [--port PORT] [--frontend-port PORT] [--verbose] [--no-sync] [--no-npm] [--npm-force]

Starts:
  - Backend: uv run python main.py start --model ... --port ...
  - Frontend: Vite dev server (web_app)

Options:
  --model           Model name (default: lama)
  --port            Backend port (default: 8080)
  --frontend-port   Frontend port hint for messaging (default: 5173)
  --verbose, -v     Enable verbose logging (sets IOPAINT_VERBOSE)
  --no-sync         Skip 'uv sync'
  --no-npm          Skip npm install step
  --npm-force       Force reinstall frontend deps
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --port) PORT="${2:-}"; shift 2 ;;
        --frontend-port) FRONTEND_PORT="${2:-}"; shift 2 ;;
        --verbose|-v) VERBOSE="1"; shift ;;
        --no-sync) NO_SYNC=1; shift ;;
        --no-npm) NO_NPM=1; shift ;;
        --npm-force) FORCE_NPM=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

need_cmd uv node npm

if [[ "$NO_SYNC" != "1" ]]; then
    log "Syncing python deps (uv sync)..."
    uv sync
fi

if [[ "$NO_NPM" != "1" ]]; then
    npm_install_if_needed "web_app" "$FORCE_NPM"
fi

log "Starting backend (model=$MODEL, port=$PORT) and frontend dev server..."
log "Frontend will typically be at http://127.0.0.1:${FRONTEND_PORT}"

IOPAINT_VERBOSE="$VERBOSE" uv run python main.py start --model "$MODEL" --port "$PORT" &
BACKEND_PID=$!

( cd web_app && npm run dev ) &
FRONTEND_PID=$!

cleanup() {
    warn "Stopping servers..."
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$BACKEND_PID" "$FRONTEND_PID"
