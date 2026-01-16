#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

MODEL="${IOPAINT_MODEL:-openai-compat}"
PORT="${IOPAINT_PORT:-8080}"
VERBOSE="${IOPAINT_VERBOSE:-}"
NO_SYNC=0
NO_NPM=0
FORCE_NPM=0

usage() {
    cat <<'USAGE'
Usage: ./run.sh prod [--model MODEL] [--port PORT] [--verbose] [--no-sync] [--no-npm] [--npm-force]

Builds frontend (web_app/dist), copies it into iopaint/web_app/, then starts backend.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --port) PORT="${2:-}"; shift 2 ;;
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

log "Building frontend..."
( cd web_app && rm -rf dist && npm run build )

log "Copying frontend assets into iopaint/web_app/..."
mkdir -p iopaint/web_app
rm -rf iopaint/web_app/*
cp -r web_app/dist/* iopaint/web_app/

log "Starting backend (model=$MODEL, port=$PORT)..."
IOPAINT_VERBOSE="$VERBOSE" uv run python main.py start --model "$MODEL" --port "$PORT"
