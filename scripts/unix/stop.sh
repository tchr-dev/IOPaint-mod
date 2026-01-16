#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

BACKEND_PORT="${IOPAINT_PORT:-8080}"
FRONTEND_PORT="${IOPAINT_FRONTEND_PORT:-5173}"

usage() {
    cat <<'USAGE'
Usage: ./run.sh stop [--backend-port PORT] [--frontend-port PORT]

Stops any LISTENing processes bound to the given ports.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend-port) BACKEND_PORT="${2:-}"; shift 2 ;;
        --frontend-port) FRONTEND_PORT="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

log "Stopping IOPaint servers..."
kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

sleep 0.5

if port_in_use "$BACKEND_PORT"; then
    warn "Port $BACKEND_PORT is still in use. Owner:"
    show_port_owner "$BACKEND_PORT"
    exit 1
fi

log "IOPaint servers stopped"
