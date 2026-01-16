#!/usr/bin/env bash
# Terminate processes spawned by launch.sh
set -euo pipefail

BACKEND_PORT=${1:-8080}
FRONTEND_PORT=5173

echo "Stopping IOPaint servers..."

# Graceful shutdown first (SIGTERM), then force kill if needed
kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null || true)

  if [[ -n "$pids" ]]; then
    echo "Stopping processes on port $port (PIDs: $pids)..."
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 1

    # Force kill if still running
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      echo "Force killing remaining processes on port $port..."
      echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
  fi
}

kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

# Wait briefly for socket release
sleep 0.5

# Verify ports are free
if lsof -ti:"$BACKEND_PORT" >/dev/null 2>&1; then
  echo "Warning: Port $BACKEND_PORT still in use"
  exit 1
fi

echo "IOPaint servers stopped"
