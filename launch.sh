#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
shift || true

MODEL="openai-compat"
PORT="8080"

usage() {
  cat <<'EOF'
Usage: ./launch.sh <dev|prod> [--model MODEL] [--port PORT]

Modes:
  dev   Start backend + Vite dev server
  prod  Build frontend, copy assets, start backend

Options:
  --model   Model name (default: openai-compat)
  --port    Backend port (default: 8080)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL=${2:-}
      shift 2
      ;;
    --port)
      PORT=${2:-}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  usage
  exit 1
fi

run_backend() {
  uv run python main.py start --model "$MODEL" --port "$PORT"
}

run_frontend_dev() {
  (cd web_app && npm install && npm run dev)
}

run_frontend_build() {
  (cd web_app && npm install && npm run build)
}

case "$MODE" in
  dev)
    echo "Starting backend (model=$MODEL, port=$PORT) and frontend dev server..."
    uv sync
    run_backend &
    BACKEND_PID=$!
    run_frontend_dev &
    FRONTEND_PID=$!

    cleanup() {
      echo "Stopping servers..."
      kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    wait "$BACKEND_PID" "$FRONTEND_PID"
    ;;
  prod)
    echo "Building frontend and starting backend (model=$MODEL, port=$PORT)..."
    uv sync
    run_frontend_build
    mkdir -p iopaint/web_app
    rm -rf iopaint/web_app/*
    cp -r web_app/dist/* iopaint/web_app/
    run_backend
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac
