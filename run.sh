#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/lib/common.sh"

usage() {
    cat <<'USAGE'
IOPaint CLI - Unified development workflow.

Usage:
  ./run.sh [options] [command] [args]

Options:
  -N, --noninteractive    Show available commands (non-interactive mode)

Commands:
   dev           Start backend + Vite dev server
   prod          Build frontend, copy assets, start backend
   stop          Stop backend/frontend by ports
   build         Build frontend only
   test          Run tests (interactive or named suite)
   jobs          Job utilities (cancel)
   docker        Docker utilities (build)
   publish       Build frontend assets + python dist

Run:
  ./run.sh <command> --help
USAGE
}

cmd="${1:-}"
shift || true

case "$cmd" in
   -h|--help|help)
     usage
     exit 0
     ;;
   -N|--noninteractive)
     # Non-interactive mode: show command list
     echo "Available commands:"
     echo "  ./run.sh dev           Start backend + Vite dev server"
     echo "  ./run.sh prod          Build frontend, copy assets, start backend"
     echo "  ./run.sh stop          Stop backend/frontend by ports"
     echo "  ./run.sh build         Build frontend only"
     echo "  ./run.sh test          Run tests (interactive or named suite)"
     echo "  ./run.sh jobs          Job utilities (cancel)"
     echo "  ./run.sh docker        Docker utilities (build)"
     echo "  ./run.sh publish       Build frontend assets + python dist"
     echo "  ./run.sh --help        Show all commands"
     exit 0
     ;;
   "")
     # Default: launch interactive CLI menu
     cd_project_root
     exec uv run python3 "$SCRIPT_DIR/scripts/unix/cli_menu.py" main
     ;;
   cli|dev|prod|stop|build|test|jobs|docker|publish)
     cd_project_root
     exec "$SCRIPT_DIR/scripts/unix/${cmd}.sh" "$@"
     ;;
   *)
     die "Unknown command: $cmd. Run --help."
     ;;
esac
