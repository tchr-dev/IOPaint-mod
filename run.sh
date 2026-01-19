#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/lib/common.sh"

usage() {
    cat <<'USAGE'
IOPaint CLI - Unified development workflow.

Usage:
  ./run.sh <command> [args]

Commands:
   cli           Interactive main menu (default when no args)
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
   "")
     # Default: launch interactive CLI
     cd_project_root
     exec "$SCRIPT_DIR/scripts/unix/cli.sh" "$@"
     ;;
   cli|dev|prod|stop|build|test|jobs|docker|publish)
     cd_project_root
     exec "$SCRIPT_DIR/scripts/unix/${cmd}.sh" "$@"
     ;;
   *)
     die "Unknown command: $cmd. Run --help."
     ;;
esac
