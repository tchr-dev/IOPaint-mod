#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/lib/common.sh"

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

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
     menu_type="main"
     while true; do
       # Use temp file for choice output to avoid capturing menu display
       if has_cmd mktemp; then
         temp_file=$(mktemp) || { echo "Failed to create temp file" >&2; exit 1; }
         if uv run python3 "$SCRIPT_DIR/scripts/unix/cli_menu.py" "$menu_type" "$temp_file"; then
           choice=$(cat "$temp_file" 2>/dev/null | tr -d '\r')
           rm -f "$temp_file"
           if [[ -z "$choice" ]]; then
             echo "Interactive CLI not available. Use --noninteractive for command list." >&2
             exit 1
           fi
         else
           rm -f "$temp_file"
           echo "Interactive CLI failed. Use --noninteractive for command list." >&2
           exit 1
         fi
       else
         echo "Interactive CLI requires 'mktemp'. Use --noninteractive for command list." >&2
         exit 1
       fi
       case "$choice" in
         quit) exit 0 ;;
         back) menu_type="main" ;;
         help) usage ;;
         invalid) echo "Invalid selection, please try again." ;;
         *.py) exec "$SCRIPT_DIR/scripts/unix/test.sh" file "$choice" ;;
         [a-z]*)
           case "$choice" in
             dev|prod|stop|build|test|jobs|docker|publish)
               exec "$SCRIPT_DIR/scripts/unix/${choice}.sh" ;;
             *) echo "Unknown command: $choice" ;;
           esac
           ;;
         [0-9]*)
           case "$menu_type" in
             main)
               case "$choice" in
                 1) menu_type="development" ;;
                 2) menu_type="test-main" ;;
                 3) menu_type="production" ;;
                 4) menu_type="utilities" ;;
                 5) usage ;;
                 6) exit 0 ;;
               esac
               ;;
             test-main)
               case "$choice" in
                 1) exec "$SCRIPT_DIR/scripts/unix/test.sh" list ;;
                 2) exec "$SCRIPT_DIR/scripts/unix/test.sh" smoke ;;
                 3) exec "$SCRIPT_DIR/scripts/unix/test.sh" full ;;
                 4) menu_type="test-files" ;;
                 5) exec "$SCRIPT_DIR/scripts/unix/test.sh" k ;;
                 6) exec "$SCRIPT_DIR/scripts/unix/test.sh" custom ;;
                 7) exec "$SCRIPT_DIR/scripts/unix/test.sh" test-lint ;;
                 8) exec "$SCRIPT_DIR/scripts/unix/test.sh" fe-build ;;
                 9) exec "$SCRIPT_DIR/scripts/unix/test.sh" fe-lint ;;
                 10) exec "$SCRIPT_DIR/scripts/unix/test.sh" fe-custom ;;
                 11) usage ;;
               esac
               ;;
           esac
           ;;
         *) echo "Unexpected choice: $choice" ;;
       esac
     done
     ;;
   cli|dev|prod|stop|build|test|jobs|docker|publish)
     cd_project_root
     exec "$SCRIPT_DIR/scripts/unix/${cmd}.sh" "$@"
     ;;
   *)
     die "Unknown command: $cmd. Run --help."
     ;;
esac
