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

DEBUG="${DEBUG:-0}"

# Parse debug flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--debug)
            DEBUG=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

export DEBUG

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
          *.py)
            # Handle single file selection
            if [[ -n "$choice" ]]; then
              if [[ "$DEBUG" == "1" ]]; then
                echo "DEBUG: Running test file: $choice" >&2
              fi
              uv run python main.py test file "iopaint/tests/$choice"
            fi
            menu_type="main"
            ;;
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
                  1) menu_type="test-files" ;;
                  2) exec uv run python main.py test smoke ;;
                  3) exec uv run python main.py test full ;;
                  4) exec uv run python main.py test k ;;
                  5) exec uv run python main.py test custom ;;
                  6) exec uv run python main.py test test-lint ;;
                  7) exec uv run python main.py test fe-build ;;
                  8) exec uv run python main.py test fe-lint ;;
                  9) exec uv run python main.py test fe-custom ;;
                  10) usage ;;
                esac
               ;;
           esac
           ;;
         *) echo "Unexpected choice: $choice" ;;
       esac
     done
     ;;
    cli|dev|prod|stop|build|jobs|docker|publish)
      cd_project_root
      exec "$SCRIPT_DIR/scripts/unix/${cmd}.sh" "$@"
      ;;
    test)
      cd_project_root
      exec uv run python main.py test "$@"
      ;;
   *)
     die "Unknown command: $cmd. Run --help."
     ;;
esac
