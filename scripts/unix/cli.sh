#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

usage() {
    cat <<'USAGE'
IOPaint CLI - Interactive main menu.

Usage:
  ./run.sh cli               # Launch interactive main menu
  ./run.sh cli --help        # Show this help

The menu provides access to all development, testing, production, and utility commands.
USAGE
}

use_python_cli() {
    # Check if uv run, questionary are available, and terminal is interactive
    [[ -t 0 && -t 1 ]] && command -v uv >/dev/null 2>&1 && uv run python3 -c "import questionary" >/dev/null 2>&1
}

python_cli_menu() {
    local choice=""
    local menu_type="main"

    while true; do
        # Run Python CLI
        if ! choice=$(uv run python3 "$SCRIPT_DIR/cli_menu.py" "$menu_type" 2>/dev/null); then
            # Python CLI failed, fallback to bash
            return 1
        fi

        case "$choice" in
            quit)
                echo "Goodbye!"
                return 0
                ;;
            back)
                menu_type="main"
                continue
                ;;
            invalid)
                echo "Invalid selection, please try again."
                sleep 1
                continue
                ;;
            *.py)
                # Direct file selection from test file menu
                echo "$choice"
                return 0
                ;;
            [a-z]*)
                # Direct command
                echo "$choice"
                return 0
                ;;
            [0-9]*)
                # Menu selection
                case "$menu_type" in
                    main)
                        case "$choice" in
                            1) menu_type="development" ;;
                            2) menu_type="test-main" ;;
                            3) menu_type="production" ;;
                            4) menu_type="utilities" ;;
                            5) echo "help" && return 0 ;;
                            6) echo "quit" && return 0 ;;
                        esac
                        ;;
                    test-main)
                        case "$choice" in
                            1) echo "list" && return 0 ;;
                            2) echo "smoke" && return 0 ;;
                            3) echo "full" && return 0 ;;
                            4) menu_type="test-files" ;;
                            5) echo "k" && return 0 ;;
                            6) echo "custom" && return 0 ;;
                            7) echo "test-lint" && return 0 ;;
                            8) echo "fe-build" && return 0 ;;
                            9) echo "fe-lint" && return 0 ;;
                            10) echo "fe-custom" && return 0 ;;
                            11) echo "help" && return 0 ;;
                        esac
                        ;;
                esac
                ;;
            *)
                echo "Unexpected choice: $choice"
                return 1
                ;;
        esac
    done
}



main() {
    if [[ "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi

    if [[ ! -t 0 || ! -t 1 ]]; then
        echo "CLI requires an interactive terminal." >&2
        echo "" >&2
        echo "Available commands:" >&2
        echo "  ./run.sh dev           Start backend + Vite dev server" >&2
        echo "  ./run.sh prod          Build frontend, copy assets, start backend" >&2
        echo "  ./run.sh stop          Stop backend/frontend by ports" >&2
        echo "  ./run.sh build         Build frontend only" >&2
        echo "  ./run.sh test          Run tests (interactive or named suite)" >&2
        echo "  ./run.sh jobs          Job utilities (cancel)" >&2
        echo "  ./run.sh docker        Docker utilities (build)" >&2
        echo "  ./run.sh publish       Build frontend assets + python dist" >&2
        echo "  ./run.sh --help        Show all commands" >&2
        exit 1
    fi

    while true; do
        if ! choice=$(python_cli_menu); then
            echo "Interactive CLI not available. Use specific commands instead." >&2
            exit 1
        fi

        case "$choice" in
            quit)
                echo "Goodbye!"
                exit 0
                ;;
            back)
                continue
                ;;
            help)
                usage
                continue
                ;;
            invalid)
                echo "Invalid choice. Please try again."
                sleep 1
                continue
                ;;
            *.py)
                # Test file selected
                echo "$choice"
                exit 0
                ;;
            [a-z]*)
                # Command selected
                echo "$choice"
                exit 0
                ;;
        esac
    done
}

main "$@"