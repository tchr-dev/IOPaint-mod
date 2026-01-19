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

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

python_cli_menu() {
    local choice=""
    local menu_type="main"

    while true; do
        # Check if we have a terminal for interactive menu
        if [[ ! -t 0 ]]; then
            return 1
        fi

        # Run Python CLI.  When wrapped in command substitution, stdout is no longer a TTY,
        # causing questionary/prompt_toolkit to disable the interactive UI.  To preserve a
        # pseudo‑terminal for the Python process while still capturing its output, use `script`.
        # On macOS, script doesn't support -c, so use a temp file approach.
        if has_cmd script && has_cmd mktemp; then
            local temp_file
            temp_file=$(mktemp) || return 1
            # Run the Python script inside script to allocate PTY and record output
            if script -q "$temp_file" uv run python3 "$SCRIPT_DIR/cli_menu.py" "$menu_type"; then
                # Read the last line from the temp file as the choice
                choice=$(tail -n 1 "$temp_file" 2>/dev/null | tr -d '\r')
                rm -f "$temp_file"
                if [[ -z "$choice" ]]; then
                    return 1
                fi
            else
                rm -f "$temp_file"
                return 1
            fi
        else
            # Fallback: run without script.  This may fail if stdout is not a TTY.
            if ! choice=$(uv run python3 "$SCRIPT_DIR/cli_menu.py" "$menu_type" 2>/dev/null); then
                return 1
            fi
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