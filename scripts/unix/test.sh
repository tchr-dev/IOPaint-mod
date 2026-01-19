#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="$LOG_DIR/test-run-$TIMESTAMP.log"

usage() {
    cat <<'USAGE'
Usage:
  ./run.sh test               # interactive menu
  ./run.sh test smoke|full|file|k|custom|fe-build|fe-lint|fe-custom [...]

Examples:
  ./run.sh test smoke
  ./run.sh test file iopaint/tests/test_model.py
  ./run.sh test k "Model"
  ./run.sh test custom "-k Model -v"
USAGE
}

print_header() {
    echo ""
    echo "Friendly Test Runner"
    echo "Log file: $LOG_FILE"
    echo ""
}

run_cmd() {
    local cmd="$1"
    echo "Running: $cmd" | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
    set +e
    bash -lc "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}
    set -e
    echo "---" | tee -a "$LOG_FILE"
    if [[ $status -ne 0 ]]; then
        echo "Command failed with status $status" | tee -a "$LOG_FILE"
        exit $status
    fi
}

print_menu() {
    cat <<'MENU'
Select a test suite:
  1) Backend smoke (iopaint/tests/test_model.py)
  2) Backend full (pytest -v)
  3) Backend single test file
  4) Backend single test name (-k)
  5) Backend custom pytest args
  6) Frontend build (npm run build)
  7) Frontend lint (npm run lint)
  8) Frontend custom npm script
  9) Quit
MENU
}

choose_suite() {
    local choice
    read -r -p "Enter choice [1-9]: " choice
    case "$choice" in
        1) echo "smoke" ;;
        2) echo "full" ;;
        3) echo "file" ;;
        4) echo "k" ;;
        5) echo "custom" ;;
        6) echo "fe-build" ;;
        7) echo "fe-lint" ;;
        8) echo "fe-custom" ;;
        9) echo "quit" ;;
        *) echo "invalid" ;;
    esac
}

main() {
    local suite="${1:-}"
    shift || true

    if [[ "$suite" == "-h" || "$suite" == "--help" ]]; then
        usage
        exit 0
    fi

    print_header

    if [[ -z "$suite" ]]; then
        print_menu
        suite="$(choose_suite)"
    fi

    case "$suite" in
        smoke)
            run_cmd "uv run pytest iopaint/tests/test_model.py -v"
            ;;
        full)
            run_cmd "uv run pytest -v"
            ;;
        file)
            local file="${1:-}"
            [[ -n "$file" ]] || die "Provide test file path."
            run_cmd "uv run pytest $file -v"
            ;;
        k)
            local pattern="${1:-}"
            [[ -n "$pattern" ]] || die "Provide -k pattern."
            run_cmd "uv run pytest -k \"$pattern\" -v"
            ;;
        custom)
            local args="${1:-}"
            [[ -n "$args" ]] || die "Provide custom pytest args (quoted)."
            run_cmd "uv run pytest $args"
            ;;
        fe-build)
            run_cmd "cd web_app && npm run build"
            ;;
        fe-lint)
            run_cmd "cd web_app && npm run lint"
            ;;
        fe-custom)
            local script="${1:-}"
            [[ -n "$script" ]] || die "Provide npm script name."
            run_cmd "cd web_app && npm run $script"
            ;;
        quit)
            echo "Bye."
            ;;
        invalid|*)
            die "Invalid choice/suite. Use: ./run.sh test --help"
            ;;
    esac

    echo "Log saved to $LOG_FILE"
}

main "$@"
