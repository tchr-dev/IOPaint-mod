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
  ./run.sh test help          # show detailed help for all subcommands
  ./run.sh test list          # list all available Python test files
  ./run.sh test fe-list       # list all available frontend test files
  ./run.sh test fe-test       # run all frontend tests
  ./run.sh test fe-test <file> # run specific frontend test file
  ./run.sh test smoke|full|file|k|custom|fe-build|fe-lint|fe-custom|test-lint [...]

Examples:
  ./run.sh smoke
  ./run.sh test file iopaint/tests/test_model.py
  ./run.sh test k "Model"
  ./run.sh test custom "-k Model -v"
  ./run.sh test list
  ./run.sh test fe-list
  ./run.sh test fe-test
  ./run.sh test fe-test CanvasView
USAGE
}

list_tests() {
    echo "Available Test Files in iopaint/tests/"
    echo "======================================"
    echo ""

    # Find all test files and sort them
    local test_files
    test_files=$(find iopaint/tests -name "test_*.py" -type f 2>/dev/null | sort)

    if [[ -z "$test_files" ]]; then
        echo "No test files found in iopaint/tests/"
        echo "Current directory: $(pwd)"
        echo "Tests directory exists: $(test -d iopaint/tests && echo 'yes' || echo 'no')"
        return
    fi

    local count=0
    while IFS= read -r file; do
        ((count++))
        local filename
        filename=$(basename "$file")

        # Try to extract a brief description from the file
        local description=""
        if [[ -f "$file" ]]; then
            # Look for class docstrings or module docstrings
            description=$(head -20 "$file" 2>/dev/null | grep -E '"""|class.*Test.*:' | head -1 | sed 's/.*"""//' | sed 's/.*class //' | sed 's/:.*//' | tr -d '"' || true)
            if [[ -z "$description" ]]; then
                # Fallback to extracting from filename
                description=$(echo "$filename" | sed 's/test_//' | sed 's/\.py//' | sed 's/_/ /g' | sed 's/\b\w/\U&/g')
            fi
        fi

        printf "%2d) %-35s - %s\n" "$count" "$filename" "${description:-Test file}"
    done <<< "$test_files"

    echo ""
    echo "Total: $count test files"
    echo ""
    echo "Run specific tests with:"
    echo "  ./run.sh test file iopaint/tests/<filename>"
    echo "  ./run.sh test k \"<pattern>\""
}

list_fe_tests() {
    echo "Available Frontend Test Files in web_app/src/"
    echo "=============================================="
    echo ""

    # Find all test files in web_app
    local test_files
    test_files=$(find web_app/src -name "*.test.tsx" -type f 2>/dev/null | sort)

    if [[ -z "$test_files" ]]; then
        echo "No test files found in web_app/src/"
        echo "Current directory: $(pwd)"
        echo "web_app/src directory exists: $(test -d web_app/src && echo 'yes' || echo 'no')"
        return
    fi

    local count=0
    while IFS= read -r file; do
        ((count++))
        local filename
        filename=$(basename "$file")

        # Try to extract a brief description from the file
        local description=""
        if [[ -f "$file" ]]; then
            # Look for describe block in first 30 lines
            description=$(head -30 "$file" 2>/dev/null | grep -E 'describe\(' | head -1 | sed 's/.*describe[[:space:]]*([[:space:]]*["\x27]//' | sed 's/["\x27].*//' | tr -d ' ' || true)
            if [[ -z "$description" ]]; then
                description=$(echo "$filename" | sed 's/\.test\.tsx//' | sed 's/-/ /g' | sed 's/\b\w/\U&/g')
            fi
        fi

        # Get relative path
        local relpath="${file#web_app/src/}"
        printf "   %-50s - %s\n" "$relpath" "${description:-Test}"
    done <<< "$test_files"

    echo ""
    echo "Total: $count frontend test files"
    echo ""
    echo "Run frontend tests with:"
    echo "  ./run.sh test fe-test                  # Run all frontend tests"
    echo "  ./run.sh test fe-test <file>           # Run specific test"
    echo "  ./run.sh test fe-test -k \"<pattern>\"  # Run tests matching pattern"
    echo ""
    echo "Or directly:"
    echo "  cd web_app && npm run test"
}

help_detailed() {
    cat <<'HELP'
Detailed Test Runner Help
=========================

SUBCOMMANDS:

  list          List all available Python test files
                Shows all test files in iopaint/tests/ with descriptions
                Use: See what tests are available before running them

  fe-list       List all available frontend test files
                Shows all test files in web_app/src/ with descriptions
                Use: See what frontend tests are available

  fe-test       Run all frontend tests
                Runs: cd web_app && npm run test
                Use: Run all Vitest tests in the frontend

  fe-test <file> Run specific frontend test file
                Runs: cd web_app && npm run test -- <file>
                Use: Run a single test file, e.g., fe-test CanvasView.test.tsx

  fe-test -k <pattern> Run frontend tests matching pattern
                Runs: cd web_app && npm run test -- -k "<pattern>"
                Use: Run tests by name pattern, e.g., fe-test -k "Canvas"

  smoke         Run backend smoke test (single test file)
                Runs: pytest iopaint/tests/test_model.py -v
                Use: Quick sanity check for basic functionality

  full          Run all backend tests
                Runs: pytest -v
                Use: Comprehensive testing of all backend functionality

  file <path>   Run specific test file
                Runs: pytest <path> -v
                Use: Test a specific file, e.g., ./run.sh test file iopaint/tests/test_budget_limits.py

  k <pattern>   Run tests matching pattern (-k flag)
                Runs: pytest -k "<pattern>" -v
                Use: Run tests by name pattern, e.g., ./run.sh test k "budget"

  custom <args> Run pytest with custom arguments
                Runs: pytest <args>
                Use: Full control over pytest, e.g., ./run.sh test custom "-k budget --tb=short"

  test-lint     Lint test files
                Runs: ruff check iopaint/tests/ && ruff format --check iopaint/tests/
                Use: Check test code quality and formatting

  fe-build      Build frontend
                Runs: cd web_app && npm run build
                Use: Build frontend production assets

  fe-lint       Lint frontend code
                Runs: cd web_app && npm run lint
                Use: Check frontend code quality

  fe-custom <script> Run custom frontend npm script
                       Runs: cd web_app && npm run <script>
                       Use: Run any frontend script, e.g., ./run.sh test fe-custom test

EXAMPLES:

  # List tests
  ./run.sh test list
  ./run.sh test fe-list

  # Run backend tests
  ./run.sh test smoke
  ./run.sh test full
  ./run.sh test k "budget"
  ./run.sh test file iopaint/tests/test_api_error_handling.py

  # Run frontend tests
  ./run.sh test fe-test
  ./run.sh test fe-test CanvasView.test.tsx
  ./run.sh test fe-test -k "Canvas"

  # Lint
  ./run.sh test test-lint
  ./run.sh test fe-lint

  # Custom pytest options
  ./run.sh test custom "--tb=short -x"
  ./run.sh test custom "-k 'test_budget' --cov=iopaint"

  # Frontend operations
  ./run.sh test fe-build
  ./run.sh test fe-custom test

LOGGING:
  All test runs are logged to ./logs/test-run-YYYYMMDD-HHMMSS.log
  Check the log file for detailed output and debugging information
HELP
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

run_cmd_allow_fail() {
    local cmd="$1"
    echo "Running: $cmd" | tee -a "$LOG_FILE"
    echo "---" | tee -a "$LOG_FILE"
    bash -lc "$cmd" 2>&1 | tee -a "$LOG_FILE"
    local status=${PIPESTATUS[0]}
    echo "---" | tee -a "$LOG_FILE"
    if [[ $status -ne 0 ]]; then
        echo "Command completed with status $status (allowed to fail)" | tee -a "$LOG_FILE"
    fi
}

print_menu() {
    cat <<'MENU'
Select a test suite:
  1) List Python test files
  2) List frontend test files
  3) Backend smoke (iopaint/tests/test_model.py)
  4) Backend full (pytest -v)
  5) Backend single test file
  6) Backend single test name (-k)
  7) Backend custom pytest args
  8) Frontend all tests
  9) Frontend lint (npm run lint)
  10) Frontend build (npm run build)
  11) Test lint (ruff check/format)
  12) Help (detailed help)
  13) Quit
MENU
}

choose_suite() {
    local choice
    read -r -p "Enter choice [1-13]: " choice
    case "$choice" in
        1) echo "list" ;;
        2) echo "fe-list" ;;
        3) echo "smoke" ;;
        4) echo "full" ;;
        5) echo "file" ;;
        6) echo "k" ;;
        7) echo "custom" ;;
        8) echo "fe-test" ;;
        9) echo "fe-lint" ;;
        10) echo "fe-build" ;;
        11) echo "test-lint" ;;
        12) echo "help" ;;
        13) echo "quit" ;;
        *) echo "invalid" ;;
    esac
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
        if ! choice=$(uv run python3 "$SCRIPT_DIR/test_menu.py" "$menu_type" 2>/dev/null); then
            # Python CLI failed, fallback to bash
            return 1
        fi

        case "$choice" in
            quit)
                echo "quit"
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
                # Direct file selection from Python CLI
                echo "$choice"
                return 0
                ;;
            [0-9]*)
                # Main menu selection
                case "$choice" in
                    1) menu_type="test-files" ;;
                    2) echo "fe-list" && return 0 ;;
                    3) echo "smoke" && return 0 ;;
                    4) echo "full" && return 0 ;;
                    5) echo "file" && return 0 ;;
                    6) echo "k" && return 0 ;;
                    7) echo "custom" && return 0 ;;
                    8) echo "fe-test" && return 0 ;;
                    9) echo "fe-lint" && return 0 ;;
                    10) echo "fe-build" && return 0 ;;
                    11) echo "test-lint" && return 0 ;;
                    12) echo "help" && return 0 ;;
                    13) echo "quit" && return 0 ;;
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
    local suite="${1:-}"
    shift || true

    if [[ "$suite" == "-h" || "$suite" == "--help" ]]; then
        usage
        exit 0
    fi

    print_header

    if [[ -z "$suite" ]]; then
        # Try Python CLI first, fallback to bash menu
        if use_python_cli; then
            suite="$(python_cli_menu)"
        else
            print_menu
            suite="$(choose_suite)"
        fi
    fi

    case "$suite" in
        list)
            list_tests
            ;;
        fe-list)
            list_fe_tests
            ;;
        fe-test)
            shift
            if [[ -z "$*" ]]; then
                run_cmd "cd web_app && npm run test"
            else
                run_cmd "cd web_app && npm run test -- $*"
            fi
            ;;
        *.py)
            # Direct file selection from Python CLI
            run_cmd "uv run pytest \"iopaint/tests/$suite\" -v"
            ;;
        smoke)
            run_cmd "uv run pytest iopaint/tests/test_model.py -v"
            ;;
        full)
            run_cmd "uv run pytest -v"
            ;;
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
        help)
            help_detailed
            ;;
        test-lint)
            echo "Running: uv run ruff check iopaint/tests/ && uv run ruff format --check iopaint/tests/" | tee -a "$LOG_FILE"
            echo "---" | tee -a "$LOG_FILE"
            bash -c "uv run ruff check iopaint/tests/; uv run ruff format --check iopaint/tests/" 2>&1 | tee -a "$LOG_FILE"
            echo "---" | tee -a "$LOG_FILE"
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
