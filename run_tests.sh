#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="$LOG_DIR/test-run-$TIMESTAMP.log"

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
  if [ $status -ne 0 ]; then
    echo "Command failed with status $status" | tee -a "$LOG_FILE"
    exit $status
  fi
}

print_menu() {
  cat <<'EOF'
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
EOF
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

prompt_file() {
  local file
  read -r -p "Test file path (e.g., iopaint/tests/test_model.py): " file
  if [ -z "$file" ]; then
    echo "No file provided."
    exit 1
  fi
  echo "$file"
}

prompt_pattern() {
  local pattern
  read -r -p "Test name pattern (-k): " pattern
  if [ -z "$pattern" ]; then
    echo "No pattern provided."
    exit 1
  fi
  echo "$pattern"
}

prompt_custom() {
  local args
  read -r -p "Custom pytest args: " args
  if [ -z "$args" ]; then
    echo "No args provided."
    exit 1
  fi
  echo "$args"
}

prompt_frontend_script() {
  local script
  read -r -p "npm script (e.g., test, build, lint): " script
  if [ -z "$script" ]; then
    echo "No script provided."
    exit 1
  fi
  echo "$script"
}

main() {
  print_header

  local suite=${1:-}
  if [ -z "$suite" ]; then
    print_menu
    suite=$(choose_suite)
  fi

  case "$suite" in
    smoke)
      run_cmd "uv run pytest iopaint/tests/test_model.py -v"
      ;;
    full)
      run_cmd "uv run pytest -v"
      ;;
    file)
      local file
      file=$(prompt_file)
      run_cmd "uv run pytest $file -v"
      ;;
    k)
      local pattern
      pattern=$(prompt_pattern)
      run_cmd "uv run pytest -k \"$pattern\" -v"
      ;;
    custom)
      local args
      args=$(prompt_custom)
      run_cmd "uv run pytest $args"
      ;;
    fe-build)
      run_cmd "cd web_app && npm run build"
      ;;
    fe-lint)
      run_cmd "cd web_app && npm run lint"
      ;;
    fe-custom)
      local script
      script=$(prompt_frontend_script)
      run_cmd "cd web_app && npm run $script"
      ;;
    quit)
      echo "Bye."
      ;;
    *)
      echo "Invalid choice."
      exit 1
      ;;
  esac

  echo "Log saved to $LOG_FILE"
}

main "$@"
