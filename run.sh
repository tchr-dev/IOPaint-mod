#!/usr/bin/env bash
set -euo pipefail

# IOPaint Wrapper Script
# Delegates all logic to manage.py

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    echo "Please install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Delegate to Python CLI
# We use 'uv run' to ensure dependencies (typer, rich, etc.) are available.
# We explicitly install the 'dev' group to ensure questionary is available.
# But 'uv run' should handle the environment.
exec uv run --with questionary --with typer --with rich python manage.py "$@"
