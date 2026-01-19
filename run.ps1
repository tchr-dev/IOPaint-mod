#!/usr/bin/env pwsh
# IOPaint Wrapper Script (Windows)
# Delegates all logic to manage.py

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Error "Error: 'uv' is not installed."
    Write-Host "Please install it via: irm https://astral.sh/uv/install.ps1 | iex"
    exit 1
}

# Delegate to Python CLI
# Using --with to compile ephemeral env if needed, though project env is preferred.
uv run --with questionary --with typer --with rich python manage.py $args
