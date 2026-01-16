#!/usr/bin/env bash
set -euo pipefail

# Shared helpers for IOPaint Unix scripts.

log()  { echo "[iopaint] $*"; }
warn() { echo "[iopaint][WARN] $*" >&2; }
die()  { echo "[iopaint][ERROR] $*" >&2; exit 1; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

need_cmd() {
    local c
    for c in "$@"; do
        if ! has_cmd "$c"; then
            die "Missing required command: $c"
        fi
    done
}

project_root() {
    local start="${1:-$(pwd)}"
    local d="$start"
    while [[ "$d" != "/" ]]; do
        if [[ -f "$d/main.py" ]]; then
            echo "$d"
            return 0
        fi
        d="$(dirname "$d")"
    done
    return 1
}

cd_project_root() {
    local here
    here="$(pwd)"
    local root
    if root="$(project_root "$here")"; then
        cd "$root"
        return 0
    fi

    local this_dir
    this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local candidate
    candidate="$(cd "$this_dir/../.." && pwd)"
    if [[ -f "$candidate/main.py" ]]; then
        cd "$candidate"
        return 0
    fi

    die "Could not locate project root (main.py). Run from within the repo."
}

npm_install_if_needed() {
    local app_dir="$1"
    local force="${2:-0}"

    need_cmd npm

    if [[ ! -d "$app_dir" ]]; then
        die "Frontend directory not found: $app_dir"
    fi

    if [[ "$force" != "1" && -d "$app_dir/node_modules" ]]; then
        return 0
    fi

    (
        cd "$app_dir"
        if [[ -f package-lock.json ]]; then
            log "Installing frontend deps (npm ci)..."
            npm ci
        else
            log "Installing frontend deps (npm install)..."
            npm install
        fi
    )
}

kill_port() {
    local port="$1"

    if has_cmd lsof; then
        local pids
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
        if [[ -n "$pids" ]]; then
            log "Stopping processes on port $port (PIDs: $pids)..."
            echo "$pids" | xargs kill 2>/dev/null || true
            sleep 1
            pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
            if [[ -n "$pids" ]]; then
                warn "Force killing remaining processes on port $port..."
                echo "$pids" | xargs kill -9 2>/dev/null || true
            fi
        fi
        return 0
    fi

    if has_cmd fuser; then
        log "Stopping processes on port $port (using fuser)..."
        fuser -k "${port}/tcp" 2>/dev/null || true
        return 0
    fi

    warn "Neither lsof nor fuser available; cannot kill by port ($port)."
}

port_in_use() {
    local port="$1"
    if has_cmd lsof; then
        lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
        return $?
    fi
    if has_cmd fuser; then
        fuser "${port}/tcp" >/dev/null 2>&1
        return $?
    fi
    return 1
}

show_port_owner() {
    local port="$1"
    if has_cmd lsof; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
    elif has_cmd fuser; then
        fuser -v "${port}/tcp" || true
    fi
}
