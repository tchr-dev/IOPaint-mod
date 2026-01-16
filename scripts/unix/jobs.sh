#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

sub="${1:-}"
shift || true

usage() {
    cat <<'USAGE'
Usage:
  ./run.sh jobs cancel [--url URL] [--db PATH] [--dry-run]

Defaults:
  URL: http://127.0.0.1:8080
  DB:  ~/.iopaint/data/budget.db
USAGE
}

cancel_jobs() {
    local base_url="http://127.0.0.1:8080"
    local db_path="${HOME}/.iopaint/data/budget.db"
    local dry_run=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --url) base_url="${2:-}"; shift 2 ;;
            --db) db_path="${2:-}"; shift 2 ;;
            --dry-run) dry_run=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown argument: $1" ;;
        esac
    done

    need_cmd sqlite3 curl jq

    log "Cancelling stuck OpenAI generation jobs..."
    log "Server: $base_url"

    if [[ ! -f "$db_path" ]]; then
        die "Database not found at $db_path"
    fi

    local stuck
    stuck="$(sqlite3 "$db_path" "SELECT id, session_id FROM generation_jobs WHERE status IN ('running','queued');" || true)"

    if [[ -z "$stuck" ]]; then
        log "No running or queued jobs found in database."
    else
        local job_count
        job_count=$(echo "$stuck" | wc -l | tr -d ' ')
        log "Found $job_count stuck job(s)."

        local cancelled=0
        while IFS='|' read -r job_id session_id; do
            [[ -n "$job_id" ]] || continue
            log "Cancelling job: $job_id (session: ${session_id:0:8}...)"

            if [[ "$dry_run" == "1" ]]; then
                log "  (dry-run) would call: POST ${base_url}/api/v1/openai/jobs/${job_id}/cancel"
                continue
            fi

            local response
            response=$(curl -s -H "X-Session-Id: $session_id" -X POST "${base_url}/api/v1/openai/jobs/${job_id}/cancel" || true)
            local status
            status=$(echo "$response" | jq -r '.status // "error"' 2>/dev/null || echo "error")

            if [[ "$status" == "cancelled" ]]; then
                log "  Cancelled"
                ((cancelled++)) || true
            else
                warn "  Status: $status"
                warn "  Response: $response"
            fi
        done <<< "$stuck"

        log "Cancelled $cancelled backend job(s)."
    fi

    cat <<'NOTE'
IMPORTANT: Clear browser state to fix stuck 'Generating...'

Run this in browser console (F12 -> Console):

  localStorage.removeItem('ZUSTAND_STATE')
  location.reload()
NOTE
}

case "$sub" in
    cancel)
        cancel_jobs "$@"
        ;;
    -h|--help|help|"")
        usage
        exit 0
        ;;
    *)
        die "Unknown jobs subcommand: $sub"
        ;;
esac
