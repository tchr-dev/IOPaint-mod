#!/usr/bin/env bash
# Cancel all running/queued OpenAI generation jobs
set -euo pipefail

BASE_URL=${1:-http://127.0.0.1:8080}
DB_PATH="${HOME}/.iopaint/data/budget.db"

echo "Cancelling stuck OpenAI generation jobs..."
echo "Server: $BASE_URL"
echo ""

# Check if database exists
if [[ ! -f "$DB_PATH" ]]; then
  echo "Error: Database not found at $DB_PATH"
  exit 1
fi

# Query database directly for stuck jobs (bypasses session issues)
stuck_jobs=$(sqlite3 "$DB_PATH" "SELECT id, session_id FROM generation_jobs WHERE status IN ('running', 'queued');")

if [[ -z "$stuck_jobs" ]]; then
  echo "No running or queued jobs found in database."
else
  job_count=$(echo "$stuck_jobs" | wc -l | tr -d ' ')
  echo "Found $job_count stuck job(s) to cancel."
  echo ""

  cancelled=0
  while IFS='|' read -r job_id session_id; do
    echo "Cancelling job: $job_id (session: ${session_id:0:8}...)"

    # Cancel via API with correct session header
    response=$(curl -s -H "X-Session-Id: $session_id" -X POST "${BASE_URL}/api/v1/openai/jobs/${job_id}/cancel")
    status=$(echo "$response" | jq -r '.status // "error"')

    if [[ "$status" == "cancelled" ]]; then
      echo "  ✓ Cancelled"
      ((cancelled++)) || true
    else
      echo "  ⚠ Status: $status"
    fi
  done <<< "$stuck_jobs"

  echo ""
  echo "Cancelled $cancelled backend job(s)."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "IMPORTANT: Clear browser state to fix stuck 'Generating...'"
echo ""
echo "Run this in browser console (F12 → Console):"
echo ""
echo "  localStorage.removeItem('ZUSTAND_STATE')"
echo "  location.reload()"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
