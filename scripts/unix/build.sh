#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

FORCE_NPM=0

usage() {
    cat <<'USAGE'
Usage: ./run.sh build [--npm-force]

Builds frontend (web_app/dist) without starting the backend.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --npm-force) FORCE_NPM=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

need_cmd node npm

npm_install_if_needed "web_app" "$FORCE_NPM"

log "Building frontend..."
( cd web_app && rm -rf dist && npm run build )

log "Frontend built successfully (web_app/dist)"
