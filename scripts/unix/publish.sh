#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

SKIP_FRONTEND=0
CLEAN=0

usage() {
    cat <<'USAGE'
Usage: ./run.sh publish [--skip-frontend] [--clean]

Builds frontend assets into iopaint/web_app and builds python sdist/wheel.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-frontend) SKIP_FRONTEND=1; shift ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

need_cmd python3

if [[ "$SKIP_FRONTEND" != "1" ]]; then
    need_cmd node npm

    if [[ "$CLEAN" == "1" ]]; then
        rm -rf web_app/dist iopaint/web_app
    fi

    npm_install_if_needed "web_app" 0

    log "Building frontend..."
    ( cd web_app && rm -rf dist && npm run build )

    log "Copying frontend assets into iopaint/web_app/..."
    rm -rf iopaint/web_app
    cp -r web_app/dist iopaint/web_app
fi

log "Building python distributions..."
rm -rf dist
python3 setup.py sdist bdist_wheel
log "Done. Artifacts in ./dist"
