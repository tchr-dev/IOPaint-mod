#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

sub="${1:-}"
shift || true

usage() {
    cat <<'USAGE'
Usage:
  ./run.sh docker build --tag <version> [--cpu] [--gpu] [--repo <image>] [--platform <plat>] [--push|--load]

Examples:
  ./run.sh docker build --tag v1.2.3 --cpu --repo yourorg/iopaint
  ./run.sh docker build --tag v1.2.3 --cpu --gpu --push
USAGE
}

build() {
    local tag=""
    local repo="${IMAGE_REPO:-cwq1913/lama-cleaner}"
    local platform="${DOCKER_PLATFORM:-linux/amd64}"
    local do_cpu=0
    local do_gpu=0
    local mode=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tag) tag="${2:-}"; shift 2 ;;
            --repo) repo="${2:-}"; shift 2 ;;
            --platform) platform="${2:-}"; shift 2 ;;
            --cpu) do_cpu=1; shift ;;
            --gpu) do_gpu=1; shift ;;
            --push) mode="--push"; shift ;;
            --load) mode="--load"; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown argument: $1" ;;
        esac
    done

    [[ -n "$tag" ]] || die "--tag is required"
    if [[ "$do_cpu" == "0" && "$do_gpu" == "0" ]]; then
        do_cpu=1
        do_gpu=1
    fi

    need_cmd docker

    local desc="Image inpainting tool powered by SOTA AI Model"
    local repo_url="${GIT_REPO_URL:-https://github.com/Sanster/lama-cleaner}"

    if [[ "$do_cpu" == "1" ]]; then
        log "Building CPU image: ${repo}:cpu-${tag}"
        docker buildx build \
            --platform "$platform" \
            --file ./docker/CPUDockerfile \
            --label org.opencontainers.image.title=lama-cleaner \
            --label org.opencontainers.image.description="$desc" \
            --label org.opencontainers.image.url="$repo_url" \
            --label org.opencontainers.image.source="$repo_url" \
            --label org.opencontainers.image.version="$tag" \
            --build-arg version="$tag" \
            --tag "${repo}:cpu-${tag}" \
            ${mode:-} \
            .
    fi

    if [[ "$do_gpu" == "1" ]]; then
        log "Building GPU image: ${repo}:gpu-${tag}"
        docker buildx build \
            --platform "$platform" \
            --file ./docker/GPUDockerfile \
            --label org.opencontainers.image.title=lama-cleaner \
            --label org.opencontainers.image.description="$desc" \
            --label org.opencontainers.image.url="$repo_url" \
            --label org.opencontainers.image.source="$repo_url" \
            --label org.opencontainers.image.version="$tag" \
            --build-arg version="$tag" \
            --tag "${repo}:gpu-${tag}" \
            ${mode:-} \
            .
    fi
}

case "$sub" in
    build)
        build "$@"
        ;;
    -h|--help|help|"")
        usage
        exit 0
        ;;
    *)
        die "Unknown docker subcommand: $sub"
        ;;
esac
