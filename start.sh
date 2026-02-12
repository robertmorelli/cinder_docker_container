#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-my-python-x86}"
PLATFORM="${PLATFORM:-linux/amd64}"

if [[ "${START_SKIP_BUILD:-0}" != "1" ]]; then
  git submodule update --init --recursive
  docker build --platform "$PLATFORM" -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

if [[ "$#" -eq 0 ]]; then
  set -- /bin/bash
fi

docker run \
  --privileged \
  --platform "$PLATFORM" \
  -it \
  --pid=host \
  -v "$SCRIPT_DIR/de_typer.py:/cinder/Tools/benchmarks/de_typer.py:ro" \
  -v "$SCRIPT_DIR/de_type2.py:/cinder/Tools/benchmarks/de_type2.py:ro" \
  -v "$SCRIPT_DIR/de_typer_boxunbox.py:/cinder/Tools/benchmarks/de_typer_boxunbox.py:ro" \
  -v "$SCRIPT_DIR/passes:/cinder/Tools/benchmarks/passes:ro" \
  -v "$SCRIPT_DIR/metadata:/cinder/Tools/benchmarks/metadata:ro" \
  -v "$SCRIPT_DIR/tests:/root/tests:ro" \
  -v "$SCRIPT_DIR/docker_bash_profile:/root/.bash_profile:ro" \
  -v "$SCRIPT_DIR/static-python-perf:/root/static-python-perf:ro" \
  "$IMAGE_NAME" \
  "$@"
