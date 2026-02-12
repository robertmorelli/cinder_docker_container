#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

inner_cmd=(
  PYTHONPATH=/cinder/Tools/benchmarks:/root
  /cinder/python
  /root/tests/regression_check/run_regression_check.py
  --summary-by-pass
)

if [[ "$#" -gt 0 ]]; then
  inner_cmd+=("$@")
fi

printf -v inner_cmd_quoted '%q ' "${inner_cmd[@]}"

START_SKIP_BUILD="${START_SKIP_BUILD:-1}" \
  bash "$SCRIPT_DIR/start.sh" /bin/bash -lc "$inner_cmd_quoted"
