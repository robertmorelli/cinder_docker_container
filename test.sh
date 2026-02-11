#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARK="deltablue"
LEVEL="advanced"
SAMPLES="6"
GRANULARITY="5"
PYTHON_IN_CONTAINER="/cinder/python"
START_SKIP_BUILD="${START_SKIP_BUILD:-1}"
PARAMS=()

usage() {
  cat <<'USAGE'
Usage: ./test.sh [options]

Runs boxunbox detyper vs de_type2 detyper in Docker and reports validity/error stats.

Options:
  --benchmark NAME|PATH   Benchmark name (default: deltablue) or path to main.py
  --level LEVEL           advanced|shallow|untyped (default: advanced)
  --samples N             Samples per granularity point (default: 6)
  --granularity N         Function step size (default: 5)
  --python PATH           Python inside container (default: /cinder/python)
  --param VALUE           Benchmark runtime arg (repeatable)
  -h, --help              Show this help

Environment:
  START_SKIP_BUILD=1      Skip Docker image build in start.sh (default: 1)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --level)
      LEVEL="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --granularity)
      GRANULARITY="$2"
      shift 2
      ;;
    --python)
      PYTHON_IN_CONTAINER="$2"
      shift 2
      ;;
    --param)
      PARAMS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run_one() {
  local label="$1"
  local script_name="$2"

  local run_cmd=(
    "$PYTHON_IN_CONTAINER"
    "/cinder/Tools/benchmarks/${script_name}"
    "$BENCHMARK"
    "--level" "$LEVEL"
    "--samples" "$SAMPLES"
    "--granularity" "$GRANULARITY"
    "--python" "$PYTHON_IN_CONTAINER"
  )
  local p
  for p in "${PARAMS[@]}"; do
    run_cmd+=("--param" "$p")
  done

  local quoted_run_cmd
  printf -v quoted_run_cmd '%q ' "${run_cmd[@]}"

  echo "== ${label} (${script_name}) =="
  local out
  out="$(
    START_SKIP_BUILD="$START_SKIP_BUILD" bash "$SCRIPT_DIR/start.sh" /bin/bash -lc "
set -euo pipefail
${quoted_run_cmd}>/tmp/${label}_run.log 2>&1
tr '\\r' '\\n' </tmp/${label}_run.log | tail -n 6
results_file=\$(tr '\\r' '\\n' </tmp/${label}_run.log | sed -n 's/.*results in //p' | tail -n 1)
${PYTHON_IN_CONTAINER} - \"\$results_file\" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as f:
    payload = json.load(f)

success = payload.get('success count', 0)
failure = payload.get('failure count', 0)
print(f'SUMMARY success={success} failure={failure} total={success + failure}')

for name, count in sorted(payload.get('failure stats', {}).items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
    print(f'FAIL {count} {name}')
PY
"
  )"

  printf '%s\n' "$out"

  local summary_line
  summary_line="$(printf '%s\n' "$out" | sed -n 's/^SUMMARY /SUMMARY /p' | tail -n 1)"
  if [[ -z "$summary_line" ]]; then
    echo "Failed to parse summary for ${label}" >&2
    exit 1
  fi

  local success failure total
  success="$(printf '%s\n' "$summary_line" | sed -E 's/.*success=([0-9]+).*/\1/')"
  failure="$(printf '%s\n' "$summary_line" | sed -E 's/.*failure=([0-9]+).*/\1/')"
  total="$(printf '%s\n' "$summary_line" | sed -E 's/.*total=([0-9]+).*/\1/')"

  printf -v "${label}_SUCCESS" '%s' "$success"
  printf -v "${label}_FAILURE" '%s' "$failure"
  printf -v "${label}_TOTAL" '%s' "$total"
}

pct() {
  local success="$1"
  local total="$2"
  awk -v s="$success" -v t="$total" 'BEGIN { if (t == 0) { printf "0.00" } else { printf "%.2f", (100.0 * s) / t } }'
}

run_one BOXUNBOX de_typer_boxunbox.py
run_one DETYPE2 de_type2.py

BOXUNBOX_RATE="$(pct "$BOXUNBOX_SUCCESS" "$BOXUNBOX_TOTAL")"
DETYPE2_RATE="$(pct "$DETYPE2_SUCCESS" "$DETYPE2_TOTAL")"

echo
echo "== Comparison =="
echo "benchmark=${BENCHMARK} level=${LEVEL} samples=${SAMPLES} granularity=${GRANULARITY}"
echo "BOXUNBOX valid: ${BOXUNBOX_SUCCESS}/${BOXUNBOX_TOTAL} (${BOXUNBOX_RATE}%)"
echo "DETYPE2 valid: ${DETYPE2_SUCCESS}/${DETYPE2_TOTAL} (${DETYPE2_RATE}%)"

if (( BOXUNBOX_SUCCESS > DETYPE2_SUCCESS )); then
  echo "Result: BOXUNBOX produced more valid programs in this run."
elif (( BOXUNBOX_SUCCESS < DETYPE2_SUCCESS )); then
  echo "Result: DETYPE2 produced more valid programs in this run."
else
  echo "Result: tie on valid program count in this run."
fi
