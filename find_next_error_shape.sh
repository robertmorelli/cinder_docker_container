#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARKS="deltablue,chaos,nqueens,richards,go,nbody,fannkuch"
LEVEL="advanced"
ROOT="/root/static-python-perf/Benchmark"
SAMPLES="80"
SEED=""
SKIP_BASELINE_FAILING="1"
CHECK_FULLY_DETYPED_FIRST="1"

usage() {
  cat <<'USAGE'
Usage: ./find_next_error_shape.sh [options]

Runs deterministic fuzzing in Docker and stops on the first transformed failure.
Prints:
  - benchmark / sample / permutation id
  - error tail (+ stderr tail)
  - transformed file path
  - snippet around the failing line in transformed output

Options:
  --benchmarks CSV            Benchmark names (default:
                             deltablue,chaos,nqueens,richards,go,nbody,fannkuch)
  --level LEVEL              advanced|shallow|untyped (default: advanced)
  --root PATH                Benchmark root inside container (default: /root/static-python-perf/Benchmark)
  --samples N                Random samples per benchmark (default: 80)
  --seed N                   RNG seed (default: current epoch seconds)
  --skip-baseline-failing 0|1
                             Skip benches whose fully-typed transform already fails
                             static typecheck (default: 1)
  --check-fully-detyped-first 0|1
                             Test fully detyped permutation before random samples (default: 1)
  -h, --help                 Show this help

Environment:
  START_SKIP_BUILD=1         Skip Docker image build in start.sh (default: 1)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    --level)
      LEVEL="$2"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --skip-baseline-failing)
      SKIP_BASELINE_FAILING="$2"
      shift 2
      ;;
    --check-fully-detyped-first)
      CHECK_FULLY_DETYPED_FIRST="$2"
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

if [[ -z "$SEED" ]]; then
  SEED="$(date +%s)"
fi

printf -v q_benchmarks '%q' "$BENCHMARKS"
printf -v q_level '%q' "$LEVEL"
printf -v q_root '%q' "$ROOT"
printf -v q_samples '%q' "$SAMPLES"
printf -v q_seed '%q' "$SEED"
printf -v q_skip '%q' "$SKIP_BASELINE_FAILING"
printf -v q_check_full '%q' "$CHECK_FULLY_DETYPED_FIRST"

START_SKIP_BUILD="${START_SKIP_BUILD:-1}" \
  bash "$SCRIPT_DIR/start.sh" /bin/bash -lc "
set -euo pipefail
export FIND_BENCHMARKS=$q_benchmarks
export FIND_LEVEL=$q_level
export FIND_ROOT=$q_root
export FIND_SAMPLES=$q_samples
export FIND_SEED=$q_seed
export FIND_SKIP_BASELINE_FAILING=$q_skip
export FIND_CHECK_FULLY_DETYPED_FIRST=$q_check_full
PYTHONPATH=/cinder/Tools/benchmarks /cinder/python - <<'PY'
from __future__ import annotations

import os
import random
import re
import sys
import traceback
from pathlib import Path

from de_typer_boxunbox import CinderDetyperBoxUnbox


def last_line(stderr: str) -> str:
    lines = stderr.strip().splitlines()
    return lines[-1] if len(lines) > 0 else '<no stderr>'


def print_transformed_snippet(file_name: str, stderr: str, window: int = 6) -> None:
    p = Path(file_name)
    if not p.exists():
        print(f'snippet: missing file {file_name}')
        return

    line_no: int | None = None
    for m in re.finditer(r'File \"([^\"]+)\", line (\\d+)', stderr):
        fname = m.group(1)
        num = int(m.group(2))
        if fname == file_name:
            line_no = num

    if line_no is None:
        m = re.search(r'line (\\d+)', stderr)
        if m is not None:
            line_no = int(m.group(1))

    if line_no is None:
        print('snippet: could not parse failing line')
        return

    lines = p.read_text(encoding='utf-8').splitlines()
    start = max(1, line_no - window)
    end = min(len(lines), line_no + window)
    print(f'--- transformed snippet ({file_name}:{line_no}) ---')
    for i in range(start, end + 1):
        print(f'{i:5d}: {lines[i - 1]}')


def report_typecheck_failure(
    bench_name: str, sample_index: int, perm_name: str, out_file: str, stderr: str
) -> None:
    print('=== ERROR SHAPE ===')
    print(f'benchmark: {bench_name}')
    print(f'sample: {sample_index}')
    print(f'perm: {perm_name}')
    print(f'error_tail: {last_line(stderr)}')
    print(f'transformed_file: {out_file}')
    print_transformed_snippet(out_file, stderr)
    tail = stderr.strip().splitlines()[-25:]
    print('--- stderr tail (last 25 lines) ---')
    for ln in tail:
        print(ln)


def report_transform_exception(
    bench_name: str, sample_index: int, perm_name: str, out_file: str, exc: BaseException
) -> None:
    print('=== ERROR SHAPE (transform exception) ===')
    print(f'benchmark: {bench_name}')
    print(f'sample: {sample_index}')
    print(f'perm: {perm_name}')
    print(f'error_tail: {type(exc).__name__}: {exc}')
    print(f'transformed_file: {out_file}')
    print('--- traceback ---')
    print(traceback.format_exc().strip())


def main() -> int:
    benchmarks = [b.strip() for b in os.environ['FIND_BENCHMARKS'].split(',') if b.strip()]
    level = os.environ['FIND_LEVEL']
    root = os.environ['FIND_ROOT']
    samples = int(os.environ['FIND_SAMPLES'])
    seed = int(os.environ['FIND_SEED'])
    skip_baseline_failing = os.environ['FIND_SKIP_BASELINE_FAILING'] == '1'
    check_fully_detyped_first = os.environ['FIND_CHECK_FULLY_DETYPED_FIRST'] == '1'

    rng = random.Random(seed)

    print(f'seed: {seed}')
    print(f'benchmarks: {benchmarks}')
    print(f'level: {level}')
    print(f'samples_per_benchmark: {samples}')
    print(f'skip_baseline_failing: {skip_baseline_failing}')
    print(f'check_fully_detyped_first: {check_fully_detyped_first}')

    for bench in benchmarks:
        bench_path = f'{root}/{bench}/{level}/main.py'
        if not Path(bench_path).exists():
            print(f'--- skip {bench}: missing benchmark file {bench_path}')
            continue

        d = CinderDetyperBoxUnbox(
            benchmark_file_name=bench_path,
            python='/cinder/python',
            scratch_dir='/tmp/detype_boxunbox',
            params=(),
        )
        print(
            f'\\n--- benchmark {bench} (functions={d.fun_count()} plan_bits={d.plan_bit_count()}) ---'
        )

        # Baseline: fully typed transform should typecheck if we want transformed-only failures.
        typed_perm = d.get_fully_typed_perm()
        typed_name = CinderDetyperBoxUnbox._perm_name(typed_perm)
        try:
            d.write_permutation(typed_perm)
        except Exception as exc:
            report_transform_exception(bench, 0, typed_name, d.perm_file_name(typed_perm), exc)
            return 0

        typed_res = d.execute_typecheck_permutation(typed_perm)
        if typed_res.returncode != 0:
            print(
                f'baseline typed fails for {bench}: {last_line(typed_res.stderr)}'
            )
            if skip_baseline_failing:
                continue
            report_typecheck_failure(bench, 0, typed_name, d.perm_file_name(typed_perm), typed_res.stderr)
            return 0

        perm_list = []
        if check_fully_detyped_first:
            perm_list.append(d.get_fully_detyped_perm())
        for _ in range(samples):
            perm_list.append(d._get_prep_perm(rng.random()))

        for i, perm in enumerate(perm_list, start=1):
            perm_name = CinderDetyperBoxUnbox._perm_name(perm)
            out_file = d.perm_file_name(perm)
            try:
                d.write_permutation(perm)
            except Exception as exc:
                report_transform_exception(bench, i, perm_name, out_file, exc)
                return 0

            res = d.execute_typecheck_permutation(perm)
            if res.returncode != 0:
                report_typecheck_failure(bench, i, perm_name, out_file, res.stderr)
                return 0

        print(f'no failure in {len(perm_list)} tested perms for {bench}')

    print('\\nNo transformed failure found in requested benchmark/sample window.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
PY
"
