#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARKS="deltablue,chaos,nqueens,richards,go,nbody,fannkuch"
LEVEL="advanced"
COUNT="10"
OUT_DIR="$SCRIPT_DIR/sampled_detyped_outputs"
SEED=""
MAX_ATTEMPTS_PER_BENCH="4000"
START_SKIP_BUILD="${START_SKIP_BUILD:-1}"

usage() {
  cat <<'USAGE'
Usage: ./sample_detyped_outputs.sh [options]

Samples detyped, typechecking permutations and writes transformed sources locally.
When a benchmark has fewer than N unique non-typed passing permutations, samples
are repeated to still emit N files.

Defaults:
  benchmarks: deltablue,chaos,nqueens,richards,go,nbody,fannkuch
  level: advanced
  count: 10 samples per benchmark
  out-dir: ./sampled_detyped_outputs

Options:
  --benchmarks CSV     Benchmark list
  --level LEVEL        advanced|shallow|untyped (default: advanced)
  --count N            Samples per benchmark (default: 10)
  --out-dir PATH       Output directory
  --seed N             Base RNG seed (default: current epoch seconds)
  --max-attempts N     Max random attempts per benchmark (default: 4000)
  -h, --help           Show this help
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
    --count)
      COUNT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS_PER_BENCH="$2"
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

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/manifest.tsv"
echo -e "benchmark\tindex\tperm\tfile" >"$MANIFEST"

IFS=',' read -r -a BENCH_ARRAY <<<"$BENCHMARKS"

bench_idx=0
for bench in "${BENCH_ARRAY[@]}"; do
  bench="$(echo "$bench" | xargs)"
  if [[ -z "$bench" ]]; then
    continue
  fi
  bench_idx=$((bench_idx + 1))
  bench_seed=$((SEED + bench_idx))
  bench_dir="$OUT_DIR/$bench"
  mkdir -p "$bench_dir"
  rm -f "$bench_dir"/sample_*.py "$bench_dir"/.tmp_*.txt

  echo "[${bench}] finding ${COUNT} passing detyped permutations (seed=${bench_seed})..."

  perms_raw="$(
    START_SKIP_BUILD="$START_SKIP_BUILD" \
      bash "$SCRIPT_DIR/start.sh" \
      env \
      SAMPLE_BENCH="$bench" \
      SAMPLE_LEVEL="$LEVEL" \
      SAMPLE_COUNT="$COUNT" \
      SAMPLE_SEED="$bench_seed" \
      SAMPLE_MAX_ATTEMPTS="$MAX_ATTEMPTS_PER_BENCH" \
      /bin/bash -lc '
set -euo pipefail
PYTHONPATH=/cinder/Tools/benchmarks /cinder/python - <<'"'"'PY'"'"'
from __future__ import annotations

import os
import random
from pathlib import Path

from de_typer_boxunbox import CinderDetyperBoxUnbox

bench = os.environ["SAMPLE_BENCH"]
level = os.environ["SAMPLE_LEVEL"]
count = int(os.environ["SAMPLE_COUNT"])
seed = int(os.environ["SAMPLE_SEED"])
max_attempts = int(os.environ["SAMPLE_MAX_ATTEMPTS"])
bench_path = f"/root/static-python-perf/Benchmark/{bench}/{level}/main.py"

if not Path(bench_path).exists():
    raise SystemExit(f"missing benchmark file: {bench_path}")

d = CinderDetyperBoxUnbox(
    benchmark_file_name=bench_path,
    python="/cinder/python",
    scratch_dir="/tmp/detype_boxunbox",
    params=(),
)

typed_perm = d.get_fully_typed_perm()
d.write_permutation(typed_perm)
typed_res = d.execute_typecheck_permutation(typed_perm)
if typed_res.returncode != 0:
    raise SystemExit(f"baseline typed fails for {bench}")

rng = random.Random(seed)
seen: set[str] = set()
found: list[str] = []

# Prefer fully-detyped first if it typechecks for this benchmark.
candidates = [d.get_fully_detyped_perm()]
attempts = 0
while len(found) < count and attempts < max_attempts:
    attempts += 1
    if len(candidates) > 0:
        perm = candidates.pop(0)
    else:
        perm = d._get_prep_perm(rng.random())
    if not any(perm):
        continue
    perm_name = CinderDetyperBoxUnbox._perm_name(perm)
    if perm_name in seen:
        continue
    seen.add(perm_name)

    d.write_permutation(perm)
    res = d.execute_typecheck_permutation(perm)
    if res.returncode == 0:
        found.append(perm_name)

if len(found) == 0:
    raise SystemExit(f"found no passing non-typed permutations for {bench}")
if len(found) < count:
    found = [found[i % len(found)] for i in range(count)]

print("__PERMS_BEGIN__")
for name in found:
    print(name)
print("__PERMS_END__")
PY
'
  )"

  mapfile -t bench_perms < <(
    printf "%s\n" "$perms_raw" \
      | tr -d '\r' \
      | awk '/^__PERMS_BEGIN__$/{on=1;next}/^__PERMS_END__$/{on=0}on'
  )

  if [[ "${#bench_perms[@]}" -ne "$COUNT" ]]; then
    echo "failed to parse ${COUNT} perms for ${bench}" >&2
    exit 1
  fi

  i=0
  for perm in "${bench_perms[@]}"; do
    i=$((i + 1))
    out_tmp="$bench_dir/.tmp_${i}_${perm}.txt"
    out_file="$bench_dir/sample_$(printf '%02d' "$i")_${perm}.py"
    START_SKIP_BUILD="$START_SKIP_BUILD" \
      bash "$SCRIPT_DIR/start.sh" \
      /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py \
      "$bench" --level "$LEVEL" --show-perm "$perm" >"$out_tmp"
    tail -n +6 "$out_tmp" >"$out_file"
    rm -f "$out_tmp"
    echo -e "${bench}\t${i}\t${perm}\t${out_file}" >>"$MANIFEST"
  done

  echo "[${bench}] wrote ${COUNT} files to ${bench_dir}"
done

echo "done. manifest: ${MANIFEST}"
