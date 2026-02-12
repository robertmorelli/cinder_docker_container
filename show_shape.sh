#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CASE_NAME=""
MANIFEST_IN_CONTAINER="/root/tests/regression_check/manifest.json"
PYTHON_IN_CONTAINER="/cinder/python"
SCRATCH_IN_CONTAINER="/tmp/detype_shape"
SHOW_STDERR_LINES="40"

usage() {
  cat <<'USAGE'
Usage: ./show_shape.sh [case_name] [options]

Shows one regression case shape end-to-end:
  1) BEFORE source (case file)
  2) AFTER source (transformed output)
  3) Static typecheck result tail

Arguments:
  case_name                    Case name from regression manifest
                               (default: first case in manifest)

Options:
  --manifest PATH              Manifest path inside container
                               (default: /root/tests/regression_check/manifest.json)
  --python PATH                Python path inside container (default: /cinder/python)
  --scratch PATH               Scratch dir inside container (default: /tmp/detype_shape)
  --stderr-lines N             Number of stderr tail lines to show (default: 40)
  -h, --help                   Show this help

Environment:
  START_SKIP_BUILD=1           Skip Docker image build in start.sh (default: 1)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_IN_CONTAINER="$2"
      shift 2
      ;;
    --python)
      PYTHON_IN_CONTAINER="$2"
      shift 2
      ;;
    --scratch)
      SCRATCH_IN_CONTAINER="$2"
      shift 2
      ;;
    --stderr-lines)
      SHOW_STDERR_LINES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$CASE_NAME" ]]; then
        CASE_NAME="$1"
        shift 1
      else
        echo "Unknown arg: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

printf -v q_case '%q' "$CASE_NAME"
printf -v q_manifest '%q' "$MANIFEST_IN_CONTAINER"
printf -v q_python '%q' "$PYTHON_IN_CONTAINER"
printf -v q_scratch '%q' "$SCRATCH_IN_CONTAINER"
printf -v q_stderr_lines '%q' "$SHOW_STDERR_LINES"

inner_cmd="$(cat <<EOF
set -euo pipefail
export SHAPE_CASE_NAME=$q_case
export SHAPE_MANIFEST=$q_manifest
export SHAPE_PYTHON=$q_python
export SHAPE_SCRATCH=$q_scratch
export SHAPE_STDERR_LINES=$q_stderr_lines
PYTHONPATH=/cinder/Tools/benchmarks /cinder/python - <<'PY'
from __future__ import annotations

import json
import traceback
from pathlib import Path

from de_typer_boxunbox import CinderDetyperBoxUnbox


def to_key(token: str):
    if token == "__TOP_LEVEL__":
        return (None, None)
    if "." in token:
        a, b = token.split(".", 1)
        return (a, b)
    return (None, token)


def line_numbered(text: str) -> str:
    out: list[str] = []
    lines = text.splitlines()
    for i, ln in enumerate(lines, start=1):
        out.append(f"{i:5d}: {ln}")
    return "\\n".join(out)


def main() -> int:
    manifest_path = Path(__import__("os").environ["SHAPE_MANIFEST"])
    python_bin = __import__("os").environ["SHAPE_PYTHON"]
    scratch_dir = __import__("os").environ["SHAPE_SCRATCH"]
    case_name = __import__("os").environ["SHAPE_CASE_NAME"]
    stderr_lines = int(__import__("os").environ["SHAPE_STDERR_LINES"])

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "manifest must be a list"
    assert len(data) > 0, "manifest must not be empty"

    if case_name == "":
        case = data[0]
        case_name = case["name"]
    else:
        matches = [row for row in data if row["name"] == case_name]
        assert len(matches) == 1, f"case not found or ambiguous: {case_name}"
        case = matches[0]

    case_file = (manifest_path.parent / case["file"]).resolve()
    enabled_passes = tuple(case.get("enabled_passes", []))
    plan = list(case.get("plan", []))

    detyper = CinderDetyperBoxUnbox(
        benchmark_file_name=str(case_file),
        python=python_bin,
        scratch_dir=scratch_dir,
        params=(),
    )

    if len(plan) == 0:
        perm = detyper.get_fully_detyped_perm()
    else:
        selected = set(to_key(item) for item in plan)
        perm = tuple(q in selected for q in detyper.fun_names)
    perm_name = CinderDetyperBoxUnbox._perm_name(perm)

    before = case_file.read_text(encoding="utf-8")

    print("=== CASE META ===")
    print(f"name: {case_name}")
    print(f"case_file: {case_file}")
    print(f"enabled_passes: {list(enabled_passes)}")
    print(f"plan: {plan}")
    print(f"perm: {perm_name}")

    print("\\n=== BEFORE ===")
    print(line_numbered(before))

    try:
        detyper.write_permutation(perm, enabled_pass_names=enabled_passes)
    except Exception as exc:
        print("\\n=== TRANSFORM EXCEPTION ===")
        print(f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc().strip())
        return 0

    out_file = Path(detyper.perm_file_name(perm))
    after = out_file.read_text(encoding="utf-8")

    print("\\n=== AFTER ===")
    print(f"transformed_file: {out_file}")
    print(line_numbered(after))

    result = detyper.execute_typecheck_permutation(perm)
    stderr = result.stderr.strip().splitlines()
    print("\\n=== TYPECHECK ===")
    print(f"return_code: {result.returncode}")
    if len(stderr) == 0:
        print("stderr_tail: <empty>")
    else:
        print(f"stderr_tail: {stderr[-1]}")
        print(f"--- stderr last {stderr_lines} lines ---")
        for ln in stderr[-stderr_lines:]:
            print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
EOF
)"

START_SKIP_BUILD="${START_SKIP_BUILD:-1}" \
  bash "$SCRIPT_DIR/start.sh" /bin/bash -lc "$inner_cmd"
