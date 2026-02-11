#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import sys
from ast import dump as ast_dump, parse, unparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from de_typer_boxunbox import CinderDetyperBoxUnbox, PASS_NAMES  # noqa: E402


EXAMPLE_DIR = Path(__file__).resolve().parent
SCRATCH_DIR = "/tmp/detype_boxunbox_examples"

# Non-pass examples need a driver configuration for invoking the transform pipeline.
SPECIAL_CASE_ENABLED_PASSES: dict[str, tuple[str, ...]] = {
    # Must run through detype_function to trigger cleanup_inline_function.
    "inline_cleanup": ("param_cast_all",),
    # No pass enabled: run only global cleanup behavior.
    "wrapper_cleanup": (),
}


def case_enabled_passes(case_name: str) -> tuple[str, ...]:
    if case_name in PASS_NAMES:
        return (case_name,)
    if case_name in SPECIAL_CASE_ENABLED_PASSES:
        return SPECIAL_CASE_ENABLED_PASSES[case_name]
    raise AssertionError(f"unknown case name: {case_name}")


def ast_key(src: str) -> str:
    return ast_dump(parse(src, type_comments=True), include_attributes=False)


def canonical_source(src: str) -> str:
    return unparse(parse(src, type_comments=True)) + "\n"


def parse_plan_keys(plan_file: Path, detyper: CinderDetyperBoxUnbox) -> tuple[tuple[str | None, str | None], ...]:
    known_keys = set(detyper.fun_names)
    selected: list[tuple[str | None, str | None]] = []
    for raw_line in plan_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue
        if line == "__TOP_LEVEL__":
            key = (None, None)
        elif "." in line:
            class_name, function_name = line.split(".", 1)
            key = (class_name, function_name)
        else:
            key = (None, line)
        assert key in known_keys, f"unknown plan key {key} in {plan_file}"
        selected.append(key)
    assert len(selected) > 0, f"empty plan in {plan_file}"
    return tuple(selected)


def permutation_for_case(from_file: Path, detyper: CinderDetyperBoxUnbox):
    case_name = from_file.name[: -len(".from.py")]
    plan_file = from_file.with_name(f"{case_name}.plan.txt")
    if not plan_file.exists():
        return detyper.get_fully_detyped_perm()
    selected_keys = set(parse_plan_keys(plan_file, detyper))
    return tuple(q_name in selected_keys for q_name in detyper.fun_names)


def run_case(from_file: Path) -> tuple[bool, str]:
    assert from_file.name.endswith(".from.py"), f"bad input file name: {from_file}"
    case_name = from_file.name[: -len(".from.py")]
    to_file = from_file.with_name(f"{case_name}.to.py")
    assert to_file.exists(), f"missing expected file: {to_file}"

    enabled_passes = case_enabled_passes(case_name)

    detyper = CinderDetyperBoxUnbox(
        benchmark_file_name=str(from_file),
        python=sys.executable,
        scratch_dir=SCRATCH_DIR,
        params=(),
    )
    perm = permutation_for_case(from_file, detyper)
    transformed = detyper._detype_by_permutation(perm, enabled_pass_names=enabled_passes)
    expected = to_file.read_text(encoding="utf-8")

    if ast_key(transformed) == ast_key(expected):
        return True, ""

    transformed_norm = canonical_source(transformed).splitlines(keepends=True)
    expected_norm = canonical_source(expected).splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            expected_norm,
            transformed_norm,
            fromfile=str(to_file),
            tofile=f"{from_file} (transformed)",
        )
    )
    return False, diff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cases",
        nargs="*",
        help="optional case names (without .from.py suffix); default runs all",
    )
    args = parser.parse_args()

    all_from_files = sorted(EXAMPLE_DIR.glob("*.from.py"))
    if len(args.cases) == 0:
        selected = all_from_files
    else:
        requested = set(args.cases)
        selected = []
        for from_file in all_from_files:
            case_name = from_file.name[: -len(".from.py")]
            if case_name in requested:
                selected.append(from_file)

        found_names = set(f.name[: -len(".from.py")] for f in selected)
        missing = sorted(requested - found_names)
        assert len(missing) == 0, f"unknown case names: {missing}"

    failures: list[tuple[str, str]] = []
    for from_file in selected:
        case_name = from_file.name[: -len(".from.py")]
        ok, detail = run_case(from_file)
        state = "ok" if ok else "fail"
        print(f"{case_name}: {state}")
        if not ok:
            failures.append((case_name, detail))

    if len(failures) == 0:
        print(f"all examples passed ({len(selected)} cases)")
        return 0

    print(f"{len(failures)} example(s) failed")
    for case_name, detail in failures:
        print(f"\n--- {case_name} diff ---")
        print(detail)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
