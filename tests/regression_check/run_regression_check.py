#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from de_typer_boxunbox import CinderDetyperBoxUnbox, PASS_NAMES  # noqa: E402


def parse_plan(plan_items: list[str], detyper: CinderDetyperBoxUnbox):
    known = set(detyper.fun_names)
    selected: list[tuple[str | None, str | None]] = []
    for token in plan_items:
        if token == "__TOP_LEVEL__":
            key = (None, None)
        elif "." in token:
            antr, fun = token.split(".", 1)
            key = (antr, fun)
        else:
            key = (None, token)
        assert key in known, f"unknown plan key {key}"
        selected.append(key)
    return tuple(selected)


def perm_from_plan(detyper: CinderDetyperBoxUnbox, plan_items: list[str]):
    if len(plan_items) == 0:
        return detyper.get_fully_detyped_perm()
    selected = set(parse_plan(plan_items, detyper))
    return tuple(q in selected for q in detyper.fun_names)


def last_error(stderr: str):
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    return lines[-1] if lines else "<no stderr>"


def run_case(case: dict[str, Any], root: Path, python_bin: str, scratch: str):
    name = case["name"]
    file_path = (root / case["file"]).resolve()
    enabled_passes = tuple(case.get("enabled_passes", []))
    for pass_name in enabled_passes:
        assert pass_name in PASS_NAMES, f"unknown pass {pass_name} in {name}"
    expect = case.get("expect", "typecheck_ok")
    err_contains = case.get("error_contains")
    plan_items = list(case.get("plan", []))

    detyper = CinderDetyperBoxUnbox(
        benchmark_file_name=str(file_path),
        python=python_bin,
        scratch_dir=scratch,
        params=(),
    )
    perm = perm_from_plan(detyper, plan_items)

    try:
        detyper.write_permutation(perm, enabled_pass_names=enabled_passes)
    except Exception as exc:
        msg = str(exc)
        if expect != "transform_fail":
            return False, f"unexpected transform failure: {msg}"
        if err_contains is not None and err_contains not in msg:
            return False, f"transform failed with wrong message: {msg}"
        return True, ""

    if expect == "transform_fail":
        return False, "expected transform failure but transform succeeded"

    result = detyper.execute_typecheck_permutation(perm)
    if expect == "typecheck_ok":
        if result.returncode == 0:
            return True, ""
        return False, f"typecheck failed: {last_error(result.stderr)}"

    if expect == "typecheck_fail":
        if result.returncode != 0:
            tail = last_error(result.stderr)
            if err_contains is not None and err_contains not in tail:
                return False, f"typecheck failed with wrong message: {tail}"
            return True, ""
        return False, "expected typecheck failure but got success"

    return False, f"unsupported expect mode: {expect}"


def print_pass_summary(case_results: list[dict[str, Any]]):
    print("\n=== pass summary (typecheck_ok cases) ===")
    pass_to_rows: dict[str, list[dict[str, Any]]] = dict((pass_name, []) for pass_name in PASS_NAMES)
    cleanup_rows: list[dict[str, Any]] = []

    for row in case_results:
        if row["expect"] != "typecheck_ok":
            continue
        enabled_passes = row["enabled_passes"]
        if len(enabled_passes) == 0:
            cleanup_rows.append(row)
            continue
        for pass_name in enabled_passes:
            pass_to_rows[pass_name].append(row)

    for pass_name in PASS_NAMES:
        rows = pass_to_rows[pass_name]
        if len(rows) == 0:
            print(f"{pass_name}: no_cases")
            continue
        fail_rows = [row for row in rows if not row["ok"]]
        state = "ok" if len(fail_rows) == 0 else "fail"
        print(f"{pass_name}: {state} ({len(rows) - len(fail_rows)}/{len(rows)})")
        if len(fail_rows) > 0:
            for row in fail_rows:
                print(f"  - {row['name']}: {row['detail']}")

    if len(cleanup_rows) > 0:
        fail_rows = [row for row in cleanup_rows if not row["ok"]]
        state = "ok" if len(fail_rows) == 0 else "fail"
        print(f"cleanup_only: {state} ({len(cleanup_rows) - len(fail_rows)}/{len(cleanup_rows)})")
        if len(fail_rows) > 0:
            for row in fail_rows:
                print(f"  - {row['name']}: {row['detail']}")

    guarded = [row for row in case_results if row["expect"] != "typecheck_ok"]
    if len(guarded) > 0:
        guard_fails = [row for row in guarded if not row["ok"]]
        state = "ok" if len(guard_fails) == 0 else "fail"
        print(f"guarded_cases: {state} ({len(guarded) - len(guard_fails)}/{len(guarded)})")
        if len(guard_fails) > 0:
            for row in guard_fails:
                print(f"  - {row['name']}: {row['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/regression_check/manifest.json")
    parser.add_argument("--python", default="/cinder/python")
    parser.add_argument("--scratch", default="/tmp/detype_regression")
    parser.add_argument("--summary-by-pass", action="store_true")
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()

    manifest_path = (REPO_ROOT / args.manifest).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "manifest must be a list"

    selected = []
    requested = set(args.cases)
    for case in data:
        name = case["name"]
        if len(requested) == 0 or name in requested:
            selected.append(case)

    if len(requested) > 0:
        selected_names = set(c["name"] for c in selected)
        missing = sorted(requested - selected_names)
        assert len(missing) == 0, f"unknown case names: {missing}"

    case_results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    root = manifest_path.parent
    for case in selected:
        name = case["name"]
        enabled_passes = list(case.get("enabled_passes", []))
        expect = case.get("expect", "typecheck_ok")
        ok, detail = run_case(case, root, args.python, args.scratch)
        print(f"{name}: {'ok' if ok else 'fail'}")
        case_results.append(
            {
                "name": name,
                "ok": ok,
                "detail": detail,
                "enabled_passes": enabled_passes,
                "expect": expect,
            }
        )
        if not ok:
            failures.append((name, detail))

    if args.summary_by_pass:
        print_pass_summary(case_results)

    if len(failures) == 0:
        print(f"all regression cases passed ({len(selected)} cases)")
        return 0

    print(f"{len(failures)} regression case(s) failed")
    for name, detail in failures:
        print(f"- {name}: {detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
