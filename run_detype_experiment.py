#!/usr/bin/env python3
"""
run_detype_experiment.py
========================

Helper script for running gradual detyping experiments on static-python-perf benchmarks.

Usage:
    # List available benchmarks
    python run_detype_experiment.py --list

    # Test a single benchmark (quick sanity check)
    python run_detype_experiment.py deltablue --test

    # Run full experiment on a benchmark
    python run_detype_experiment.py deltablue --samples 50

    # Show detyped source code
    python run_detype_experiment.py deltablue --show-detyped

    # Detype specific functions by index
    python run_detype_experiment.py deltablue --detype-indices 0,3,5
"""

import argparse
import sys
import os

# Add the tools directory to path
sys.path.insert(0, '/cinder/Tools/benchmarks')

from static_python_perf_detyper import StaticPythonPerfDetyper

BENCHMARK_ROOT = "/root/static-python-perf/Benchmark"
PYTHON_PATH = "/cinder/python"
SCRATCH_DIR = "/tmp/detype_experiments"

# Known benchmarks in static-python-perf
BENCHMARKS = [
    "call_method", "call_method_slots", "call_simple", "chaos",
    "deltablue", "django", "django_sample", "espionage", "evolution",
    "fannkuch", "float", "futen", "go", "http2", "meteor", "nbody",
    "nqueens", "pidigits", "pystone", "pythonflow", "richards",
    "sample_fsm", "slowsha", "spectralnorm", "stats", "take5",
    "wagtail", "zulip"
]


def get_benchmark_path(name: str, typing_level: str = "advanced") -> str:
    """Get the path to a benchmark's main.py file."""
    return f"{BENCHMARK_ROOT}/{name}/{typing_level}/main.py"


def list_benchmarks():
    """List all available benchmarks."""
    print("Available benchmarks in static-python-perf:\n")
    for name in BENCHMARKS:
        path = get_benchmark_path(name)
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
        if exists:
            # Check typing levels
            for level in ["advanced", "shallow", "untyped"]:
                level_path = get_benchmark_path(name, level)
                if os.path.exists(level_path):
                    print(f"      - {level}")


def test_benchmark(name: str, typing_level: str = "advanced"):
    """Quick test of a benchmark - run fully typed and fully detyped."""
    path = get_benchmark_path(name, typing_level)
    if not os.path.exists(path):
        print(f"Error: Benchmark not found at {path}")
        return False

    print(f"Testing benchmark: {name} ({typing_level})")
    print(f"Path: {path}\n")

    detyper = StaticPythonPerfDetyper(
        benchmark_path=path,
        python=PYTHON_PATH,
        scratch_dir=SCRATCH_DIR
    )

    print(f"Found {detyper.fun_count()} functions:\n")
    for i, (class_name, func_name) in enumerate(detyper.fun_names):
        if class_name:
            print(f"  {i:3}: {class_name}.{func_name}")
        elif func_name:
            print(f"  {i:3}: {func_name}")
        else:
            print(f"  {i:3}: <top-level>")

    # Test fully typed
    print("\n" + "="*60)
    print("TEST 1: Fully typed (no detyping)")
    print("="*60)
    typed_perm = detyper.get_fully_typed_perm()
    result = detyper.run_permutation(typed_perm)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"Output: {result.stdout[:200]}")
    if result.returncode != 0:
        print(f"Error: {result.stderr[:500]}")
        return False

    # Test fully detyped
    print("\n" + "="*60)
    print("TEST 2: Fully detyped (with box-and-convert)")
    print("="*60)
    detyped_perm = detyper.get_fully_detyped_perm()
    result = detyper.run_permutation(detyped_perm)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"Output: {result.stdout[:200]}")
    if result.returncode != 0:
        print(f"Error: {result.stderr[:500]}")
        return False

    print("\n" + "="*60)
    print("SUCCESS: Both typed and detyped versions work!")
    print("="*60)
    return True


def show_detyped(name: str, typing_level: str = "advanced", max_lines: int = 200):
    """Show the detyped source code."""
    path = get_benchmark_path(name, typing_level)
    if not os.path.exists(path):
        print(f"Error: Benchmark not found at {path}")
        return

    detyper = StaticPythonPerfDetyper(
        benchmark_path=path,
        python=PYTHON_PATH,
        scratch_dir=SCRATCH_DIR
    )

    detyped_perm = detyper.get_fully_detyped_perm()
    source = detyper.detype_by_permutation(detyped_perm)

    print(f"Detyped source for {name} ({typing_level}):")
    print("="*60 + "\n")

    lines = source.split('\n')
    for i, line in enumerate(lines[:max_lines]):
        print(f"{i+1:4}: {line}")

    if len(lines) > max_lines:
        print(f"\n... ({len(lines) - max_lines} more lines)")


def run_experiment(name: str, typing_level: str = "advanced",
                   samples: int = 50, granularity: int = 1):
    """Run a full detyping experiment."""
    path = get_benchmark_path(name, typing_level)
    if not os.path.exists(path):
        print(f"Error: Benchmark not found at {path}")
        return

    print(f"Running experiment on: {name} ({typing_level})")
    print(f"Samples per level: {samples}")
    print(f"Granularity: {granularity}\n")

    detyper = StaticPythonPerfDetyper(
        benchmark_path=path,
        python=PYTHON_PATH,
        scratch_dir=SCRATCH_DIR
    )

    results_file = detyper.find_permutation_errors(samples=samples, granularity=granularity)
    print(f"\nResults saved to: {results_file}")


def detype_specific(name: str, indices: list[int], typing_level: str = "advanced"):
    """Detype only specific functions by index."""
    path = get_benchmark_path(name, typing_level)
    if not os.path.exists(path):
        print(f"Error: Benchmark not found at {path}")
        return

    detyper = StaticPythonPerfDetyper(
        benchmark_path=path,
        python=PYTHON_PATH,
        scratch_dir=SCRATCH_DIR
    )

    # Create permutation with only specified indices detyped
    perm = tuple(i in indices for i in range(detyper.fun_count()))

    print(f"Detyping functions at indices: {indices}")
    print(f"Functions being detyped:")
    for i, (class_name, func_name) in enumerate(detyper.fun_names):
        if i in indices:
            if class_name:
                print(f"  {i}: {class_name}.{func_name}")
            elif func_name:
                print(f"  {i}: {func_name}")
            else:
                print(f"  {i}: <top-level>")

    print("\nRunning...")
    result = detyper.run_permutation(perm)

    print(f"\nReturn code: {result.returncode}")
    if result.stdout:
        print(f"Output:\n{result.stdout[:500]}")
    if result.returncode != 0:
        print(f"Error:\n{result.stderr}")

    # Show the generated file path
    print(f"\nGenerated file: {detyper.perm_file_name(perm)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run gradual detyping experiments on static-python-perf benchmarks"
    )
    parser.add_argument("benchmark", nargs="?", help="Benchmark name (e.g., deltablue)")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--test", action="store_true", help="Quick test (typed + fully detyped)")
    parser.add_argument("--show-detyped", action="store_true", help="Show detyped source code")
    parser.add_argument("--samples", type=int, default=50, help="Samples per proportion level")
    parser.add_argument("--granularity", type=int, default=1, help="Step size for proportion")
    parser.add_argument("--typing-level", default="advanced",
                        choices=["advanced", "shallow", "untyped"],
                        help="Typing level to use")
    parser.add_argument("--detype-indices", type=str,
                        help="Comma-separated indices of functions to detype")
    parser.add_argument("--max-lines", type=int, default=200,
                        help="Max lines to show with --show-detyped")

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    if not args.benchmark:
        parser.print_help()
        return

    if args.test:
        test_benchmark(args.benchmark, args.typing_level)
    elif args.show_detyped:
        show_detyped(args.benchmark, args.typing_level, args.max_lines)
    elif args.detype_indices:
        indices = [int(x.strip()) for x in args.detype_indices.split(",")]
        detype_specific(args.benchmark, indices, args.typing_level)
    else:
        run_experiment(args.benchmark, args.typing_level, args.samples, args.granularity)


if __name__ == "__main__":
    main()
