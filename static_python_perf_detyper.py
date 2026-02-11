"""
static_python_perf_detyper.py
=============================

Detyper adapted for static-python-perf single-file benchmarks.
Implements the box-and-convert pattern for safe gradual detyping.

Unlike CinderDetyper which expects runner + lib files,
this handles the single main.py structure of static-python-perf.

Box-and-Convert Pattern:
    BEFORE (typed):
        def foo(x: int64, y: double) -> int64:
            return x + int64(y)

        foo(my_x, my_y)

    AFTER (detyped with safe boundary):
        def foo(_x, _y):
            x: int64 = int64(_x)
            y: double = double(_y)
            return x + int64(y)

        foo(box(my_x), box(my_y))
"""

from ast import (
    parse, unparse, FunctionDef, AsyncFunctionDef, ClassDef, AnnAssign, Assign,
    expr, Name, iter_child_nodes, AST, fix_missing_locations, JoinedStr,
    NodeTransformer, Import, ImportFrom, Constant, Call, Attribute, Load, Store,
    Module, alias, arg, arguments, iter_fields, NodeVisitor
)
from functools import cache, reduce
from itertools import chain
from random import sample
from subprocess import run, CompletedProcess
from time import perf_counter
from multiprocessing import Pool, cpu_count, Value, Lock
from datetime import datetime
from json import dump
from tempfile import mkdtemp
from atexit import register
from os import path, makedirs, unlink
from shutil import rmtree
from typing import Tuple, Iterator, Optional
from copy import deepcopy

try:
    import black
    def format_code(src: str) -> str:
        return black.format_str(src, mode=black.Mode(line_length=100))
except ImportError:
    def format_code(src: str) -> str:
        return src

# Type aliases
Permutation = Tuple[bool, ...]
GuideKey = tuple[str | None, str | None]
GuideType = dict[GuideKey, bool]
QNameType = set[GuideKey]

TOP_LEVEL = (None, None)

# Primitive types that need box/convert treatment
PRIMITIVE_TYPES = frozenset({
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'double', 'cbool', 'char', 'single'
})


def extract_type_name(annotation: expr) -> str | None:
    """Extract simple type name from annotation."""
    if isinstance(annotation, Name):
        return annotation.id
    return None


def is_primitive_type(annotation: expr) -> bool:
    """Check if annotation is a Static Python primitive type."""
    type_name = extract_type_name(annotation)
    return type_name in PRIMITIVE_TYPES if type_name else False


def get_primitive_params(fn: FunctionDef | AsyncFunctionDef) -> list[tuple[str, str]]:
    """
    Extract (param_name, type_name) for parameters with primitive types.

    Args:
        fn: Function definition node

    Returns:
        List of (param_name, type_name) tuples for primitive-typed params
    """
    primitives = []
    args = fn.args

    for a in chain(args.posonlyargs, args.args, args.kwonlyargs):
        if a.annotation:
            type_name = extract_type_name(a.annotation)
            if type_name and type_name in PRIMITIVE_TYPES:
                primitives.append((a.arg, type_name))

    return primitives


def detype_with_conversions(fn: FunctionDef | AsyncFunctionDef) -> dict[str, str]:
    """
    Transform a function to use box-and-convert pattern.

    Renames primitive-typed parameters (x -> _x), clears annotations,
    and inserts conversion statements at the start of the function body.

    BEFORE:
        def foo(x: int64, y: double) -> int64:
            return x + int64(y)

    AFTER:
        def foo(_x, _y):
            x: int64 = int64(_x)
            y: double = double(_y)
            return x + int64(y)

    Args:
        fn: Function definition node (modified in place)

    Returns:
        Dict mapping original param names to their types (for call site transformation)
    """
    # 1. Get primitive params before clearing annotations
    primitives = get_primitive_params(fn)
    param_types = {name: type_name for name, type_name in primitives}

    # 2. Rename params and clear annotations
    param_map = {}  # old_name -> new_name
    args = fn.args

    for a in chain(args.posonlyargs, args.args, args.kwonlyargs):
        if a.annotation:
            type_name = extract_type_name(a.annotation)
            if type_name and type_name in PRIMITIVE_TYPES:
                old_name = a.arg
                new_name = f"_{old_name}"
                param_map[old_name] = new_name
                a.arg = new_name
        a.annotation = None

    # Clear vararg/kwarg annotations too
    if args.vararg and args.vararg.annotation:
        args.vararg.annotation = None
    if args.kwarg and args.kwarg.annotation:
        args.kwarg.annotation = None

    # 3. Handle default arguments - convert primitive defaults to boxed values
    # For now, we leave defaults as-is since they should be compatible

    # 4. Build conversion statements
    conversion_stmts = []
    for old_name, type_name in primitives:
        new_name = param_map[old_name]
        # x: int64 = int64(_x)
        stmt = AnnAssign(
            target=Name(id=old_name, ctx=Store()),
            annotation=Name(id=type_name, ctx=Load()),
            value=Call(
                func=Name(id=type_name, ctx=Load()),
                args=[Name(id=new_name, ctx=Load())],
                keywords=[]
            ),
            simple=1
        )
        conversion_stmts.append(stmt)

    # 5. Prepend conversion statements to function body
    if conversion_stmts:
        fn.body = conversion_stmts + fn.body

    # 6. Clear return type
    fn.returns = None
    if hasattr(fn, 'type_comment'):
        fn.type_comment = None

    return param_types


def detype_function_simple(fn: FunctionDef | AsyncFunctionDef) -> None:
    """
    Simple detyping - just remove annotations without box/convert.
    Used for functions that don't have primitive params or when
    we want basic detyping.
    """
    args = fn.args

    for a in chain(args.posonlyargs, args.args, args.kwonlyargs):
        a.annotation = None

    if args.vararg and args.vararg.annotation:
        args.vararg.annotation = None
    if args.kwarg and args.kwarg.annotation:
        args.kwarg.annotation = None

    fn.returns = None
    if hasattr(fn, 'type_comment'):
        fn.type_comment = None


def detype_body_annotations(node: AST) -> None:
    """
    Remove type annotations from assignments within a node's body.
    Converts AnnAssign to Assign and clears type_comment on Assign.
    """
    for child in iter_child_nodes(node):
        if isinstance(child, AnnAssign):
            # Convert annotated assignment to regular assignment
            child.__class__ = Assign
            child.targets = [child.target]
            if getattr(child, "value", None) is None:
                child.value = Constant(value=None)
            if hasattr(child, "annotation"):
                del child.annotation
            if hasattr(child, 'type_comment'):
                child.type_comment = None
        elif isinstance(child, Assign):
            if hasattr(child, 'type_comment'):
                child.type_comment = None
        elif isinstance(child, (FunctionDef, AsyncFunctionDef, ClassDef)):
            # Recurse into nested structures
            detype_body_annotations(child)


class BoxCallSites(NodeTransformer):
    """
    Wrap arguments in box() when calling detyped functions.

    BEFORE:
        result = foo(my_x, my_y)

    AFTER:
        result = foo(box(my_x), box(my_y))

    This transformer handles:
    - Simple function calls
    - Method calls (skips 'self')
    - Nested calls
    - Already-boxed expressions (no double-boxing)
    """

    def __init__(self, detyped_functions: set[str], primitive_params: dict[str, dict[str, str]]):
        """
        Args:
            detyped_functions: Names of functions that were detyped
            primitive_params: {func_name: {param_name: type_name}}
        """
        self.detyped_functions = detyped_functions
        self.primitive_params = primitive_params

    def visit_Call(self, node: Call) -> Call:
        # Process children first (handle nested calls)
        self.generic_visit(node)

        func_name = self._get_func_name(node.func)
        if func_name not in self.detyped_functions:
            return node

        # Get which params need boxing for this function
        params_info = self.primitive_params.get(func_name, {})
        if not params_info:
            # No primitive params, no boxing needed
            return node

        # Determine if this is a method call (has self as first arg)
        is_method = isinstance(node.func, Attribute)

        # Box positional args
        new_args = []
        for i, arg_node in enumerate(node.args):
            # Skip self for method calls
            if self._should_skip_boxing(arg_node, i, is_method):
                new_args.append(arg_node)
            elif self._already_boxed(arg_node):
                new_args.append(arg_node)
            else:
                new_args.append(self._wrap_in_box(arg_node))
        node.args = new_args

        # Box keyword args
        for kw in node.keywords:
            if kw.arg and kw.arg in params_info:
                if not self._already_boxed(kw.value):
                    kw.value = self._wrap_in_box(kw.value)

        return node

    def _get_func_name(self, func: expr) -> str | None:
        """Extract function name from call expression."""
        if isinstance(func, Name):
            return func.id
        if isinstance(func, Attribute):
            return func.attr
        return None

    def _should_skip_boxing(self, arg_node: expr, index: int, is_method: bool) -> bool:
        """Determine if an argument should NOT be boxed."""
        # Never box 'self' (first arg of method call when it's a Name 'self')
        if is_method and index == 0:
            if isinstance(arg_node, Name) and arg_node.id == "self":
                return True
        return False

    def _already_boxed(self, node: expr) -> bool:
        """Check if expression is already wrapped in box()."""
        return (isinstance(node, Call) and
                isinstance(node.func, Name) and
                node.func.id == "box")

    def _wrap_in_box(self, node: expr) -> Call:
        """Wrap an expression in box()."""
        return Call(
            func=Name(id="box", ctx=Load()),
            args=[node],
            keywords=[]
        )


class FunctionCallCollector(NodeVisitor):
    """Collect all function/method names that are called in the AST."""

    def __init__(self):
        self.called_functions: set[str] = set()

    def visit_Call(self, node: Call):
        if isinstance(node.func, Name):
            self.called_functions.add(node.func.id)
        elif isinstance(node.func, Attribute):
            self.called_functions.add(node.func.attr)
        self.generic_visit(node)


def ensure_box_import(tree: Module) -> None:
    """
    Add 'box' to __static__ imports if not present.

    static-python-perf benchmarks typically already have:
        from __static__ import CheckedList, box, cast, cbool, clen, int64, inline

    But we need to verify box is included.
    """
    for node in tree.body:
        if isinstance(node, ImportFrom) and node.module == "__static__":
            # Check if box is already imported
            if any(a.name == "box" for a in node.names):
                return
            # Add box to existing import
            node.names.append(alias(name="box", asname=None))
            return

    # No __static__ import found - add one
    # Insert after other imports
    insert_pos = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, (Import, ImportFrom)):
            insert_pos = i + 1
        elif not isinstance(node, expr):  # Skip docstrings
            break

    new_import = ImportFrom(
        module="__static__",
        names=[alias(name="box", asname=None)],
        level=0
    )
    tree.body.insert(insert_pos, new_import)


def ensure_primitive_imports(tree: Module, needed_types: set[str]) -> None:
    """
    Ensure all needed primitive types are imported from __static__.
    """
    if not needed_types:
        return

    for node in tree.body:
        if isinstance(node, ImportFrom) and node.module == "__static__":
            existing = {a.name for a in node.names}
            for type_name in needed_types:
                if type_name not in existing:
                    node.names.append(alias(name=type_name, asname=None))
            return


class StaticPythonPerfDetyper:
    """
    Detyper adapted for static-python-perf single-file benchmarks.

    Implements the box-and-convert pattern for safe gradual detyping:
    1. Track which params had primitive types
    2. Rename params (x -> _x) and clear annotations
    3. Insert conversion statements (x: int64 = int64(_x))
    4. Transform call sites to wrap args in box()
    5. Ensure box is imported
    """

    def __init__(self,
                 benchmark_path: str,
                 python: str,
                 scratch_dir: str,
                 params: tuple[str, ...] = ()):
        """
        Args:
            benchmark_path: Path to the benchmark main.py file
            python: Path to the Cinder python binary
            scratch_dir: Directory for generated files
            params: Additional params to pass to benchmark
        """
        self.benchmark_path = benchmark_path
        self.python = python
        self.scratch_dir = scratch_dir
        self.params = params

        # Parse and enumerate functions
        self.class_antrs, self.fun_names = self._enumerate_funs()

    def fun_count(self) -> int:
        """Number of functions (including top-level) that can be detyped."""
        return len(self.fun_names)

    @cache
    def _read_source(self) -> str:
        """Read benchmark source file."""
        with open(self.benchmark_path, encoding="utf-8") as f:
            return f.read()

    @cache
    def _read_ast(self) -> Module:
        """Parse benchmark source into AST."""
        return parse(self._read_source(), type_comments=True)

    def _enumerate_funs(self) -> tuple[dict, tuple]:
        """
        Enumerate all functions in the benchmark file.

        Returns:
            (class_ancestors_graph, ordered_function_qualified_names)
        """
        ordered_q_names = [TOP_LEVEL]
        q_fun_names_set: QNameType = {TOP_LEVEL}
        antr_graph: dict[str | None, frozenset[str]] = {None: frozenset()}

        def antr_classes(bases: list[expr]) -> frozenset[str]:
            """Get all ancestor classes from base expressions."""
            antr_names = tuple(
                b.id for b in bases
                if isinstance(b, Name) and b.id in antr_graph
            )
            anter_set = frozenset(antr_names)
            antr_name_sets = tuple(antr_graph[name] for name in antr_names)
            return reduce(lambda a, b: a.union(b), antr_name_sets, anter_set)

        def update_antr_graph(class_node: ClassDef):
            """Update ancestor graph with a new class."""
            class_name = class_node.name
            if class_name not in antr_graph:
                antr_graph[class_name] = antr_classes(class_node.bases)

        def get_antr_fun_names(fun_name: str, antr_name: str | None) -> tuple:
            """Get all ancestor versions of a function."""
            if antr_name not in antr_graph:
                return ()
            anters = antr_graph[antr_name]
            return tuple((anter, fun_name) for anter in anters)

        def fun_exists(antr_name: str | None, fun_name: str) -> bool:
            return (antr_name, fun_name) in q_fun_names_set

        def is_new_function(fun_name: str, antr_name: str | None) -> bool:
            """Check if this is a new function (not an override)."""
            if fun_exists(antr_name, fun_name):
                return False
            antr_funs = get_antr_fun_names(fun_name, antr_name)
            return not any(fun_exists(*qn) for qn in antr_funs)

        def count_gen(node: AST, antr_name: str | None = None, in_function: bool = False):
            """Recursively count and register functions."""
            for child in iter_child_nodes(node):
                is_fun = isinstance(child, (FunctionDef, AsyncFunctionDef))
                is_class = isinstance(child, ClassDef)

                if is_fun and not in_function:
                    if is_new_function(child.name, antr_name):
                        key = (antr_name, child.name)
                        q_fun_names_set.add(key)
                        ordered_q_names.append(key)
                    # Recurse into function (but mark we're inside)
                    yield from count_gen(child, antr_name=antr_name, in_function=True)
                elif is_class and not in_function:
                    update_antr_graph(child)
                    yield from count_gen(child, antr_name=child.name, in_function=False)

        # Run enumeration
        list(count_gen(self._read_ast()))

        return antr_graph, tuple(ordered_q_names)

    def _get_qualified_name(self, fn: FunctionDef | AsyncFunctionDef,
                            class_name: str | None = None) -> GuideKey:
        """Get qualified name for a function."""
        return (class_name, fn.name)

    def _walk_functions(self, tree: AST, class_name: str | None = None
                        ) -> Iterator[tuple[FunctionDef | AsyncFunctionDef, str | None]]:
        """
        Yield all (function_node, class_name) pairs in the tree.
        class_name is None for top-level functions.
        """
        for child in iter_child_nodes(tree):
            if isinstance(child, (FunctionDef, AsyncFunctionDef)):
                yield (child, class_name)
                # Don't recurse into nested functions for now
            elif isinstance(child, ClassDef):
                yield from self._walk_functions(child, class_name=child.name)

    @cache
    def _get_fun_q_names(self, fun_name: str, antr_name: str | None = None) -> tuple:
        """Get all qualified names for a function including ancestors."""
        if antr_name not in self.class_antrs:
            return ((antr_name, fun_name),)
        anters = self.class_antrs[antr_name]
        anter_fun_names = tuple(
            qn for qn in ((a, fun_name) for a in anters)
            if qn in self.fun_names
        )
        return ((antr_name, fun_name),) + anter_fun_names

    def _get_canonical_q_name(self, fun_name: str, antr_name: str | None = None) -> GuideKey:
        """Get the canonical qualified name (handles inheritance)."""
        q_names = self._get_fun_q_names(fun_name, antr_name)
        # If there's an ancestor version, use that
        if len(q_names) > 1:
            return q_names[1]
        return q_names[0]

    def detype_by_permutation(self, perm: Permutation) -> str:
        """
        Apply box-and-convert pattern based on permutation.

        For each function in the permutation that should be detyped:
        1. Track which params had primitive types
        2. Rename params (x -> _x) and clear annotations
        3. Insert conversion statements (x: int64 = int64(_x))
        4. Transform call sites to wrap args in box()
        5. Ensure box is imported

        Args:
            perm: Boolean tuple indicating which functions to detype

        Returns:
            Transformed source code as string
        """
        # Create fresh copy of AST
        tree = parse(self._read_source(), type_comments=True)
        guide = dict(zip(self.fun_names, perm))

        # Phase 1: Collect info about functions to detype
        detyped_functions: set[str] = set()
        primitive_params: dict[str, dict[str, str]] = {}
        needed_types: set[str] = set()

        for fn, class_name in self._walk_functions(tree):
            q_name = self._get_canonical_q_name(fn.name, class_name)
            if guide.get(q_name, False):
                primitives = get_primitive_params(fn)
                if primitives:
                    param_dict = {name: type_name for name, type_name in primitives}
                    primitive_params[fn.name] = param_dict
                    needed_types.update(param_dict.values())
                detyped_functions.add(fn.name)

        # Phase 2: Apply transformations to functions
        for fn, class_name in self._walk_functions(tree):
            q_name = self._get_canonical_q_name(fn.name, class_name)
            if guide.get(q_name, False):
                # Use box-and-convert if function has primitive params
                if fn.name in primitive_params:
                    detype_with_conversions(fn)
                else:
                    detype_function_simple(fn)
                # Also detype body annotations
                detype_body_annotations(fn)

        # Phase 3: Transform call sites (only if we have detyped functions with primitives)
        if primitive_params:
            transformer = BoxCallSites(detyped_functions, primitive_params)
            tree = transformer.visit(tree)

        # Phase 4: Ensure box is imported (if we did any boxing)
        if primitive_params:
            ensure_box_import(tree)

        # Phase 5: Ensure primitive types are imported (for conversions)
        ensure_primitive_imports(tree, needed_types)

        # Phase 6: Handle top-level if specified
        if guide.get(TOP_LEVEL, False):
            detype_body_annotations(tree)

        fix_missing_locations(tree)
        return format_code(unparse(tree))

    @staticmethod
    @cache
    def _perm_name(perm: Permutation) -> str:
        """Convert permutation to hex string name."""
        return hex(int("".join(str(int(b)) for b in perm), 2))

    @cache
    def _perm_from_name(self, perm_str: str) -> Permutation:
        """Convert hex string back to permutation tuple."""
        n = int(perm_str, 16)
        bits = bin(n)[2:].zfill(self.fun_count())
        return tuple(c == "1" for c in bits)

    def _ensure_scratch_dir(self):
        """Ensure scratch directory exists."""
        if not path.exists(self.scratch_dir):
            makedirs(self.scratch_dir, exist_ok=True)

    def perm_file_name(self, perm: Permutation) -> str:
        """Get output file path for a permutation."""
        base = path.basename(self.benchmark_path).replace(".py", "")
        perm_name = self._perm_name(perm)
        return f"{self.scratch_dir}/{base}_{perm_name}.py"

    def write_permutation(self, perm: Permutation) -> str:
        """
        Write detyped version of benchmark to scratch dir.

        Returns:
            Path to the written file
        """
        self._ensure_scratch_dir()
        source = self.detype_by_permutation(perm)
        file_path = self.perm_file_name(perm)
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(source)
        return file_path

    def execute_permutation(self, perm: Permutation) -> CompletedProcess:
        """Execute a permutation and return the result."""
        file_path = self.perm_file_name(perm)
        cmd = " ".join((
            self.python,
            "-X jit",
            "-X jit-enable-jit-list-wildcards",
            "-X jit-shadow-frame",
            file_path,
            *self.params
        ))
        return run(cmd, capture_output=True, text=True, shell=True)

    def run_permutation(self, perm: Permutation) -> CompletedProcess:
        """Write and execute a permutation."""
        self.write_permutation(perm)
        return self.execute_permutation(perm)

    def _get_prep_perm(self, proportion: float) -> Permutation:
        """Generate a random permutation with given proportion of detyped functions."""
        typed_count = int(self.fun_count() * proportion)
        untyped_count = self.fun_count() - typed_count
        return tuple(sample(
            [False] * typed_count + [True] * untyped_count,
            k=self.fun_count()
        ))

    def get_fully_typed_perm(self) -> Permutation:
        """Get permutation where nothing is detyped."""
        return tuple([False] * self.fun_count())

    def get_fully_detyped_perm(self) -> Permutation:
        """Get permutation where everything is detyped."""
        return tuple([True] * self.fun_count())

    def find_permutation_errors(self, samples: int = 50, granularity: int = 1):
        """
        Run experiment to find which permutations cause errors.

        Args:
            samples: Number of random samples per proportion level
            granularity: Step size for proportion levels
        """
        self._ensure_scratch_dir()

        def xp_gen():
            # Always test fully typed and fully detyped
            yield self.get_fully_typed_perm()
            yield self.get_fully_detyped_perm()
            # Test intermediate proportions
            for i in range(1, self.fun_count(), granularity):
                proportion = i / self.fun_count()
                for _ in range(samples):
                    yield self._get_prep_perm(proportion)

        perms = list(xp_gen())
        results = {}
        successes = []
        failures = []

        print(f"Running {len(perms)} permutations...")

        for i, perm in enumerate(perms):
            perm_name = self._perm_name(perm)
            result = self.run_permutation(perm)
            results[perm_name] = {
                "returncode": result.returncode,
                "stdout": result.stdout[:1000] if result.stdout else "",
                "stderr": result.stderr[:1000] if result.stderr else ""
            }

            if result.returncode == 0:
                successes.append(perm_name)
            else:
                failures.append(perm_name)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(perms)} complete ({len(successes)} successes, {len(failures)} failures)")

        # Save results
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = path.basename(self.benchmark_path).replace(".py", "")
        results_file = f"{self.scratch_dir}/results_{base}_{stamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            dump({
                "benchmark": self.benchmark_path,
                "total": len(perms),
                "success_count": len(successes),
                "failure_count": len(failures),
                "successes": successes,
                "failures": failures,
                "details": results
            }, f, indent=2)

        print(f"\nResults: {len(successes)} successes, {len(failures)} failures")
        print(f"Saved to: {results_file}")

        return results_file


# Worker functions for multiprocessing
class PermutationWorker:
    _detyper = None
    _counter = None
    _lock = None

    @staticmethod
    def init_worker(detyper, counter, lock):
        PermutationWorker._detyper = detyper
        PermutationWorker._counter = counter
        PermutationWorker._lock = lock

    @staticmethod
    def run_perm(perm: Permutation) -> tuple[str, dict]:
        result = PermutationWorker._detyper.run_permutation(perm)
        perm_name = StaticPythonPerfDetyper._perm_name(perm)

        with PermutationWorker._lock:
            PermutationWorker._counter.value += 1

        return perm_name, {
            "returncode": result.returncode,
            "stdout": result.stdout[:500] if result.stdout else "",
            "stderr": result.stderr[:500] if result.stderr else ""
        }


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python static_python_perf_detyper.py <benchmark_path> [python_path] [scratch_dir]")
        print("\nExample:")
        print("  python static_python_perf_detyper.py /root/static-python-perf/Benchmark/deltablue/advanced/main.py")
        sys.exit(1)

    benchmark_path = sys.argv[1]
    python_path = sys.argv[2] if len(sys.argv) > 2 else "/cinder/python"
    scratch_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/detype_experiments"

    print(f"Benchmark: {benchmark_path}")
    print(f"Python: {python_path}")
    print(f"Scratch dir: {scratch_dir}")

    detyper = StaticPythonPerfDetyper(
        benchmark_path=benchmark_path,
        python=python_path,
        scratch_dir=scratch_dir
    )

    print(f"\nFound {detyper.fun_count()} functions to potentially detype:")
    for i, name in enumerate(detyper.fun_names):
        class_name, func_name = name
        if class_name:
            print(f"  {i}: {class_name}.{func_name}")
        elif func_name:
            print(f"  {i}: {func_name}")
        else:
            print(f"  {i}: <top-level>")

    # Run a quick test
    print("\n--- Testing fully typed (no detyping) ---")
    typed_perm = detyper.get_fully_typed_perm()
    result = detyper.run_permutation(typed_perm)
    print(f"Return code: {result.returncode}")
    if result.returncode != 0:
        print(f"Stderr: {result.stderr[:500]}")

    print("\n--- Testing fully detyped ---")
    detyped_perm = detyper.get_fully_detyped_perm()
    result = detyper.run_permutation(detyped_perm)
    print(f"Return code: {result.returncode}")
    if result.returncode != 0:
        print(f"Stderr: {result.stderr[:500]}")

    # Show example of detyped code
    print("\n--- Sample detyped code (first 100 lines) ---")
    detyped_source = detyper.detype_by_permutation(detyped_perm)
    for i, line in enumerate(detyped_source.split('\n')[:100]):
        print(f"{i+1:4}: {line}")


if __name__ == "__main__":
    main()
