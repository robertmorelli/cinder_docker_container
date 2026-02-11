# Adapting de_typer.py for Safe Gradual Detyping

## Problem Statement

When `de_typer.py` removes type annotations from functions, errors occur because:

1. **Static → Static calls skip type guards** (by design, for performance)
2. A typed caller passes primitives (`int64`, `double`) expecting the callee to handle them
3. The detyped callee receives raw primitive values without proper conversion
4. Runtime crashes or incorrect behavior results

## Proposed Solution: Box-and-Convert Pattern

Transform both the function signature AND call sites:

```python
# BEFORE (typed)
def foo(x: int64, y: double) -> int64:
    return x + int64(y)

foo(my_x, my_y)

# AFTER (detyped with safe boundary)
def foo(_x, _y):
    x: int64 = int64(_x)
    y: double = double(_y)
    return x + int64(y)

foo(box(my_x), box(my_y))
```

### Why This Works

Per `type_checking_guide.md`:

| Mechanism | Behavior |
|-----------|----------|
| `box(value)` | Converts primitive to Python object (safe for dynamic code) |
| `int64(value)` | Explicit conversion that **always validates** at runtime |

By boxing at call site and converting inside the function:
- The value safely crosses the typed/untyped boundary
- The explicit `int64()` / `double()` / etc. conversion provides runtime type checking
- We avoid the "guards skipped" problem entirely

## Required Changes to de_typer.py

### 1. Track Parameter Types Before Removal

Currently `detype_fun_params` just clears annotations. We need to **remember** what they were:

```python
def detype_fun_params_with_conversion(fn: FunctionDef) -> list[tuple[str, str]]:
    """
    Returns list of (param_name, type_name) for params that had primitive types.
    Renames params to _param_name and clears annotations.
    """
    conversions = []
    args = fn.args
    args_to_process = chain(args.posonlyargs, args.args, args.kwonlyargs)

    for a in args_to_process:
        if a.annotation and is_primitive_type(a.annotation):
            type_name = get_type_name(a.annotation)
            original_name = a.arg
            conversions.append((original_name, type_name))
            a.arg = f"_{original_name}"  # Rename parameter
        a.annotation = None

    # Handle *args and **kwargs if they have primitive annotations
    if args.vararg and args.vararg.annotation:
        args.vararg.annotation = None
    if args.kwarg and args.kwarg.annotation:
        args.kwarg.annotation = None

    return conversions
```

### 2. Identify Primitive Types

Add a helper to identify which types need box/convert treatment:

```python
PRIMITIVE_TYPES = frozenset({
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'double', 'cbool', 'char', 'single'
})

def is_primitive_type(annotation: expr) -> bool:
    """Check if annotation is a Static Python primitive type."""
    if isinstance(annotation, Name):
        return annotation.id in PRIMITIVE_TYPES
    return False

def get_type_name(annotation: expr) -> str:
    """Extract type name from annotation."""
    if isinstance(annotation, Name):
        return annotation.id
    raise ValueError(f"Unsupported annotation type: {type(annotation)}")
```

### 3. Insert Conversion Statements at Function Start

After detyping params, insert conversion assignments:

```python
def insert_param_conversions(fn: FunctionDef, conversions: list[tuple[str, str]]):
    """
    Insert 'x: type = type(_x)' statements at the start of function body.
    """
    conversion_stmts = []
    for param_name, type_name in conversions:
        # Create: x: type = type(_x)
        stmt = AnnAssign(
            target=Name(id=param_name, ctx=Store()),
            annotation=Name(id=type_name, ctx=Load()),
            value=Call(
                func=Name(id=type_name, ctx=Load()),
                args=[Name(id=f"_{param_name}", ctx=Load())],
                keywords=[]
            ),
            simple=1
        )
        conversion_stmts.append(stmt)

    fn.body = conversion_stmts + fn.body
```

### 4. Transform Call Sites

This is the harder part. Need to find all calls to detyped functions and wrap arguments in `box()`:

```python
class CallSiteTransformer(NodeTransformer):
    def __init__(self, detyped_functions: set[str]):
        self.detyped_functions = detyped_functions

    def visit_Call(self, node: Call) -> Call:
        self.generic_visit(node)  # Transform nested calls first

        func_name = self._get_func_name(node.func)
        if func_name in self.detyped_functions:
            # Wrap each argument in box()
            node.args = [self._wrap_in_box(arg) for arg in node.args]
            # Handle keyword arguments too
            for kw in node.keywords:
                kw.value = self._wrap_in_box(kw.value)

        return node

    def _get_func_name(self, func: expr) -> str | None:
        if isinstance(func, Name):
            return func.id
        if isinstance(func, Attribute):
            return func.attr  # method name
        return None

    def _wrap_in_box(self, expr: expr) -> Call:
        # Don't double-box
        if isinstance(expr, Call) and isinstance(expr.func, Name) and expr.func.id == 'box':
            return expr
        return Call(
            func=Name(id='box', ctx=Load()),
            args=[expr],
            keywords=[]
        )
```

### 5. Handle Method Calls and `self`

The existing `do_not_box` method handles `self`. Extend it:

```python
def should_skip_boxing(self, arg: expr, param_index: int, is_method: bool) -> bool:
    """Determine if an argument should NOT be boxed."""
    # Never box 'self' (first arg of method)
    if is_method and param_index == 0 and isinstance(arg, Name) and arg.id == 'self':
        return True
    # Already boxed
    if isinstance(arg, Call) and isinstance(arg.func, Name) and arg.func.id == 'box':
        return True
    return False
```

### 6. Handle Return Types

Return types also need consideration. If a detyped function returns a primitive:

```python
# BEFORE
def foo(x: int64) -> int64:
    return x * 2

# AFTER - caller expects int64, function returns boxed
def foo(_x) -> int64:  # Keep return annotation for caller's benefit
    x: int64 = int64(_x)
    return x * 2  # x is int64, so return is fine
```

Since we convert `_x` to `x: int64` inside the function, operations on `x` produce `int64` results, so the return type should be preserved. **No change needed for returns.**

### 7. Ensure `box` is Imported

Add import for `box` if not present:

```python
def ensure_box_import(tree: Module):
    """Add 'from __static__ import box' if not present."""
    has_box_import = False
    for node in tree.body:
        if isinstance(node, ImportFrom) and node.module == '__static__':
            if any(alias.name == 'box' for alias in node.names):
                has_box_import = True
                break

    if not has_box_import:
        # Find existing __static__ import and add box, or create new import
        for node in tree.body:
            if isinstance(node, ImportFrom) and node.module == '__static__':
                node.names.append(alias(name='box', asname=None))
                return

        # No __static__ import found, add one
        new_import = ImportFrom(
            module='__static__',
            names=[alias(name='box', asname=None)],
            level=0
        )
        # Insert after any existing imports
        insert_pos = 0
        for i, node in enumerate(tree.body):
            if isinstance(node, (Import, ImportFrom)):
                insert_pos = i + 1
        tree.body.insert(insert_pos, new_import)
```

## Modified Detyping Flow

```python
def _detype_funs_safe(self, tree: AST, guide: GuideType) -> str:
    """
    Detype functions using the box-and-convert pattern.
    """
    # Track which functions are being detyped
    detyped_functions: set[str] = set()

    def detype_walker(node: AST, antr_name: str | None = None):
        for child_node in iter_child_nodes(node):
            if isinstance(child_node, FunctionDef):
                if fun_should_be_detyped(child_node.name, antr_name):
                    # 1. Collect param types and rename params
                    conversions = detype_fun_params_with_conversion(child_node)
                    # 2. Insert conversion statements
                    insert_param_conversions(child_node, conversions)
                    # 3. Clear return type (optional - see discussion above)
                    child_node.returns = None
                    # 4. Track this function
                    detyped_functions.add(child_node.name)
            elif isinstance(child_node, ClassDef):
                detype_walker(child_node, antr_name=child_node.name)

    # Phase 1: Detype function signatures
    detype_walker(tree)

    # Phase 2: Transform call sites
    transformer = CallSiteTransformer(detyped_functions)
    tree = transformer.visit(tree)

    # Phase 3: Ensure box is imported
    ensure_box_import(tree)

    fix_missing_locations(tree)
    return split_lines(unparse(tree))
```

## Adapting for static-python-perf Benchmarks

### File Structure Differences

The current `de_typer.py` expects:
- `{name}.py` (runner)
- `{name}_lib.py` (library with typed code)

The static-python-perf benchmarks may have different structures. Check if they follow this pattern or need adjustment in `__init__`:

```python
# Current assumption (de_typer.py:84)
self.lib_file_name = f"{CinderDetyper.file_trunc(runner_file_name)}_lib.py"

# May need to make this configurable:
def __init__(self, runner_file_name: str, lib_file_name: str | None = None, ...):
    if lib_file_name is None:
        self.lib_file_name = f"{CinderDetyper.file_trunc(runner_file_name)}_lib.py"
    else:
        self.lib_file_name = lib_file_name
```

### JIT Configuration

The current JIT flags in `execute_permutation` (line 332-340):

```python
cmd = " ".join((
    self.python,
    "-X jit",
    f"-X jit-list-file=jitlist_{runner_trunc}.txt",
    "-X jit-enable-jit-list-wildcards",
    "-X jit-shadow-frame",
    "-X install-strict-loader",
    new_runner_name,
    *self.params
))
```

The static-python-perf benchmarks may need different flags. Consider making this configurable:

```python
def __init__(self, ..., jit_flags: list[str] | None = None):
    self.jit_flags = jit_flags or [
        "-X jit",
        "-X jit-enable-jit-list-wildcards",
        "-X jit-shadow-frame",
        "-X install-strict-loader",
    ]
```

### Cross-Module Calls

If the benchmarks have calls between modules (not just runner → lib), the call site transformation needs to handle both:
1. Calls within the lib module (already handled)
2. Calls from runner to lib (need to transform runner too)

Current code only detypes the lib. May need to also transform call sites in the runner:

```python
def _renamed_lib_in_runner(self, perm: Permutation):
    tree = self._read_runner_ast()
    # ... existing import renaming ...

    # NEW: Transform call sites in runner too
    detyped_functions = self._get_detyped_function_names(perm)
    transformer = CallSiteTransformer(detyped_functions)
    tree = transformer.visit(tree)

    fix_missing_locations(tree)
    return split_lines(unparse(tree))
```

## Edge Cases to Handle

### 1. Default Arguments

```python
# BEFORE
def foo(x: int64 = int64(0)) -> int64:
    return x

# AFTER
def foo(_x=0):  # Default must be boxed value
    x: int64 = int64(_x)
    return x
```

Need to transform default values too.

### 2. *args and **kwargs

```python
# BEFORE
def foo(*args: int64) -> int64:
    return sum(args)

# This is tricky - each element of args needs conversion
# AFTER
def foo(*_args):
    args = tuple(int64(a) for a in _args)
    return sum(args)
```

### 3. Nested Calls

```python
# BEFORE
foo(bar(x))  # Both foo and bar are detyped

# AFTER
foo(box(bar(box(x))))  # Need to box bar's result too
```

The `CallSiteTransformer` handles this via `generic_visit` processing children first.

### 4. Object Types vs Primitives

Only primitive types need box/convert. Object types (`str`, `list`, custom classes) don't need `box()`:

```python
def foo(x: int64, name: str):  # Only x needs box/convert
    ...

# Becomes
def foo(_x, name):  # name stays as-is
    x: int64 = int64(_x)
    ...
```

## Testing Strategy

1. **Unit test the transformation**: Parse a typed function, transform it, verify the AST structure
2. **Test with simple benchmarks first**: Pick the simplest static-python-perf benchmark
3. **Verify all permutations run**: The fully-typed and fully-untyped cases should both work
4. **Compare error rates**: The new approach should have 0 errors (vs the current approach's partial failures)

## Summary

The box-and-convert pattern should work because:
1. `box()` safely converts primitives to Python objects for crossing typed/untyped boundaries
2. Explicit type conversions (`int64()`, `double()`) always validate at runtime
3. This bypasses the "static→static guards skipped" design that causes current failures

Main implementation tasks:
1. Modify `detype_fun_params` to remember types and rename params
2. Add `insert_param_conversions` to add conversion statements
3. Implement `CallSiteTransformer` to wrap arguments in `box()`
4. Handle edge cases (defaults, *args, method calls)
5. Make file structure and JIT flags configurable for different benchmarks

---

# Part 2: Adapting for static-python-perf Benchmarks

## Benchmark Structure Analysis

The static-python-perf benchmarks have a **different structure** than what the current de_typer.py expects:

### Current de_typer.py Expects:
```
benchmark/
├── deltablue_static.py       # Runner file
├── deltablue_static_lib.py   # Library file (gets detyped)
└── jitlist_deltablue_static.txt
```

### static-python-perf Structure:
```
static-python-perf/Benchmark/
├── deltablue/
│   ├── advanced/main.py      # Fully typed
│   ├── shallow/main.py       # Shallow types
│   └── untyped/main.py       # No types
├── richards/
│   ├── advanced/main.py
│   ├── shallow/main.py
│   └── untyped/main.py
... (28 benchmarks total)
```

**Key Insight**: Each benchmark is a **single self-contained `main.py` file**, not a runner + library pair.

## Required Adaptation: `StaticPythonPerfDetyper`

Create a new class that adapts the detyping approach for single-file benchmarks:

```python
class StaticPythonPerfDetyper:
    """
    Detyper adapted for static-python-perf single-file benchmarks.

    Unlike CinderDetyper which expects runner + lib files,
    this handles the single main.py structure.
    """

    def __init__(self,
                 benchmark_path: str,  # e.g., "/root/static-python-perf/Benchmark/deltablue/advanced/main.py"
                 python: str,
                 scratch_dir: str,
                 params: tuple[str, ...] = ()):
        self.benchmark_path = benchmark_path
        self.python = python
        self.scratch_dir = scratch_dir
        self.params = params

        # Parse the file once
        self._ast_cache = None
        self.class_antrs, self.fun_names = self._enumerate_funs()

    @cache
    def _read_ast(self):
        with open(self.benchmark_path, encoding="utf-8") as f:
            return parse(f.read(), type_comments=True)

    def _enumerate_funs(self):
        """Same logic as CinderDetyper but for single file."""
        # ... enumerate all functions in the file
        pass
```

## Core Box-and-Convert Implementation

### Step 1: Track Primitive Parameter Types

```python
PRIMITIVE_TYPES = frozenset({
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64',
    'double', 'cbool', 'char', 'single'
})

def get_primitive_params(fn: FunctionDef) -> list[tuple[str, str]]:
    """
    Extract (param_name, type_name) for parameters with primitive types.
    """
    primitives = []
    args = fn.args
    for arg in chain(args.posonlyargs, args.args, args.kwonlyargs):
        if arg.annotation:
            type_name = _extract_type_name(arg.annotation)
            if type_name in PRIMITIVE_TYPES:
                primitives.append((arg.arg, type_name))
    return primitives

def _extract_type_name(annotation: expr) -> str | None:
    """Extract simple type name from annotation."""
    if isinstance(annotation, Name):
        return annotation.id
    return None
```

### Step 2: Rename Parameters and Insert Conversions

```python
def detype_with_conversions(fn: FunctionDef) -> None:
    """
    Transform a function to use box-and-convert pattern.

    BEFORE:
        def foo(x: int64, y: double) -> int64:
            return x + int64(y)

    AFTER:
        def foo(_x, _y):
            x: int64 = int64(_x)
            y: double = double(_y)
            return x + int64(y)
    """
    # 1. Get primitive params before clearing annotations
    primitives = get_primitive_params(fn)

    # 2. Rename params and clear annotations
    param_map = {}  # old_name -> new_name
    args = fn.args
    for arg in chain(args.posonlyargs, args.args, args.kwonlyargs):
        if arg.annotation:
            type_name = _extract_type_name(arg.annotation)
            if type_name in PRIMITIVE_TYPES:
                old_name = arg.arg
                new_name = f"_{old_name}"
                param_map[old_name] = new_name
                arg.arg = new_name
        arg.annotation = None

    # Clear vararg/kwarg annotations too
    if args.vararg and args.vararg.annotation:
        args.vararg.annotation = None
    if args.kwarg and args.kwarg.annotation:
        args.kwarg.annotation = None

    # 3. Build conversion statements
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

    # 4. Prepend to function body
    fn.body = conversion_stmts + fn.body

    # 5. Clear return type
    fn.returns = None
    fn.type_comment = None
```

### Step 3: Transform Call Sites

```python
class BoxCallSites(NodeTransformer):
    """
    Wrap arguments in box() when calling detyped functions.

    BEFORE:
        result = foo(my_x, my_y)

    AFTER:
        result = foo(box(my_x), box(my_y))
    """

    def __init__(self, detyped_functions: set[str], primitive_params: dict[str, list[str]]):
        """
        Args:
            detyped_functions: Names of functions that were detyped
            primitive_params: {func_name: [param_names_that_were_primitive]}
        """
        self.detyped_functions = detyped_functions
        self.primitive_params = primitive_params

    def visit_Call(self, node: Call) -> Call:
        # Process children first (handle nested calls)
        self.generic_visit(node)

        func_name = self._get_func_name(node.func)
        if func_name not in self.detyped_functions:
            return node

        # Get which params need boxing
        params_to_box = self.primitive_params.get(func_name, [])

        # Box positional args
        new_args = []
        for i, arg in enumerate(node.args):
            # Skip self for method calls
            if self._is_self(arg, i):
                new_args.append(arg)
            elif not self._already_boxed(arg):
                new_args.append(self._wrap_in_box(arg))
            else:
                new_args.append(arg)
        node.args = new_args

        # Box keyword args
        for kw in node.keywords:
            if not self._already_boxed(kw.value):
                kw.value = self._wrap_in_box(kw.value)

        return node

    def _get_func_name(self, func: expr) -> str | None:
        if isinstance(func, Name):
            return func.id
        if isinstance(func, Attribute):
            return func.attr
        return None

    def _is_self(self, arg: expr, index: int) -> bool:
        return index == 0 and isinstance(arg, Name) and arg.id == "self"

    def _already_boxed(self, expr: expr) -> bool:
        return (isinstance(expr, Call) and
                isinstance(expr.func, Name) and
                expr.func.id == "box")

    def _wrap_in_box(self, expr: expr) -> Call:
        return Call(
            func=Name(id="box", ctx=Load()),
            args=[expr],
            keywords=[]
        )
```

### Step 4: Ensure Box Import

```python
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
            if any(alias.name == "box" for alias in node.names):
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

    new_import = ImportFrom(
        module="__static__",
        names=[alias(name="box", asname=None)],
        level=0
    )
    tree.body.insert(insert_pos, new_import)
```

## Complete Detyping Flow for static-python-perf

```python
class StaticPythonPerfDetyper:
    # ... init and enumeration methods ...

    def detype_by_permutation(self, perm: Permutation) -> str:
        """
        Apply box-and-convert pattern based on permutation.

        For each function in the permutation that should be detyped:
        1. Track which params had primitive types
        2. Rename params (x -> _x) and clear annotations
        3. Insert conversion statements (x: int64 = int64(_x))
        4. Transform call sites to wrap args in box()
        5. Ensure box is imported
        """
        tree = parse(self._read_source())
        guide = dict(zip(self.fun_names, perm))

        # Phase 1: Collect info about functions to detype
        detyped_functions: set[str] = set()
        primitive_params: dict[str, list[str]] = {}

        for fn in self._walk_functions(tree):
            q_name = self._get_qualified_name(fn)
            if guide.get(q_name, False):
                # This function should be detyped
                primitives = get_primitive_params(fn)
                primitive_params[fn.name] = [name for name, _ in primitives]
                detyped_functions.add(fn.name)

        # Phase 2: Apply transformations to functions
        for fn in self._walk_functions(tree):
            q_name = self._get_qualified_name(fn)
            if guide.get(q_name, False):
                detype_with_conversions(fn)

        # Phase 3: Transform call sites
        transformer = BoxCallSites(detyped_functions, primitive_params)
        tree = transformer.visit(tree)

        # Phase 4: Ensure box is imported
        ensure_box_import(tree)

        # Phase 5: Handle top-level if specified
        if guide.get(TOP_LEVEL, False):
            self._detype_top_level(tree)

        fix_missing_locations(tree)
        return unparse(tree)

    def _walk_functions(self, tree: AST) -> Iterator[FunctionDef]:
        """Yield all FunctionDef nodes in the tree."""
        for node in ast.walk(tree):
            if isinstance(node, FunctionDef):
                yield node
```

## Example: deltablue Benchmark Transformation

### Before (typed):
```python
def chain_test(n: int64) -> None:
    planner = recreate_planner()
    prev: Variable | None = None
    # ...
    i: int64 = 0
    while i < n + 1:
        name = "v%s" % box(i)
        # ...
```

### After (detyped with box-and-convert):
```python
def chain_test(_n):
    n: int64 = int64(_n)  # Safe conversion
    planner = recreate_planner()
    prev = None  # Annotation removed
    # ...
    i: int64 = 0  # Local vars inside function body stay typed
    while i < n + 1:
        name = "v%s" % box(i)
        # ...

# At call sites:
chain_test(box(n))  # Primitive boxed before crossing boundary
```

## Running Modes for static-python-perf

### Mode 1: Compare Typing Levels
Compare the existing versions:
```bash
# Run advanced (fully typed)
python -X jit /root/static-python-perf/Benchmark/deltablue/advanced/main.py

# Run shallow
python -X jit /root/static-python-perf/Benchmark/deltablue/shallow/main.py

# Run untyped
python -X jit /root/static-python-perf/Benchmark/deltablue/untyped/main.py
```

### Mode 2: Gradual Detyping Experiment
```python
# Use the adapted detyper
detyper = StaticPythonPerfDetyper(
    benchmark_path="/root/static-python-perf/Benchmark/deltablue/advanced/main.py",
    python="/cinder/python",
    scratch_dir="/tmp/detype_experiments"
)

# Run permutation experiments
detyper.find_permutation_errors(samples=50, granularity=1)
```

### Mode 3: Find Minimum Boxing
```python
# Find which expressions need boxing for each permutation
detyper.find_permutation_is_boxable(samples=50)
```

## Implementation Checklist

1. [x] Create `StaticPythonPerfDetyper` class
   - [x] Single-file parsing (no runner/lib split)
   - [x] Function enumeration for single file
   - [x] Permutation naming/caching

2. [x] Implement box-and-convert pattern
   - [x] `get_primitive_params()` - extract primitive type info
   - [x] `detype_with_conversions()` - rename + insert conversions
   - [x] `BoxCallSites` transformer - wrap call args in box()
   - [x] `ensure_box_import()` - add box to imports

3. [x] Handle edge cases
   - [x] Default arguments: `def foo(x: int64 = 0)` → `def foo(_x=0)`
   - [x] Method calls: don't box `self`
   - [x] Already boxed expressions
   - [x] *args/**kwargs with primitive annotations
   - [x] Nested calls: `foo(bar(x))` → `foo(box(bar(box(x))))`

4. [x] Execution infrastructure
   - [x] JIT flags for static-python-perf
   - [x] Scratch directory management
   - [x] Results collection and analysis

5. [ ] Testing
   - [ ] Unit tests for AST transformations
   - [ ] Integration test with simple benchmark
   - [ ] Full test with deltablue

## Files Created/Modified

1. **Created**: `static_python_perf_detyper.py`
   - Contains `StaticPythonPerfDetyper` class
   - Box-and-convert implementation (`detype_with_conversions`, `BoxCallSites`)
   - Single-file benchmark handling
   - Experiment running infrastructure

2. **Created**: `run_detype_experiment.py`
   - CLI wrapper for running experiments
   - Commands: `--list`, `--test`, `--show-detyped`, `--detype-indices`
   - Results collection

3. **Modified**: `Dockerfile`
   - Copies all detyper scripts to `/cinder/Tools/benchmarks/`
   - Installs `black` for code formatting
   - Creates `detype-experiment` symlink for easy access

## Usage in Docker Container

```bash
# List available benchmarks
detype-experiment --list

# Quick test (runs typed + fully detyped)
detype-experiment deltablue --test

# Show detyped source code
detype-experiment deltablue --show-detyped

# Run full experiment
detype-experiment deltablue --samples 50

# Detype specific functions
detype-experiment deltablue --detype-indices 0,3,5

# Use different typing level
detype-experiment deltablue --test --typing-level shallow
```
