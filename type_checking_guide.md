# Static Python Type Checking Guide

## Overview

Static Python has a multi-layered type checking system. Understanding when checks happen is crucial for writing safe code.

## When Type Checks Happen

### 1. Compile-Time Checks (Static Compiler)

When using the static compiler directly, **all type errors are caught at compile time**:

```bash
python -m cinderx.compiler --static mymodule.py
```

Example error:
```
cinderx.compiler.errors.TypedSyntaxError: type mismatch: str received for positional arg 'x', expected int64
```

This is the **strictest mode** and catches errors before any code runs.

### 2. Runtime Type Guards (Cross-Module Calls)

When non-static code calls into static code, **runtime type guards are enforced**:

```python
# static_module.py
import __static__
from __static__ import int64

def compute(x: int64) -> int64:
    return x * 2
```

```python
# caller.py (non-static)
from static_module import compute
compute("hello")  # TypeError: compute expected 'int' for argument x, got 'str'
```

### 3. No Guards for Static-to-Static Calls

When static code calls other static code, **type guards are skipped**:

```python
# all_static.py
import __static__
from __static__ import int64

def compute(x: int64) -> int64:
    return x * 2

def main():
    compute("hello")  # NO ERROR - guards skipped!
```

This is **by design** for performance. The compiler already verified types at compile time, so runtime checks are redundant.

### 4. Explicit Conversions Always Check

Using explicit type constructors always validates:

```python
from __static__ import int64
x = int64("not a number")  # ValueError: invalid literal for int()
```

### 5. Checked Containers Always Enforce

`CheckedList` and `CheckedDict` enforce types regardless of caller:

```python
from __static__ import CheckedList, CheckedDict

lst: CheckedList[int] = CheckedList[int]([1, 2, 3])
lst.append("string")  # TypeError: chklist[int].append()() argument 1 expected int

d: CheckedDict[str, int] = CheckedDict[str, int]({"a": 1})
d["b"] = "not int"  # TypeError: bad value 'str' for chkdict[str, int]
```

## Summary Table

| Scenario | Type Checking | When |
|----------|---------------|------|
| Static compiler (`-m cinderx.compiler --static`) | Full compile-time | Before execution |
| Non-static → Static function call | Runtime guard | At function entry |
| Static → Static function call | **None** | Skipped for performance |
| `int64(value)`, `cbool(value)` | Runtime conversion | At conversion |
| `CheckedList[T]`, `CheckedDict[K,V]` | Runtime enforcement | At every operation |
| `cast(Type, value)` | Runtime check | At cast |

## Why Are Static-to-Static Guards Skipped?

From `cinderx/Jit/codegen/gen_asm.cpp`:
```cpp
if (code->co_flags & CO_STATICALLY_COMPILED) {
    // If we've been invoked statically we can skip all of the
    // argument checking because we know our args have been
    // provided correctly.
```

**Rationale:**
1. The static compiler already verified types at compile time
2. Runtime guards add overhead (~3 machine instructions per argument)
3. Within a static module, the compiler guarantees type safety

## Historical Context

This behavior has been the design since Static Python's inception. The documentation explicitly states:

> "These checks are extremely fast, and **omitted when the caller function is also part of a Static Python module**."
> — CinderDoc/static_python/README.md

There is no evidence this was ever different. The design philosophy prioritizes:
- **Correctness** at module boundaries (runtime guards)
- **Performance** within modules (compile-time verification only)

## How to Get Strict Type Checking

### Option 1: Use the Static Compiler Directly (Recommended)

```bash
# Compile and check for errors
python -m cinderx.compiler --static mymodule.py

# Or compile to .pyc
python -m cinderx.compiler --static -c --output mymodule.pyc mymodule.py
```

This catches ALL type errors at compile time before the code runs.

### Option 2: Use Checked Containers

Replace regular containers with checked versions:

```python
# Instead of:
data: list[int] = []

# Use:
from __static__ import CheckedList
data: CheckedList[int] = CheckedList[int]([])
```

### Option 3: Add Explicit Casts

Use `cast()` to add runtime validation points:

```python
from __static__ import cast, int64

def process(x: int64) -> int64:
    # Validate x is actually an int64 at this point
    validated: int64 = cast(int64, x)
    return validated * 2
```

### Option 4: Structure Code for Boundary Checks

Keep static code in separate modules from dynamic code. Type guards trigger at module boundaries:

```
static_lib/
  __init__.py      # import __static__
  compute.py       # Pure static code

app/
  main.py          # Regular Python, calls into static_lib
```

### Option 5: Build-Time Type Checking

Integrate the static compiler into your build process:

```makefile
typecheck:
    find . -name "*.py" -exec grep -l "import __static__" {} \; | \
    xargs -I {} python -m cinderx.compiler --static {}
```

## No Flag for Forcing Runtime Guards

There is **no JIT flag** to force runtime type guards for static-to-static calls. The available JIT flags (see `python -X jit-help`) do not include such an option.

If you need guaranteed runtime type checking:
1. Use the static compiler for compile-time checks
2. Use `CheckedList`/`CheckedDict` for container types
3. Use `cast()` for explicit validation points
4. Structure code so calls cross static/non-static boundaries

## Primitive Types vs Object Types

**Primitives** (`int64`, `cbool`, `double`, etc.):
- JIT optimization hints
- No automatic runtime guards within static code
- Explicit conversions (`int64(x)`) do validate

**Object types** (`str`, `list`, custom classes):
- Guards at static module boundaries
- No guards for static-to-static calls
- Use `CheckedList`/`CheckedDict` for enforcement

## Conclusion

Static Python's type checking is designed for a balance of safety and performance:

- **Maximum safety**: Use the static compiler (`-m cinderx.compiler --static`)
- **Runtime safety at boundaries**: Automatic when calling from non-static code
- **Performance within modules**: Guards skipped, trust the compiler
- **Explicit safety**: Use `CheckedList`, `CheckedDict`, and `cast()`
