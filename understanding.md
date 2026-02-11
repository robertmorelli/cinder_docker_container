# Detyper Working Notes

## Purpose
- We have two standalone detypers in this repo:
  - `de_type2.py`: baseline detyper derived from `de_typer.py`.
  - `de_typer_boxunbox.py`: boundary-aware box/unbox detyper derived from `de_typer.py`.
- Goal is to evaluate gradual typing behavior by selectively detyping function boundaries while preserving static checks as much as possible.

## Non-Negotiable Style Rule
- Both `de_type2.py` and `de_typer_boxunbox.py` are based on `de_typer.py`.
- Always follow the coding style of `de_typer.py` and reuse its proven logic where possible.
- In particular, keep/reuse working pieces like graph/dependency analysis and function enumeration structure unless there is a concrete correctness reason to diverge.

## Runtime Environment
- These scripts are expected to run in the Docker environment from this repo.
- Use `start.sh` as the entrypoint.
- Cache-friendly execution:
  - `START_SKIP_BUILD=1 bash start.sh ...` skips image rebuild and reuses Docker cache.
- `start.sh` mounts these host files into container paths:
  - `de_typer.py`
  - `de_type2.py`
  - `de_typer_boxunbox.py`
  - `static-python-perf/`

## Typecheck Command
- Static check command used by detyper workflows:
  - `python -m cinderx.compiler --static`

## Box/Unbox Strategy (Current Intent)
- Transforming is now split into independent per-function passes, not one all-or-nothing detype.
- Per function, pass state is keyed by qualified function identity (same strategy family as `de_typer.py`):
  - `(ancestor_class, function_name)` via class graph analysis.
  - Overrides share the same qualified key, so variants stay linked.

## Boundary Policy Sets
- `box_types`:
  - primitive static scalar types (includes `float`).
  - boundary behavior: primitive coercion + boxing.
- `construct_types`:
  - runtime-enforced static containers (`CheckedList`, `CheckedDict`, `CheckedSet`, `Array`).
  - boundary behavior: constructor call, not cast.
- cast fallback:
  - non-primitive, non-constructor annotations use `cast(T, dyn)`.

## Multi-Pass Plan Model
- Pass names (9):
  - `param_box`
  - `param_construct`
  - `param_cast`
  - `body_box`
  - `body_construct`
  - `body_cast`
  - `return_box`
  - `return_construct`
  - `return_cast`
- Permutations are now over `functions * passes` plan bits.
- `find_permutation_errors()` always includes:
  - fully typed,
  - fully detyped (all passes on),
  - each single-pass-only permutation (one pass enabled across all functions),
  - sampled mixed permutations.
- This allows isolating which erasure category causes failures.
- Architecture is now explicitly de_typer-style at plan level:
  - ordered `plan_names` are generated from `fun_names` as `(qualified_function, pass_name)` tuples,
  - `plan_bit_count` is `len(plan_names)`,
  - permutation-to-guide conversion is split into:
    - `_guide_from_fun_permutation()` for function-level bool vectors,
    - `_guide_from_plan_permutation()` for full plan vectors.

## Inline Function Constraint
- `@inline` functions are sensitive to transformed body shape.
- Prologue insertion can break inline expectations.
- Current approach: reuse normal transformation logic, then run cleanup pass for inline functions to keep compatible shape (single-return form) when possible.

## Common Failure Shapes Seen in Deltablue
- `cinderx.compiler.errors.TypedSyntaxError` dominates.
- Frequent categories include:
  - primitive/dynamic assignment mismatch,
  - invalid primitive call args,
  - incorrect boundary coercions,
  - container generic mismatches.
- Historical transformer bug (fixed earlier):
  - `TypeError: 'FunctionDef' object is not iterable` from inline-related transform issues.

## Host-Side Comparison Script
- `test.sh` runs both detypers from outside container and reports:
  - success/failure totals,
  - top failure types,
  - direct A/B validity comparison.
- Example:
  - `./test.sh --benchmark deltablue --samples 20 --granularity 5`

## How To Run Tests
- Quick A/B from host:
  - `START_SKIP_BUILD=1 ./test.sh --benchmark deltablue --samples 2 --granularity 10`
- Deterministic pass-only sanity check:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --samples 0 --granularity 10`
- Deterministic clear-signal report (typed/detyped + every pass-only case):
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --signals`
- Run boxunbox directly in container:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --samples 8 --granularity 5`
- Run baseline directly in container:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_type2.py deltablue --samples 8 --granularity 5`
- Show transformed source for one permutation:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --show-perm 0x2000000400022`
- Pass-only breakdown example (inside one run):
  - run with `--samples 1 --granularity 10`,
  - parse `info_dump` for pass-only permutations generated by `_perm_with_only_pass()`.

## Current Status Snapshot
- Latest deterministic pass-only deltablue run (`samples=0`, `granularity=20`) produced:
  - `success=6`, `failure=5`, `total=11`.
  - failures: all `TypedSyntaxError`.
- Per-pass-only outcomes from that run:
  - `param_box`: success.
  - `param_construct`: success.
  - `param_cast`: success.
  - `body_box`: fails (`int64 cannot be assigned to dynamic`).
  - `body_construct`: fails (`int64 cannot be assigned to dynamic`).
  - `body_cast`: fails (`dynamic cannot be assigned to int64`).
  - `return_construct`: success.
  - `return_box`: fails (`invalid union type Union[cbool, dynamic] ...`).
  - `return_cast`: success.
- Compatibility sweep for all currently-OK passes:
  - OK set: `param_box`, `param_construct`, `param_cast`, `return_construct`, `return_cast`.
  - all subset combinations of that set were tested (`31/31` successful).
  - full combined OK set also succeeds (`ALL_OK_TOGETHER ok`).
- Latest sampled deltablue run (`samples=1`, `granularity=10`) remained in the same regime:
  - `success=4`, `failure=40`, `total=44`.

## Theory For The 4 Failing Passes
- Failing passes are:
  - `body_box`
  - `body_construct`
  - `body_cast`
  - `return_box`
- `body_box` and `body_cast` are mainly failing because body annassign rewrite currently erases annotations too broadly:
  - evidence (`body_box`): transformed failure at `out.mark = mark` (line 83) with `int64 -> dynamic` mismatch.
  - evidence (`body_cast`): transformed failure at `if self.my_output.mark != mark ...` (line 124) with `dynamic -> int64` mismatch.
  - theory: rewriting `AnnAssign` for `self.<field>` turns typed instance fields into dynamic fields, then later typed reads/writes fail.
- `body_construct` fails for the same structural reason at container constants/typed slots:
  - evidence: `return STRENGTHS[self.strength]` (line 48) with `int64 -> dynamic` mismatch.
  - theory: rewriting typed container annassign (like `CheckedList[...]`) into plain assign drops static container metadata used by downstream indexing/type checks.
- `return_box` fails because primitive-return detype introduces dynamic into boolean/short-circuit paths before reliable projection:
  - evidence: `if cbool(c.is_input()) and c.is_satisfied():` (line 348) triggers `Union[cbool, dynamic]` invalid union.
  - theory: callsite return projection is incomplete for unresolved/dynamic-dispatch attribute calls, so dynamic leaks into primitive boolean flow.

## Robust Fix Theory
- For body passes:
  - detype only local `Name`-target annassign inside function bodies first.
  - do not detype `Attribute` targets (`self.x`) without a full field-type replacement model.
  - do not detype module/class-level typed constants unless we can preserve equivalent static metadata.
  - add explicit asserts for unsupported annassign target/scope shapes so failures are immediate and diagnosable.
- For `return_box`:
  - apply primitive return detype only when all relevant callsites are target-resolved and can be projected.
  - if unresolved callsites remain, skip/assert primitive return detype for that function.
  - longer-term: add lightweight local type-environment tracking so calls like `c.is_satisfied()` can be resolved from annotated local variable types.

## Error Shapes (Observed)
- Dominant:
  - `cinderx.compiler.errors.TypedSyntaxError: type mismatch: int64 cannot be assigned to dynamic`
  - `cinderx.compiler.errors.TypedSyntaxError: type mismatch: dynamic cannot be assigned to int64`
  - `cinderx.compiler.errors.TypedSyntaxError: type mismatch: int cannot be assigned to int64`
- Occasional:
  - compiler `AssertionError` during return type checking

## Representative Failure Examples
- Name-collision method rewrite bug:
  - In one transformed output, `Constraint.__init__(self, strength)` became:
    - `Constraint.__init__(box(int64(self)), strength)`
  - This indicates boundary metadata from one `__init__` leaked into another class's `__init__` callsite.
- Unbound class-style method call indexing bug (fixed):
  - `ClassName.method(self, ...)` was rewritten as if bound, so arg index 0 (`self`) got coerced for parameter 0.
  - Fix: detect unbound class-style calls and offset positional rewrite by one.
- Primitive/dynamic mismatch at method boundary:
  - Typed method body line:
    - `if self.my_output.mark != mark and cbool(stronger(...))`
  - Error shows `mark` seen as `dynamic` in typed context, consistent with boxed arg insertion at wrong callsites.
- CheckedList construction mismatch:
  - Transformed line:
    - `self.v = cast(CheckedList[Constraint], [])`
  - Runtime raises:
    - `TypeError: expected chklist[Constraint], got list`
  - `cast` validates type; it does not construct a `CheckedList`.

## Fixes Implemented So Far
- Reworked boundary metadata and planning to be de_typer-style qualified-key based:
  - metadata keyed by qualified function key, not bare method name.
  - call rewrite resolution uses class context (`self/cls` and `ClassName.method(...)`) where available.
- Split boundary policy into three paths:
  - `box_types` (primitive scalar box/coerce),
  - `construct_types` (constructor-based runtime static containers),
  - cast fallback.
- Fixed constructor-vs-cast for container annotations:
  - `CheckedList[T]`-style annassigns now use constructor path instead of `cast(..., list)`.
- Fixed method-call arg indexing for unbound class-style calls:
  - detect `ClassName.method(self, ...)` and shift rewritten arg index by one.
- Blocked boundary rewriting for dunder method attribute calls (`__init__`, etc.) to avoid high-risk cross-class leakage.
- Added explicit multi-pass toggles so parameter/body/return erasures can be tested independently.
- Expanded body erasure into explicit variants (box/construct/cast) so params/body/returns now share the same 3-way policy split.
- Added dedicated helper families for each split:
  - `detype_params_*`,
  - `detype_body_*`,
  - `detype_return_*`.
- Added class-constructor boundary call rewrite (`ClassName(...)` -> class `__init__` metadata) to make primitive parameter detype pass valid for constructor args.
- Added deterministic `--signals` mode for direct pass/fail reporting by pass.
- Fixed sequential-run correctness bug:
  - `_read_ast()` is cached, but permutations now detype from `deepcopy(_read_ast())` to avoid cross-permutation AST mutation contamination.
- Tried and rejected one experiment:
  - skipping primitive return projection increased union/dynamic failures and reduced success.

## In-Depth Theory Of What Goes Wrong
- A major issue was boundary rewriting keyed too coarsely.
  - Qualified-key + class-context resolution reduced the worst leakage.
  - Remaining leakage still exists for calls where receiver type is not statically resolvable in AST-only analysis.
- Type erasure plus unannotated assignment changes static meaning:
  - Removing annotations can downgrade previously static values to dynamic.
  - Later primitive-sensitive operations then fail (`int64 <-> dynamic` mismatches).
- `cast(T, x)` is projection/check, not conversion/construction.
  - It is valid for reference/non-primitive projection.
  - It cannot replace real constructors for runtime-enforced container implementations (e.g. `CheckedList[T]`).
- Box/construct insertion is only safe when the transform can prove:
  - exact callee identity/signature,
  - expected runtime representation (`chklist` vs `list`),
  - and no unsupported argument form (`*args`, `**kwargs`) at that callsite.
- Remaining dominant errors suggest boundary context is still too coarse:
  - body annassign erasure currently drops static slot/container metadata in ways that propagate dynamic into typed paths,
  - primitive return projection is incomplete at unresolved dynamic-dispatch callsites, which can still widen bool paths into invalid unions.

## Can Box/Unbox Fix These Error Types?
- Yes, partially:
  - `int64 <-> dynamic` mismatches can be reduced by correctly placed boundary projections/coercions.
  - Some call arg type mismatches are directly fixable by better call-target precision.
- Not by itself:
  - `chklist` vs `list` needs constructor-aware rewriting, not just box/cast.
  - Method-name-collision rewrites require better call graph precision, not more boxing.

## Practical Fix Directions
- Keep strengthening target-aware rewrite:
  - only rewrite calls when receiver class identity is known from AST context.
  - skip/abort (assert) when call target cannot be resolved safely.
- Add explicit unsupported asserts where precision is not available:
  - ambiguous same-name method targets,
  - dynamic dispatch shapes the analysis cannot resolve,
  - starred arg forms when rewriting is required.
- Add container-aware rewrite rules:
  - detect `CheckedList[T]` annotation and construct `CheckedList[T](...)` instead of `cast(CheckedList[T], ...)`.
- Restrict body detype scope:
  - start with local-name annassign only and assert on `Attribute`/module/class-level annassign shapes.
- Consider preserving selected annotations (or annotation-equivalent typed constructions) where erasure would collapse essential static/container semantics.
- Next likely high-value fix:
  - gate `return_box` by callsite resolvability (or assert when unresolved) to prevent dynamic leakage into cbool/primitive control-flow paths.

## Practical Guidance for Future Edits
- Prefer incremental changes in `de_typer_boxunbox.py` and validate with small sample runs first.
- Keep transformations conservative when signatures are ambiguous (especially methods with same name across classes).
- Preserve standalone script behavior (no library/test-harness dependency assumptions).
