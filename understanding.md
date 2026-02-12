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

## Non-Negotiable Workflow Rule
- Always run tests first after each meaningful transformer change, before additional refactors.
- Minimum gate for `de_typer_boxunbox.py`:
  - `python3 -m py_compile de_typer_boxunbox.py`
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --test`
- For pass architecture changes, also run:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --signals`

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

## Added Limitation: Imported Nominal Cast Targets
- Known limitation family:
  - boundary rewrites that emit `cast(Name, dyn)` or `cast(module.Type, dyn)` for imported nominal types can fail with:
    - `TypedSyntaxError: cast to unknown type`
- Practical impact:
  - multi-file imported nominal benchmarks (for example `sample_fsm`) can fail for this reason.
  - this is not a typical single-file benchmark shape.
- Policy:
  - keep this as a known limitation signal in regressions.
  - default fuzz list excludes `sample_fsm` so fuzzing finds new transform bugs in single-file targets first.
  - strict mode now asserts on this shape with limitation key `imported_nominal_cast`.

## Box/Unbox Strategy (Current Intent)
- Transforming is now split into independent per-function passes, not one all-or-nothing detype.
- Per function, pass state is keyed by qualified function identity (same strategy family as `de_typer.py`):
  - `(ancestor_class, function_name)` via class graph analysis.
  - Overrides share the same qualified key, so variants stay linked.

## Boundary Policy Sets
- `box_primitive_types`:
  - primitive static scalar types (does **not** include Python `float`).
  - boundary behavior: primitive coercion + boxing.
- `explicit_cast_types`:
  - explicit scalar names that must use cast policy.
  - currently includes `float`.
- `construct_container_types`:
  - runtime-enforced static containers (`CheckedList`, `CheckedDict`, `CheckedSet`, `Array`).
  - boundary behavior: constructor call, not cast.
- cast fallback bin:
  - non-primitive, non-constructor annotations use `cast(T, dyn)` via the `other` cast bin.

## Multi-Pass Plan Model
- Pass names are generated from:
  - scope: `param`, `body`, `return`
  - logic-family bins:
    - `box`: `primitive`
    - `construct`: `container`
    - `cast`: `all`
- Total passes: `3 * 3 = 9`.
- Example pass names:
  - `param_box_primitive`
  - `body_construct_container`
  - `return_cast_all`
- Permutations are function-level only (same as original `de_typer.py`):
  - one plan bit per qualified function key.
  - all enabled functions in a permutation use the active pass profile.
- `find_permutation_errors()` always includes:
  - fully typed,
  - fully detyped (all passes on),
  - sampled mixed permutations.
- Pass-category isolation is done by `--signals` pass profiles:
  - typed baseline,
  - fully detyped (all passes),
  - per-pass profile with all functions enabled and exactly one pass active.
- Architecture is now explicitly de_typer-style at plan level:
  - ordered `plan_names` are the same qualified function keys from `fun_names`,
  - `plan_bit_count == len(fun_names)`,
  - class-ancestor unification of overrides remains in place.

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
  - note: this executes `2 + 9` runs in signal mode (same function mask, different pass profiles).
- Run boxunbox directly in container:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --samples 8 --granularity 5`
- Run baseline directly in container:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_type2.py deltablue --samples 8 --granularity 5`
- Show transformed source for one permutation:
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --show-perm 0x2000000400022`
- Pass-only breakdown:
  - use `--signals` (pass-only cases are emitted directly by the script).

## Current Status Snapshot (Logic-Family Passes)
- Gate tests run after pass expansion:
  - `python3 -m py_compile de_typer_boxunbox.py`: pass.
  - `START_SKIP_BUILD=1 bash start.sh /cinder/python /cinder/Tools/benchmarks/de_typer_boxunbox.py deltablue --test`:
    - typed return code: `0`
    - fully detyped return code: `1`
    - dominant failure: `TypedSyntaxError` at transformed line `out.mark = mark` (`int64` assigned to `dynamic`).
- `--signals` now reports 9 generated pass names and `plan bits: 50` for deltablue.
- `--signals` currently can stall in runtime execution for some pass-only cases; this is an execution-time issue, not pass-name generation.
- Latest DeltaBlue pass-only outcomes:
  - `param_box_primitive`: ok
  - `param_construct_container`: ok
  - `param_cast_all`: ok
  - `body_box_primitive`: fail (`int64 cannot be assigned to dynamic`)
  - `body_construct_container`: fail (`int64 cannot be assigned to dynamic`)
  - `body_cast_all`: fail (`dynamic cannot be assigned to int64`)
  - `return_box_primitive`: ok
  - `return_construct_container`: ok
  - `return_cast_all`: ok

## Reduced Advanced Correctness Sample (Recommended)
- Recommended fast sample set (advanced only):
  - `deltablue`
  - `call_simple`
  - `call_method_slots`
  - `chaos`
  - `nqueens`
  - `richards`
  - `sample_fsm`
- Why this set:
  - includes known-good baselines (`call_simple`, `call_method_slots`, `deltablue`),
  - includes known stressors for cast/construct/arg-mapping (`chaos`, `nqueens`, `richards`, `sample_fsm`),
  - avoids benchmarks with external-data dependencies for routine regression checks.
- After moving `float` to cast policy, this sample produced:
  - typed baseline: `7/7` ok.
  - combined pass set (`param_* + return_*`): `3/7` ok, `4/7` fail.
  - pass-only outcomes:
    - `param_box`: `6 ok / 1 fail`
    - `param_construct`: `6 ok / 1 fail`
    - `param_cast`: `5 ok / 2 fail`
    - `return_box`: `7 ok / 0 fail`
    - `return_construct`: `6 ok / 1 fail`
    - `return_cast`: `5 ok / 2 fail`
- Improvement observed from float rebin:
  - prior `chaos` failures in primitive passes (`can't box non-primitive: float`) no longer appear.
  - `chaos` now passes `param_box` and `return_box`; remaining failures are cast-related.
- Failure categories in this reduced set:
  - combined failures:
    - `cast_unknown_type` (`2`)
    - `arg_type_mismatch` (`2`)
  - by pass:
    - `param_box`: `arg_type_mismatch`
    - `param_construct`: `arg_type_mismatch`
    - `param_cast`: `cast_unknown_type`
    - `return_box`: none in this sample
    - `return_construct`: `int_required_runtime`
    - `return_cast`: `cast_unknown_type`, `iterator_runtime_typeerror`

## Advanced Benchmark Sweep (Working Pass Set, Pre-Float-Rebin)
- Scope:
  - only `*/advanced/main.py` benchmarks were tested.
  - pass set tested: `param_box`, `param_construct`, `param_cast`, `return_box`, `return_construct`, `return_cast`.
  - this full-sweep snapshot was taken before moving `float` from primitive-box policy to cast policy.
- Aggregate (`24` advanced benchmarks):
  - typed baseline: `typed_ok=16`, `typed_fail=8`, `typed_error=0`.
  - combined working-pass set: `combined_ok=9`, `combined_fail=14`, `combined_error=1`.
  - per-pass success counts:
    - `param_box`: `14 ok / 9 fail / 1 error`
    - `param_construct`: `15 ok / 8 fail / 1 error`
    - `param_cast`: `15 ok / 8 fail / 1 error`
    - `return_box`: `15 ok / 8 fail / 1 error`
    - `return_construct`: `14 ok / 9 fail / 1 error`
    - `return_cast`: `12 ok / 11 fail / 1 error`
- Combined-pass success benchmarks:
  - `call_method`, `call_method_slots`, `call_simple`, `deltablue`, `evolution`, `fannkuch`, `http2`, `nbody`, `pystone`.
- Important interpretation note:
  - several benchmarks fail even in typed baseline (e.g. missing data file in `futen`, pre-existing compiler/runtime issues in others), so not every sweep failure is introduced by detyping.
- Dominant non-baseline failure shapes for this pass set:
  - `can't box non-primitive: float` (primitive boxing assumptions mismatch),
  - `type mismatch: Array[int64] ... expected int` (container constructor/projection mismatch),
  - `cast to unknown type` / iterator cast issues (cast fallback too broad in these programs).

## Theory For The 3 Failing Passes
- Failing passes are:
  - `body_box`
  - `body_construct`
  - `body_cast`
- `body_box` and `body_cast` are mainly failing because body annassign rewrite currently erases annotations too broadly:
  - evidence (`body_box`): transformed failure at `out.mark = mark` (line 83) with `int64 -> dynamic` mismatch.
  - evidence (`body_cast`): transformed failure at `if self.my_output.mark != mark ...` (line 124) with `dynamic -> int64` mismatch.
  - theory: rewriting `AnnAssign` for `self.<field>` turns typed instance fields into dynamic fields, then later typed reads/writes fail.
- `body_construct` fails for the same structural reason at container constants/typed slots:
  - evidence: `return STRENGTHS[self.strength]` (line 48) with `int64 -> dynamic` mismatch.
  - theory: rewriting typed container annassign (like `CheckedList[...]`) into plain assign drops static container metadata used by downstream indexing/type checks.

## Root-Cause Verdict And Priority
- The remaining failing passes now share one root-cause family:
  - `body_box`, `body_construct`, `body_cast` fail because annassign erasure currently changes storage semantics (typed field/container/constant state becomes effectively dynamic in downstream use).
- Easiest pass to fix first: `body_construct`.
  - rationale: failures are concentrated around container/constants metadata loss and can be reduced quickly by scoping rewrite to local-name annassign only, with asserts on unsupported targets/scopes.
  - this is lower-risk than broad body_box/body_cast changes because construct targets are easier to isolate.

## Robust Fix Theory
- For body passes:
  - detype only local `Name`-target annassign inside function bodies first.
  - do not detype `Attribute` targets (`self.x`) without a full field-type replacement model.
  - do not detype module/class-level typed constants unless we can preserve equivalent static metadata.
  - add explicit asserts for unsupported annassign target/scope shapes so failures are immediate and diagnosable.

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
- Added unresolved attribute-call return fallback:
  - if a method name has a single consistent return annotation across classes, use it to project return at ambiguous attribute callsites.
  - this moved `return_box` from failing to passing in deltablue pass-only checks.
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
  - callsite projection is improved, but body erasure remains the dominant blocker.

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
  - add target/scope-aware body detype guards (local-name only first), then reintroduce broader body variants with explicit asserts.

## Practical Guidance for Future Edits
- Prefer incremental changes in `de_typer_boxunbox.py` and validate with small sample runs first.
- Keep transformations conservative when signatures are ambiguous (especially methods with same name across classes).
- Preserve standalone script behavior (no library/test-harness dependency assumptions).
