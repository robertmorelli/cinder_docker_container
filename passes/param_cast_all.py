"""
Pass: param_cast_all

Erases cast-policy parameter annotations and reprojects with `cast(...)` (or scalar constructor where configured).

Transform shape:
before:
  def echo(x: Foo) -> Foo: ...
after:
  def echo(_x) -> Foo:
      x: Foo = cast(Foo, _x)
      ...
"""

from ast import AnnAssign, FunctionDef, Load, Name, Store
from copy import deepcopy
from itertools import chain

from .context import PassContext


PASS_NAME = "param_cast_all"
POLICY = "cast"


def _fresh_hidden_name(base_name: str, used_names: set[str]) -> str:
    # Hidden-name generator keeps rewritten arg namespace stable and collision-free.
    hidden = f"_{base_name}"
    while hidden in used_names:
        hidden = f"_{hidden}"
    used_names.add(hidden)
    return hidden


def apply(
    fn: FunctionDef,
    pass_state: dict[str, bool],
    used_names: set[str],
    conversion_stmts: list[AnnAssign],
    ctx: PassContext,
) -> bool:
    # Param detyping pass for generic cast-policy annotations.
    changed = False
    for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs):
        annotation = a.annotation
        if annotation is None:
            continue
        if a.arg in ("self", "cls"):
            continue
        # Guard: this pass should only run on cast-policy annotations.
        if ctx.annotation_policy(annotation) != POLICY:
            continue
        pass_name = ctx.pass_name_for_annotation("param", annotation)
        if pass_name != PASS_NAME:
            continue
        if not pass_state[pass_name]:
            continue
        # Rewrite signature arg and inject cast/construct reprojection statement.
        old_name = a.arg
        hidden_name = _fresh_hidden_name(old_name, used_names)
        a.arg = hidden_name
        annotation_copy = deepcopy(annotation)
        conversion_stmts.append(
            AnnAssign(
                target=Name(id=old_name, ctx=Store()),
                annotation=annotation_copy,
                value=ctx.wrap_cast_or_construct(annotation_copy, Name(id=hidden_name, ctx=Load())),
                simple=1,
            )
        )
        # Remove source annotation to complete detyping.
        a.annotation = None
        changed = True
    return changed
