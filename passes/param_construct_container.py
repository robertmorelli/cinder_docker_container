"""
Pass: param_construct_container

Erases constructor-backed container parameter annotations and reconstructs at function entry.
Example: `xs: CheckedList[int64]` becomes `_xs` plus `xs: CheckedList[int64] = CheckedList[int64](_xs)`.

Transform shape:
before:
  def first(xs: CheckedList[int64]) -> int64: ...
after:
  def first(_xs) -> int64:
      xs: CheckedList[int64] = CheckedList[int64](_xs)
      ...
"""

from ast import AnnAssign, FunctionDef, Load, Name, Store
from copy import deepcopy
from itertools import chain

from .context import PassContext


PASS_NAME = "param_construct_container"
POLICY = "construct"


def _fresh_hidden_name(base_name: str, used_names: set[str]) -> str:
    # Hidden-name generator keeps rewritten parameter names collision-free.
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
    # Same rewrite skeleton as other param passes, but policy/pass-name ownership differs.
    changed = False
    for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs):
        annotation = a.annotation
        if annotation is None:
            continue
        if a.arg in ("self", "cls"):
            continue
        # Only constructor-policy parameters are owned by this pass.
        if ctx.annotation_policy(annotation) != POLICY:
            continue
        pass_name = ctx.pass_name_for_annotation("param", annotation)
        if pass_name != PASS_NAME:
            continue
        if not pass_state[pass_name]:
            continue
        # 1) hide original arg name, 2) inject projected annassign at function start.
        old_name = a.arg
        hidden_name = _fresh_hidden_name(old_name, used_names)
        a.arg = hidden_name
        annotation_copy = deepcopy(annotation)
        conversion_stmts.append(
            AnnAssign(
                target=Name(id=old_name, ctx=Store()),
                annotation=annotation_copy,
                value=ctx.wrap_construct(annotation_copy, Name(id=hidden_name, ctx=Load())),
                simple=1,
            )
        )
        # Annotation removal is the detyping boundary.
        a.annotation = None
        changed = True
    return changed
