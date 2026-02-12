"""
Pass: param_box_primitive

Erases primitive parameter annotations and inserts an in-function primitive reprojection.
Example: `def f(x: int64)` -> `def f(_x); x: int64 = int64(_x)`.

Transform shape:
before:
  def add_one(x: int64) -> int64: ...
after:
  def add_one(_x) -> int64:
      x: int64 = int64(_x)
      ...
"""

from ast import AnnAssign, Load, Name, Store
from copy import deepcopy
from itertools import chain
from ast import FunctionDef

from .context import PassContext


PASS_NAME = "param_box_primitive"
POLICY = "box"


def _fresh_hidden_name(base_name: str, used_names: set[str]) -> str:
    # Preserve user names by shifting the original parameter behind one or more "_" prefixes.
    # This mirrors the old in-file helper behavior exactly.
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
    # Track whether this pass changed anything in this function.
    changed = False
    # Visit all user-facing parameter slots this pass supports.
    for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs):
        annotation = a.annotation
        # Untyped arg: nothing to erase.
        if annotation is None:
            continue
        # Instance/class receiver parameters stay typed.
        if a.arg in ("self", "cls"):
            continue
        # This pass only owns primitive/box-policy annotations.
        if ctx.annotation_policy(annotation) != POLICY:
            continue
        # Require exact pass-name ownership to avoid cross-bin rewrites.
        pass_name = ctx.pass_name_for_annotation("param", annotation)
        if pass_name != PASS_NAME:
            continue
        # Plan disabled for this pass/function.
        if not pass_state[pass_name]:
            continue

        # Rewrite `x: T` param to hidden dynamic param `_x`.
        old_name = a.arg
        hidden_name = _fresh_hidden_name(old_name, used_names)
        a.arg = hidden_name

        # Add explicit inbound projection `x: T = T(_x)` at function entry.
        annotation_copy = deepcopy(annotation)
        conversion_stmts.append(
            AnnAssign(
                target=Name(id=old_name, ctx=Store()),
                annotation=annotation_copy,
                value=ctx.coerce_primitive(annotation_copy, Name(id=hidden_name, ctx=Load())),
                simple=1,
            )
        )
        # Remove source annotation: this is the actual detyping step.
        a.annotation = None
        changed = True
    return changed
