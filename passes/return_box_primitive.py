"""
Pass: return_box_primitive

Erases primitive return annotation and projects result at call boundary.

Transform shape:
before:
  def inc(x: int64) -> int64: ...
after:
  def inc(x: int64): ...
  # callers project the boundary result back to int64
"""

from ast import FunctionDef

from .context import PassContext


PASS_NAME = "return_box_primitive"
POLICY = "box"


def apply(fn: FunctionDef, pass_state: dict[str, bool], ctx: PassContext) -> bool:
    # Own only primitive/box return annotations.
    if ctx.annotation_policy(fn.returns) != POLICY:
        return False
    # Require exact pass-name ownership.
    pass_name = ctx.pass_name_for_annotation("return", fn.returns)
    if pass_name != PASS_NAME:
        return False
    # Respect per-function pass enablement.
    if not pass_state[pass_name]:
        return False
    # Remove the return annotation to detype this boundary.
    fn.returns = None
    return True
