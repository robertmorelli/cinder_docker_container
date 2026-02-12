"""
Pass: return_cast_container_passthrough

Return-erasure variant for pass-through static containers using cast projection.

Transform shape:
before:
  def make_arr() -> Array[int64]: ...
after:
  def make_arr(): ...
  # callers project as cast(Array[int64], make_arr())
"""

from ast import FunctionDef

from .context import PassContext


PASS_NAME = "return_cast_container_passthrough"
POLICY = "cast"


def apply(fn: FunctionDef, pass_state: dict[str, bool], ctx: PassContext) -> bool:
    # Own only cast-policy return annotations in passthrough-container bin.
    if ctx.annotation_policy(fn.returns) != POLICY:
        return False
    # Require exact pass-name ownership.
    pass_name = ctx.pass_name_for_annotation("return", fn.returns)
    if pass_name != PASS_NAME:
        return False
    # Respect per-function pass enablement.
    if not pass_state[pass_name]:
        return False
    # Remove return annotation and defer projection to call-boundary rewrite.
    fn.returns = None
    return True
