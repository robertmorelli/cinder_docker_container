"""
Pass: return_construct_container

Erases constructor-backed container return annotation and reconstructs at call boundary.

Transform shape:
before:
  def make_items() -> CheckedList[int64]: ...
after:
  def make_items(): ...
  # callers reconstruct via CheckedList[int64](make_items())
"""

from ast import FunctionDef

from .context import PassContext


PASS_NAME = "return_construct_container"
POLICY = "construct"


def apply(fn: FunctionDef, pass_state: dict[str, bool], ctx: PassContext) -> bool:
    # Own only constructor-policy return annotations.
    if ctx.annotation_policy(fn.returns) != POLICY:
        return False
    # Require exact pass-name ownership.
    pass_name = ctx.pass_name_for_annotation("return", fn.returns)
    if pass_name != PASS_NAME:
        return False
    # Respect per-function pass enablement.
    if not pass_state[pass_name]:
        return False
    # Remove return annotation to make return boundary dynamic.
    fn.returns = None
    return True
