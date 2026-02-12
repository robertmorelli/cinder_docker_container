"""
Pass: return_cast_all

Erases cast-policy return annotation and projects/casts at call boundary.

Transform shape:
before:
  def make_foo() -> Foo: ...
after:
  def make_foo(): ...
  # callers project as cast(Foo, make_foo())
"""

from ast import FunctionDef

from .context import PassContext


PASS_NAME = "return_cast_all"
POLICY = "cast"


def apply(fn: FunctionDef, pass_state: dict[str, bool], ctx: PassContext) -> bool:
    # Own only cast-policy return annotations.
    if ctx.annotation_policy(fn.returns) != POLICY:
        return False
    # Require exact pass-name ownership.
    pass_name = ctx.pass_name_for_annotation("return", fn.returns)
    if pass_name != PASS_NAME:
        return False
    # Respect per-function pass enablement.
    if not pass_state[pass_name]:
        return False
    # Remove return annotation and let boundary rewrite handle caller projection.
    fn.returns = None
    return True
