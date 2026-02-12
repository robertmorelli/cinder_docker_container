"""
Pass: body_cast_container_passthrough

Body-erasure variant for pass-through static containers using cast projection.

Transform shape:
before:
  arr: Array[int64] = xs
after:
  arr = cast(Array[int64], xs)
  # later reads project as cast(Array[int64], arr)[...]
"""

from ast import Constant, expr

from .context import PassContext


PASS_NAME = "body_cast_container_passthrough"
POLICY = "cast"


def apply(annotation: expr, value: expr | None, pass_state: dict[str, bool], ctx: PassContext) -> expr | None:
    # This pass handles cast-policy passthrough container annotations only.
    if ctx.annotation_policy(annotation) != POLICY:
        return None
    # Require pass-name match for container passthrough cast bin.
    pass_name = ctx.pass_name_for_annotation("body", annotation)
    if pass_name != PASS_NAME:
        return None
    # Respect per-function pass enablement.
    if not pass_state[pass_name]:
        return None
    # Body erasure keeps assignment site dynamic; `None` stays explicit constant.
    if value is None:
        return Constant(value=None)
    # Passthrough container shape: cast(Container[T], value)
    return ctx.wrap_cast_or_construct(annotation, value)
