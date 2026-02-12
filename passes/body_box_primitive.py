"""
Pass: body_box_primitive

Erases primitive local AnnAssign into dynamic assignment and uses primitive projection on later reads.

Transform shape:
before:
  n: int64 = 41
after:
  n = box(int64(41))
  # later reads are projected by body reproject logic in main transformer
"""

from ast import Constant, expr

from .context import PassContext


PASS_NAME = "body_box_primitive"
POLICY = "box"


def apply(annotation: expr, value: expr | None, pass_state: dict[str, bool], ctx: PassContext) -> expr | None:
    # Own only primitive/box-policy annotations.
    if ctx.annotation_policy(annotation) != POLICY:
        return None
    # Require exact pass-name ownership to avoid cross-bin rewrites.
    pass_name = ctx.pass_name_for_annotation("body", annotation)
    if pass_name != PASS_NAME:
        return None
    # Plan says this body pass is disabled.
    if not pass_state[pass_name]:
        return None
    # Body erasure keeps assignment site dynamic; `None` stays explicit constant.
    if value is None:
        return Constant(value=None)
    # Primitive shape: box(Primitive(value))
    return ctx.wrap_box(ctx.coerce_primitive(annotation, value))
