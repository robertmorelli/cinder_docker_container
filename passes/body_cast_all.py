"""
Pass: body_cast_all

Erases cast-policy local AnnAssign and inserts cast-based reprojection around value flow.

Transform shape:
before:
  w: Widget = make_widget()
after:
  w = cast(Widget, make_widget())
  # later reads use cast(Widget, w) in projection sites
"""

from ast import Constant, expr

from .context import PassContext


PASS_NAME = "body_cast_all"
POLICY = "cast"


def apply(annotation: expr, value: expr | None, pass_state: dict[str, bool], ctx: PassContext) -> expr | None:
    # Own only cast-policy annotations.
    if ctx.annotation_policy(annotation) != POLICY:
        return None
    # Require pass-name match for generic cast bin.
    pass_name = ctx.pass_name_for_annotation("body", annotation)
    if pass_name != PASS_NAME:
        return None
    # Respect plan bit for this pass/function.
    if not pass_state[pass_name]:
        return None
    # Known limitation: optional/union flow narrowing is not handled in this pass.
    if ctx.is_optional_or_union_annotation(annotation):
        return None
    # Body erasure keeps assignment site dynamic; `None` stays explicit constant.
    if value is None:
        return Constant(value=None)
    # Cast bin shape: cast(T, value) (or scalar constructor policy hook).
    return ctx.wrap_cast_or_construct(annotation, value)
