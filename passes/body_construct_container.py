"""
Pass: body_construct_container

Erases constructor-backed container local AnnAssign and reconstructs at assignment/use boundaries.

Transform shape:
before:
  xs: CheckedList[int64] = make_items()
after:
  xs = CheckedList[int64](make_items())
  # later reads are reprojected by shared body read projection logic
"""

from ast import Constant, expr

from .context import PassContext


PASS_NAME = "body_construct_container"
POLICY = "construct"


def apply(annotation: expr, value: expr | None, pass_state: dict[str, bool], ctx: PassContext) -> expr | None:
    # Own only constructor-policy annotations.
    if ctx.annotation_policy(annotation) != POLICY:
        return None
    # Require pass-name match for construct-container bin.
    pass_name = ctx.pass_name_for_annotation("body", annotation)
    if pass_name != PASS_NAME:
        return None
    # Respect plan bit for this pass/function.
    if not pass_state[pass_name]:
        return None
    # Body erasure keeps assignment site dynamic; `None` stays explicit constant.
    if value is None:
        return Constant(value=None)
    # Constructor bin shape: T(value)
    return ctx.wrap_construct(annotation, value)
