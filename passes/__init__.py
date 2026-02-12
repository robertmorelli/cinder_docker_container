"""
Pass registry for de_typer_boxunbox.

The main detyper iterates these tuples in order. Each callable performs one specific
pass rewrite attempt and either:
- returns a change marker (`True` / rewritten node), or
- returns no-op (`False` / `None`) when its pass does not apply.
"""

# Parameter-erasure pass entrypoints.
from .param_box_primitive import apply as apply_param_box_primitive
from .param_construct_container import apply as apply_param_construct_container
from .param_cast_all import apply as apply_param_cast_all
from .param_cast_container_passthrough import apply as apply_param_cast_container_passthrough

# Body-annassign erasure pass entrypoints.
from .body_box_primitive import apply as apply_body_box_primitive
from .body_construct_container import apply as apply_body_construct_container
from .body_cast_all import apply as apply_body_cast_all
from .body_cast_container_passthrough import apply as apply_body_cast_container_passthrough

# Return-annotation erasure pass entrypoints.
from .return_box_primitive import apply as apply_return_box_primitive
from .return_construct_container import apply as apply_return_construct_container
from .return_cast_all import apply as apply_return_cast_all
from .return_cast_container_passthrough import apply as apply_return_cast_container_passthrough


# Ordered parameter-pass pipeline.
PARAM_PASS_APPLIERS = (
    apply_param_box_primitive,
    apply_param_construct_container,
    apply_param_cast_all,
    apply_param_cast_container_passthrough,
)

# Ordered body-pass pipeline.
BODY_PASS_APPLIERS = (
    apply_body_box_primitive,
    apply_body_construct_container,
    apply_body_cast_all,
    apply_body_cast_container_passthrough,
)

# Ordered return-pass pipeline.
RETURN_PASS_APPLIERS = (
    apply_return_box_primitive,
    apply_return_construct_container,
    apply_return_cast_all,
    apply_return_cast_container_passthrough,
)
