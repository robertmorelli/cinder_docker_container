"""
Shared pass context for per-pass AST transforms.

Each pass module receives a `PassContext` instance so pass logic can stay local to
the pass file while reusing common policy helpers from the main detyper.
"""

from __future__ import annotations

from ast import expr
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PassContext:
    """
    Injectable helper surface for pass modules.

    Passes call these helpers instead of duplicating main-detyping policy code.
    This keeps pass modules focused on "when to apply" and "what shape to emit."
    """
    # Maps annotation AST -> policy string ("box" / "construct" / "cast" / "passthrough").
    annotation_policy: Callable[[expr | None], str]
    # Maps scope + annotation AST -> canonical pass name.
    pass_name_for_annotation: Callable[[str, expr | None], str | None]
    # Primitive projection helper, e.g. int64(node) when annotation is int64.
    coerce_primitive: Callable[[expr | None, expr], expr]
    # Runtime constructor wrapper helper, e.g. CheckedList[T](node).
    wrap_construct: Callable[[expr, expr], expr]
    # Cast/construct projection helper for cast-policy types.
    wrap_cast_or_construct: Callable[[expr, expr], expr]
    # Box wrapper helper used for primitive body erasure.
    wrap_box: Callable[[expr], expr]
    # Detects passthrough static-container annotation roots.
    is_passthrough_container_annotation: Callable[[expr | None], bool]
    # Optional/union detector used by cast-all body pass guard.
    is_optional_or_union_annotation: Callable[[expr | None], bool]
