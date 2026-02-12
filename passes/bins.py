"""
Type bin definitions for de_typer_boxunbox.

These bins drive pass-name generation and policy routing:
- box primitive
- construct container
- cast all (fallback)
- cast container passthrough
"""

# Primitive static types that use box/unbox boundaries.
box_primitive_type_order = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "double",
    "single",
    "char",
    "cbool",
)
box_primitive_types = frozenset(box_primitive_type_order)

# Explicit cast roots (kept for extension; currently empty).
explicit_cast_type_order = ()
explicit_cast_types = frozenset(explicit_cast_type_order)

# Scalar constructor roots that should use runtime constructor projection.
scalar_construct_type_order = ("float",)
scalar_construct_types = frozenset(scalar_construct_type_order)

# Types intentionally excluded from detyping because they require richer flow/protocol handling.
nogo_types = frozenset(
    (
        "Iterator",
        "Iterable",
        "Generator",
        "AsyncIterator",
        "AsyncGenerator",
        "Coroutine",
        "Protocol",
        "Callable",
    )
)

# Container roots that are safe to reconstruct with explicit runtime constructor calls.
container_construct_type_order = (
    "CheckedList",
    "CheckedDict",
    "CheckedSet",
)
container_construct_types = frozenset(container_construct_type_order)

# Static container roots that should pass through and be reprojected via cast, not construction.
container_passthrough_type_order = (
    "Array",
    "Dict",
    "List",
    "Set",
    "Tuple",
)
container_passthrough_types = frozenset(container_passthrough_type_order)

# Pass-logic bin identifiers used in pass-name generation.
BOX_LOGIC_BIN = "primitive"
CONSTRUCT_LOGIC_BIN = "container"
CAST_LOGIC_BIN = "all"
CAST_CONTAINER_PASSTHROUGH_LOGIC_BIN = "container_passthrough"

# Root-name to bin lookups for fast classification.
construct_bin_by_root = dict((root_name, CONSTRUCT_LOGIC_BIN) for root_name in container_construct_types)
cast_container_passthrough_bin_by_root = dict(
    (root_name, CAST_CONTAINER_PASSTHROUGH_LOGIC_BIN) for root_name in container_passthrough_types
)

# Fallback cast bin for non-specialized cast-policy types.
CAST_FALLBACK_BIN = "other"
