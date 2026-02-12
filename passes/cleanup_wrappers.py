"""
Cleanup pass: wrapper cleanup

Removes redundant nested wrappers such as repeated `cast(T, cast(T, x))` and idempotent unary wrappers.

Shape:
- `cast(T, cast(T, x)) -> cast(T, x)`
- `int64(int64(x)) -> int64(x)`
- `box(box(x)) -> box(x)`
"""

from __future__ import annotations

from ast import Call, Name, NodeTransformer, expr

from .bins import box_primitive_types, scalar_construct_types


def annotations_equal(a: expr, b: expr) -> bool:
    # Local import avoids leaking AST dump dependency outside cleanup pass.
    from ast import dump as ast_dump

    return ast_dump(a, include_attributes=False) == ast_dump(b, include_attributes=False)


idempotent_name_wrappers = box_primitive_types.union(scalar_construct_types).union(frozenset(("box",)))


class RedundantWrapperCleaner(NodeTransformer):
    @staticmethod
    def is_unary_name_call(node: expr, name: str) -> bool:
        return (
            isinstance(node, Call)
            and isinstance(node.func, Name)
            and node.func.id == name
            and len(node.args) == 1
            and len(node.keywords) == 0
        )

    @staticmethod
    def is_cast_call(node: expr) -> bool:
        return (
            isinstance(node, Call)
            and isinstance(node.func, Name)
            and node.func.id == "cast"
            and len(node.args) == 2
            and len(node.keywords) == 0
        )

    def visit_Call(self, node: Call):
        self.generic_visit(node)

        # Collapse cast(cast(x)) when annotation is identical.
        if self.is_cast_call(node):
            annotation = node.args[0]
            inner = node.args[1]
            if self.is_cast_call(inner) and annotations_equal(annotation, inner.args[0]):
                return inner

        # Collapse idempotent unary wrappers like int64(int64(x)), box(box(x)).
        if isinstance(node.func, Name) and node.func.id in idempotent_name_wrappers:
            inner = node.args[0] if len(node.args) == 1 and len(node.keywords) == 0 else None
            if inner is not None and self.is_unary_name_call(inner, node.func.id):
                return inner

        return node
