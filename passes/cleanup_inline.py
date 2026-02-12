"""
Cleanup pass: inline cleanup

Collapses generated parameter-projection prologue in `@inline` functions into a direct return expression
when shape constraints are satisfied.

Shape:
- before:
    @inline
    def f(_x):
        x: int64 = int64(_x)
        return x
- after:
    @inline
    def f(_x):
        return int64(_x)
"""

from __future__ import annotations

from ast import AnnAssign, Attribute, Call, ClassDef, FunctionDef, Load, Name, NodeTransformer, Return, expr
from copy import deepcopy


def is_inline_function(fn: FunctionDef) -> bool:
    # Inline decorator can appear as `@inline` or qualified attribute ending in `.inline`.
    for decorator in fn.decorator_list:
        if isinstance(decorator, Name) and decorator.id == "inline":
            return True
        if isinstance(decorator, Attribute) and decorator.attr == "inline":
            return True
    return False


def _matches_inbound_conversion(stmt: AnnAssign, hidden_name: str) -> bool:
    # Accept either `x: T = T(_x)` or `x: T = cast(T, _x)`-style single-arg/2-arg call projections.
    value = stmt.value
    if not isinstance(value, Call):
        return False
    if len(value.keywords) != 0:
        return False
    if len(value.args) == 1:
        arg0 = value.args[0]
        return isinstance(arg0, Name) and isinstance(arg0.ctx, Load) and arg0.id == hidden_name
    if len(value.args) == 2:
        arg1 = value.args[1]
        return isinstance(arg1, Name) and isinstance(arg1.ctx, Load) and arg1.id == hidden_name
    return False


class InlineArgInliner(NodeTransformer):
    def __init__(self, replacements: dict[str, expr]):
        self.replacements = replacements

    def visit_Name(self, node: Name):
        # Replace reads of local projected names with their source-boundary expression.
        if isinstance(node.ctx, Load) and node.id in self.replacements:
            return deepcopy(self.replacements[node.id])
        return node

    def visit_FunctionDef(self, node: FunctionDef):
        # Do not inline across nested scope boundaries.
        return node

    def visit_ClassDef(self, node: ClassDef):
        # Do not inline into nested class scope.
        return node


def cleanup_inline_function(fn: FunctionDef) -> bool:
    # Only inline-decorated functions are candidates.
    if not is_inline_function(fn):
        return True

    replacement_map: dict[str, expr] = {}
    body_index = 0
    positional = tuple(fn.args.posonlyargs) + tuple(fn.args.args) + tuple(fn.args.kwonlyargs)
    for arg in positional:
        hidden_name = arg.arg
        if not hidden_name.startswith("_"):
            continue
        if body_index >= len(fn.body):
            return False
        stmt = fn.body[body_index]
        if not isinstance(stmt, AnnAssign):
            return False
        if not isinstance(stmt.target, Name):
            return False
        replacement_name = stmt.target.id
        if replacement_name.startswith("_"):
            return False
        if stmt.annotation is None:
            return False
        if not _matches_inbound_conversion(stmt, hidden_name):
            return False
        # Preserve the original projection expression so `return x` becomes `return T(_x)` (not `return _x`).
        replacement_map[replacement_name] = deepcopy(stmt.value)
        body_index += 1

    # Resulting shape must end with exactly one return.
    if body_index == 0:
        return len(fn.body) == 1 and isinstance(fn.body[0], Return)
    if body_index >= len(fn.body):
        return False
    if not isinstance(fn.body[body_index], Return):
        return False
    if len(fn.body) != body_index + 1:
        return False

    # Rewrite `return x` into direct return of projected expression.
    inliner = InlineArgInliner(replacement_map)
    cleaned_return = inliner.visit(fn.body[body_index])
    fn.body = [cleaned_return]
    return True
