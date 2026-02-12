"""
Scope-aware strategy decider for detyping annotations.

This module is the source of truth for selecting:
- detyping strategy (`box` / `construct` / `cast`),
- concrete pass name for a given scope (`param` / `body` / `return`),
- and no-go / passthrough behavior.
"""

from __future__ import annotations

from ast import BinOp, BitOr, Constant, Name, Subscript, expr
from dataclasses import dataclass
from typing import Literal

from .bins import (
    BOX_LOGIC_BIN,
    CAST_CONTAINER_PASSTHROUGH_LOGIC_BIN,
    CAST_LOGIC_BIN,
    CONSTRUCT_LOGIC_BIN,
    box_primitive_types,
    cast_container_passthrough_bin_by_root,
    construct_bin_by_root,
    explicit_cast_types,
    nogo_types,
)

ScopeName = Literal["param", "body", "return"]
StrategyName = Literal["box", "construct", "cast", "passthrough", "nogo"]


@dataclass(frozen=True)
class StrategyDecision:
    strategy: StrategyName
    pass_name: str | None
    bin_name: str | None

    @property
    def can_detype(self) -> bool:
        return self.pass_name is not None


def is_none_annotation(annotation: expr | None) -> bool:
    if annotation is None:
        return True
    if isinstance(annotation, Constant):
        return annotation.value is None
    if isinstance(annotation, Name):
        return annotation.id == "None"
    return False


def is_dynamic_annotation(annotation: expr | None) -> bool:
    return isinstance(annotation, Name) and annotation.id == "dynamic"


def is_primitive_annotation(annotation: expr | None) -> bool:
    return isinstance(annotation, Name) and annotation.id in box_primitive_types


def is_constructor_annotation(annotation: expr | None) -> bool:
    return (
        isinstance(annotation, Subscript)
        and isinstance(annotation.value, Name)
        and annotation.value.id in construct_bin_by_root
    )


def is_passthrough_container_annotation(annotation: expr | None) -> bool:
    return (
        isinstance(annotation, Subscript)
        and isinstance(annotation.value, Name)
        and annotation.value.id in cast_container_passthrough_bin_by_root
    )


def annotation_root_name(annotation: expr | None) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, Name):
        return annotation.id
    if isinstance(annotation, Subscript):
        return annotation_root_name(annotation.value)
    return None


def is_nogo_annotation(annotation: expr | None) -> bool:
    root_name = annotation_root_name(annotation)
    return root_name in nogo_types


def is_optional_or_union_annotation(annotation: expr | None) -> bool:
    if annotation is None:
        return False
    root_name = annotation_root_name(annotation)
    if root_name in ("Optional", "Union"):
        return True
    return isinstance(annotation, BinOp) and isinstance(annotation.op, BitOr)


def _pass_name(scope_name: ScopeName, strategy: Literal["box", "construct", "cast"], bin_name: str) -> str:
    return f"{scope_name}_{strategy}_{bin_name}"


def decide_scope_strategy(scope_name: ScopeName, annotation: expr | None) -> StrategyDecision:
    if annotation is None or is_none_annotation(annotation) or is_dynamic_annotation(annotation):
        return StrategyDecision(strategy="passthrough", pass_name=None, bin_name=None)

    if is_nogo_annotation(annotation):
        return StrategyDecision(strategy="nogo", pass_name=None, bin_name=None)

    if isinstance(annotation, Name) and annotation.id in explicit_cast_types:
        bin_name = CAST_LOGIC_BIN
        return StrategyDecision(strategy="cast", pass_name=_pass_name(scope_name, "cast", bin_name), bin_name=bin_name)

    if is_primitive_annotation(annotation):
        bin_name = BOX_LOGIC_BIN
        return StrategyDecision(strategy="box", pass_name=_pass_name(scope_name, "box", bin_name), bin_name=bin_name)

    if is_constructor_annotation(annotation):
        assert isinstance(annotation, Subscript), "constructor annotation must be subscript"
        assert isinstance(annotation.value, Name), "constructor annotation base must be name"
        root_name = annotation.value.id
        assert root_name in construct_bin_by_root, f"unknown constructor annotation root: {root_name}"
        bin_name = construct_bin_by_root[root_name]
        assert bin_name == CONSTRUCT_LOGIC_BIN, f"unexpected constructor bin: {bin_name}"
        return StrategyDecision(
            strategy="construct",
            pass_name=_pass_name(scope_name, "construct", bin_name),
            bin_name=bin_name,
        )

    if is_passthrough_container_annotation(annotation):
        assert isinstance(annotation, Subscript), "passthrough container annotation must be subscript"
        assert isinstance(annotation.value, Name), "passthrough container annotation base must be name"
        root_name = annotation.value.id
        assert (
            root_name in cast_container_passthrough_bin_by_root
        ), f"unknown passthrough container annotation root: {root_name}"
        bin_name = cast_container_passthrough_bin_by_root[root_name]
        assert (
            bin_name == CAST_CONTAINER_PASSTHROUGH_LOGIC_BIN
        ), f"unexpected passthrough cast bin: {bin_name}"
        return StrategyDecision(strategy="cast", pass_name=_pass_name(scope_name, "cast", bin_name), bin_name=bin_name)

    bin_name = CAST_LOGIC_BIN
    return StrategyDecision(strategy="cast", pass_name=_pass_name(scope_name, "cast", bin_name), bin_name=bin_name)


def decide_param_strategy(annotation: expr | None) -> StrategyDecision:
    return decide_scope_strategy("param", annotation)


def decide_body_strategy(annotation: expr | None) -> StrategyDecision:
    return decide_scope_strategy("body", annotation)


def decide_return_strategy(annotation: expr | None) -> StrategyDecision:
    return decide_scope_strategy("return", annotation)


def annotation_policy(annotation: expr | None) -> StrategyName:
    # Scope-independent policy query for call-boundary rewrites.
    return decide_scope_strategy("param", annotation).strategy


def pass_name_for_annotation(scope_name: ScopeName, annotation: expr | None) -> str | None:
    return decide_scope_strategy(scope_name, annotation).pass_name
