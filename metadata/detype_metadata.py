from __future__ import annotations

from ast import AST, ClassDef, FunctionDef, Name, iter_child_nodes
from dataclasses import dataclass
from functools import reduce


GuideKey = tuple[str | None, str | None]


@dataclass(frozen=True)
class DetypeMetadata:
    class_antrs: dict[None | str, frozenset[str]]
    fun_names: tuple[GuideKey, ...]
    ambiguous_method_names: frozenset[str]
    plan_names: tuple[GuideKey, ...]

    @staticmethod
    def _class_root_names(class_antrs: dict[None | str, frozenset[str]], class_name: str) -> frozenset[str]:
        assert class_name in class_antrs, f"unknown class in ancestor graph: {class_name}"
        ancestors = class_antrs[class_name]
        if len(ancestors) == 0:
            return frozenset((class_name,))
        roots = frozenset(a for a in ancestors if len(class_antrs[a]) == 0)
        if len(roots) == 0:
            # Defensive fallback for any unexpected graph shape.
            return frozenset((class_name,))
        return roots

    @staticmethod
    def _enumerate_funs(tree: AST, top_level: GuideKey):
        ordered_q_names = [top_level]
        q_fun_names_set: set[GuideKey] = {top_level}

        antr_graph: dict[None | str, frozenset[str]] = {None: frozenset()}

        def antr_classes(anter_exprs: list[Name]) -> frozenset[str]:
            assert all(isinstance(a, Name) for a in anter_exprs), "only simple inheritance supported"
            antr_names: tuple[str, ...] = tuple(a.id for a in anter_exprs if a.id in antr_graph)
            anter_set: frozenset[str] = frozenset(antr_names)
            antr_name_sets: tuple[frozenset[str], ...] = tuple(
                frozenset(antr_graph[antr_name]) for antr_name in antr_names
            )
            return reduce(lambda a, b: a.union(b), antr_name_sets, anter_set)

        def update_antr_graph(class_node: ClassDef):
            class_name = class_node.name
            assert class_name not in antr_graph, "class name shadowing not supported"
            antr_graph[class_name] = frozenset(antr_classes(class_node.bases))  # type: ignore[arg-type]

        def antr_fun_gen(fun_name: str, antr_name: str | None = None):
            assert antr_name in antr_graph, f"function processed before class ancestors: {antr_name} {fun_name}"
            anters = antr_graph[antr_name]
            return tuple((anter, fun_name) for anter in anters)

        def fun_exists(antr_name: str | None, fun_name: str):
            return (antr_name, fun_name) in q_fun_names_set

        def is_fun_overload(fun_name: str, antr_name: str | None = None):
            assert not fun_exists(antr_name, fun_name), "function name shadowing not supported"
            return not any(fun_exists(*q_name) for q_name in antr_fun_gen(fun_name, antr_name=antr_name))

        def count_gen(node: AST, antr_name: str | None = None, fun_name: str | None = None):
            inside_fun = fun_name is not None
            inside_class = antr_name is not None
            for child_node in iter_child_nodes(node):
                is_fun = isinstance(child_node, FunctionDef)
                is_class = isinstance(child_node, ClassDef)
                assert not (is_fun and inside_fun), "function inside function not supported"
                assert not (is_class and inside_fun), "class inside function not supported"
                assert not (is_class and inside_class), "class inside class not supported"
                if is_fun:
                    if is_fun_overload(child_node.name, antr_name):
                        key = (antr_name, child_node.name)
                        q_fun_names_set.add(key)
                        ordered_q_names.append(key)
                        yield 1
                    else:
                        yield 0
                    yield from count_gen(child_node, antr_name=antr_name, fun_name=child_node.name)
                elif is_class:
                    update_antr_graph(child_node)
                    yield from count_gen(child_node, antr_name=child_node.name, fun_name=fun_name)
                else:
                    yield 0

        fun_count = sum(count_gen(tree)) + 1
        assert fun_count == len(ordered_q_names), "function counting failure"
        return antr_graph, tuple(ordered_q_names)

    @classmethod
    def _enumerate_ambiguous_method_names(
        cls, class_antrs: dict[None | str, frozenset[str]], fun_names: tuple[GuideKey, ...]
    ) -> frozenset[str]:
        roots_by_method: dict[str, set[frozenset[str]]] = {}
        for antr_name, fun_name in fun_names:
            if antr_name is None:
                continue
            root_sig = cls._class_root_names(class_antrs, antr_name)
            roots_by_method.setdefault(fun_name, set()).add(root_sig)
        return frozenset(method_name for method_name, root_sigs in roots_by_method.items() if len(root_sigs) > 1)

    @staticmethod
    def _enumerate_plan_names(fun_names: tuple[GuideKey, ...], ambiguous_method_names: frozenset[str]):
        def include_plan_name(q_name: GuideKey):
            antr_name, fun_name = q_name
            if antr_name is None:
                return True
            return fun_name not in ambiguous_method_names

        return tuple(q_name for q_name in fun_names if include_plan_name(q_name))

    @classmethod
    def build(cls, tree: AST, top_level: GuideKey):
        class_antrs, fun_names = cls._enumerate_funs(tree, top_level)
        ambiguous_method_names = cls._enumerate_ambiguous_method_names(class_antrs, fun_names)
        plan_names = cls._enumerate_plan_names(fun_names, ambiguous_method_names)
        assert len(plan_names) <= len(fun_names), "plan name enumeration mismatch"
        return cls(
            class_antrs=class_antrs,
            fun_names=fun_names,
            ambiguous_method_names=ambiguous_method_names,
            plan_names=plan_names,
        )

