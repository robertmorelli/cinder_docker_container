from ast import (
    parse,
    dump as ast_dump,
    unparse,
    FunctionDef,
    ClassDef,
    AnnAssign,
    Assign,
    expr,
    Name,
    iter_child_nodes,
    AST,
    fix_missing_locations,
    NodeTransformer,
    Import,
    ImportFrom,
    Constant,
    Call,
    Attribute,
    Load,
    Store,
    alias,
    Return,
    Starred,
    Subscript,
)
from functools import cache, reduce
from itertools import chain
from random import sample
from subprocess import run, CompletedProcess
from time import perf_counter
from multiprocessing import Pool, cpu_count
from datetime import datetime
from json import dump
from builtins import __dict__ as btn_dict
from multiprocessing import Value, Lock
from time import sleep
from os import path, makedirs, environ
from copy import deepcopy
from hashlib import sha1
from typing import Tuple
from argparse import ArgumentParser

try:
    import black
except Exception:
    black = None


def split_lines(src: str) -> str:
    if black is None:
        return src
    return black.format_str(src, mode=black.Mode(line_length=1))


Permutation = Tuple[bool, ...]
GuideKey = tuple[str | None, str | None]
PlanKey = GuideKey
PassState = dict[str, bool]
GuideType = dict[GuideKey, PassState]
QNameType = set[GuideKey]

PASS_SCOPE_PARAM = "param"
PASS_SCOPE_BODY = "body"
PASS_SCOPE_RETURN = "return"
PASS_SCOPES = (PASS_SCOPE_PARAM, PASS_SCOPE_BODY, PASS_SCOPE_RETURN)

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

explicit_cast_type_order = (
)
explicit_cast_types = frozenset(explicit_cast_type_order)

scalar_construct_type_order = (
    "float",
)
scalar_construct_types = frozenset(scalar_construct_type_order)

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

container_construct_type_order = (
    "CheckedList",
    "CheckedDict",
    "CheckedSet",
)
container_passthrough_type_order = (
    "Array",
    "Dict",
    "List",
    "Set",
    "Tuple",
)
container_construct_types = frozenset(container_construct_type_order)
container_passthrough_types = frozenset(container_passthrough_type_order)
construct_container_types = container_construct_types.union(container_passthrough_types)

CAST_FALLBACK_BIN = "other"


def _sanitize_pass_token(name: str) -> str:
    out = []
    previous_underscore = False
    for ch in name:
        if ch.isalnum():
            out.append(ch.lower())
            previous_underscore = False
        else:
            if not previous_underscore:
                out.append("_")
                previous_underscore = True
    token = "".join(out).strip("_")
    assert token != "", f"invalid pass token: {name!r}"
    return token


BOX_LOGIC_BIN = "primitive"
CONSTRUCT_LOGIC_BIN = "container"
CAST_LOGIC_BIN = "all"
construct_bin_by_root = dict((root_name, CONSTRUCT_LOGIC_BIN) for root_name in construct_container_types)

TYPE_BIN_SPECS: tuple[tuple[str, str], ...] = (
    ("box", BOX_LOGIC_BIN),
    ("construct", CONSTRUCT_LOGIC_BIN),
    ("cast", CAST_LOGIC_BIN),
)

PASS_SPECS: tuple[tuple[str, str, str, str], ...] = tuple(
    (
        scope_name,
        policy_name,
        bin_name,
        f"{scope_name}_{policy_name}_{bin_name}",
    )
    for scope_name in PASS_SCOPES
    for policy_name, bin_name in TYPE_BIN_SPECS
)
PASS_NAMES: tuple[str, ...] = tuple(pass_name for _, _, _, pass_name in PASS_SPECS)
PASS_NAME_LOOKUP: dict[tuple[str, str, str], str] = dict(
    ((scope_name, policy_name, bin_name), pass_name)
    for scope_name, policy_name, bin_name, pass_name in PASS_SPECS
)
PASS_NAMES_BY_SCOPE: dict[str, tuple[str, ...]] = dict(
    (
        scope_name,
        tuple(pass_name for pass_scope, _, _, pass_name in PASS_SPECS if pass_scope == scope_name),
    )
    for scope_name in PASS_SCOPES
)
PASS_COUNT = len(PASS_NAMES)


def is_builtin_class_name(name: str):
    return name in btn_dict and isinstance(btn_dict[name], type)


TIME = False
TOP_LEVEL = (None, None)


class Timer:
    def __init__(self, name):
        self.name = name
        self.start = perf_counter() if TIME else 0

    def end(self):
        if TIME:
            dt = perf_counter() - self.start
            print(f"{self.name}: {dt:.6f}s")


class PermutationLocker:
    _g_counter = None
    _g_lock = None

    @staticmethod
    def _init_worker(counter, lock):
        PermutationLocker._g_counter = counter
        PermutationLocker._g_lock = lock

    @staticmethod
    def run_perm_worker(args):
        me, perm = args
        stderr = me.run_permutation(perm)
        with PermutationLocker._g_lock:
            PermutationLocker._g_counter.value += 1
        return CinderDetyperBoxUnbox._perm_name(perm), stderr


class CinderDetyperBoxUnbox:
    permutation_to_result = dict()

    def __init__(
        self,
        benchmark_file_name: str,
        python: str,
        scratch_dir: str,
        params: tuple[str, ...] = (),
    ):
        t1 = Timer("init")
        self.params = params
        self.python = python
        self.benchmark_file_name = benchmark_file_name
        self.scratch_dir = scratch_dir
        self.benchmark_dir = path.dirname(path.abspath(benchmark_file_name))

        t2 = Timer("enumerate")
        self.class_antrs, self.fun_names = self._enumerate_funs()
        self.plan_names = self._enumerate_plan_names()
        assert len(self.plan_names) == len(self.fun_names), "plan name enumeration mismatch"
        t2.end()
        t1.end()

    def fun_count(self):
        return len(self.fun_names)

    def pass_count(self):
        return PASS_COUNT

    def plan_bit_count(self):
        return len(self.plan_names)

    def _enumerate_plan_names(self) -> tuple[PlanKey, ...]:
        # keep de_typer-style function-level planning: one bit per qualified function
        return self.fun_names

    @staticmethod
    @cache
    def read_text(file_name):
        with open(file_name, encoding="utf-8") as f:
            return f.read()
        raise RuntimeError("python file not found")

    @cache
    def _read_ast(self):
        return parse(CinderDetyperBoxUnbox.read_text(self.benchmark_file_name), type_comments=True)

    def _enumerate_funs(self):
        ordered_q_names = [TOP_LEVEL]
        q_fun_names_set: QNameType = {TOP_LEVEL}

        antr_graph: dict[None | str, frozenset[str]] = {None: frozenset()}

        def antr_classes(anter_exprs: list[expr]) -> frozenset[str]:
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
            antr_graph[class_name] = frozenset(antr_classes(class_node.bases))

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

        fun_count = sum(count_gen(self._read_ast())) + 1
        assert fun_count == len(ordered_q_names), "function counting failure"
        return antr_graph, tuple(ordered_q_names)

    def _detype_funs(self, tree: AST, guide: GuideType) -> str:
        @cache
        def get_fun_q_names(fun_name: str, antr_name=None):
            assert antr_name in self.class_antrs, "class ancestor graph incomplete"
            anters = self.class_antrs[antr_name]
            anter_fun_names = tuple(
                q_name for q_name in zip(anters, (fun_name,) * len(anters)) if q_name in self.fun_names
            )
            return ((antr_name, fun_name),) + anter_fun_names

        @cache
        def fun_q_names_by_method_name(fun_name: str) -> tuple[GuideKey, ...]:
            return tuple(q_fun_name for q_fun_name in self.fun_names if q_fun_name[1] == fun_name)

        def get_fun_q_name(fun_name: str, antr_name=None) -> GuideKey:
            q_fun_name, *alternate_names = get_fun_q_names(fun_name, antr_name=antr_name)
            assert len(alternate_names) <= 1, f"function identity failure: one of {alternate_names}"
            return alternate_names[0] if alternate_names else q_fun_name

        def get_fun_pass_state(fun_name: str, antr_name: str | None = None) -> PassState:
            q_fun_name: GuideKey = get_fun_q_name(fun_name, antr_name=antr_name)
            assert q_fun_name in guide, "function type status unspecified"
            return guide[q_fun_name]

        def fun_has_any_pass_enabled(fun_name: str, antr_name: str | None = None):
            pass_state = get_fun_pass_state(fun_name, antr_name=antr_name)
            return any(pass_state[pass_name] for pass_name in PASS_NAMES)

        def is_none_annotation(annotation: expr | None) -> bool:
            if annotation is None:
                return True
            if isinstance(annotation, Constant):
                return annotation.value is None
            if isinstance(annotation, Name):
                return annotation.id == "None"
            return False

        def is_primitive_annotation(annotation: expr | None) -> bool:
            return isinstance(annotation, Name) and annotation.id in box_primitive_types

        def is_dynamic_annotation(annotation: expr | None) -> bool:
            return isinstance(annotation, Name) and annotation.id == "dynamic"

        def is_box_call(node: expr) -> bool:
            return isinstance(node, Call) and isinstance(node.func, Name) and node.func.id == "box"

        def wrap_box(node: expr):
            needed_imports.add("box")
            return Call(func=Name(id="box", ctx=Load()), args=[node], keywords=[])

        def wrap_cast(annotation: expr, node: expr):
            needed_imports.add("cast")
            return Call(
                func=Name(id="cast", ctx=Load()),
                args=[deepcopy(annotation), node],
                keywords=[],
            )

        def wrap_scalar_construct(annotation: expr, node: expr):
            assert isinstance(annotation, Name), "scalar construct annotation must be simple name"
            return Call(
                func=Name(id=annotation.id, ctx=Load()),
                args=[node],
                keywords=[],
            )

        def wrap_cast_or_construct(annotation: expr, node: expr):
            if isinstance(annotation, Name) and annotation.id in scalar_construct_types:
                return wrap_scalar_construct(annotation, node)
            return wrap_cast(annotation, node)

        def is_constructor_annotation(annotation: expr | None) -> bool:
            return (
                isinstance(annotation, Subscript)
                and isinstance(annotation.value, Name)
                and annotation.value.id in construct_container_types
            )

        def is_passthrough_container_annotation(annotation: expr | None) -> bool:
            return (
                isinstance(annotation, Subscript)
                and isinstance(annotation.value, Name)
                and annotation.value.id in container_passthrough_types
            )

        def annotation_root_name(annotation: expr | None) -> str | None:
            if annotation is None:
                return None
            if isinstance(annotation, Name):
                return annotation.id
            if isinstance(annotation, Subscript):
                return annotation_root_name(annotation.value)
            if isinstance(annotation, Attribute):
                return annotation.attr
            return None

        def annotation_detype_info(annotation: expr | None) -> tuple[str, str | None]:
            if annotation is None or is_none_annotation(annotation) or is_dynamic_annotation(annotation):
                return ("passthrough", None)
            root_name = annotation_root_name(annotation)
            if root_name in nogo_types:
                return ("passthrough", None)
            if isinstance(annotation, Name) and annotation.id in explicit_cast_types:
                return ("cast", CAST_LOGIC_BIN)
            if is_primitive_annotation(annotation):
                assert isinstance(annotation, Name), "primitive annotations must be simple names"
                return ("box", BOX_LOGIC_BIN)
            if is_constructor_annotation(annotation):
                assert isinstance(annotation, Subscript), "constructor annotation must be subscript"
                assert isinstance(annotation.value, Name), "constructor annotation base must be name"
                root_name = annotation.value.id
                assert root_name in construct_bin_by_root, f"unknown constructor annotation root: {root_name}"
                return ("construct", construct_bin_by_root[root_name])
            return ("cast", CAST_LOGIC_BIN)

        def annotation_policy(annotation: expr | None) -> str:
            policy, _ = annotation_detype_info(annotation)
            return policy

        def pass_name_for_annotation(scope_name: str, annotation: expr | None) -> str | None:
            policy, bin_name = annotation_detype_info(annotation)
            if policy == "passthrough":
                return None
            assert bin_name is not None, f"missing type bin for non-passthrough annotation: {annotation!r}"
            key = (scope_name, policy, bin_name)
            assert key in PASS_NAME_LOOKUP, f"pass mapping missing for {key}"
            return PASS_NAME_LOOKUP[key]

        def wrap_construct(annotation: expr, node: expr):
            assert is_constructor_annotation(annotation), f"expected constructor annotation, got: {annotation}"
            return Call(
                func=deepcopy(annotation),
                args=[node],
                keywords=[],
            )

        def coerce_primitive(annotation: expr | None, node: expr):
            if not is_primitive_annotation(annotation):
                return node
            return Call(
                func=Name(id=annotation.id, ctx=Load()),
                args=[node],
                keywords=[],
            )

        def fresh_hidden_name(name: str, used: set[str]) -> str:
            new_name = f"_{name}"
            while new_name in used:
                new_name = f"_{new_name}"
            used.add(new_name)
            return new_name

        def is_inline_function(fn: FunctionDef) -> bool:
            for decorator in fn.decorator_list:
                if isinstance(decorator, Name) and decorator.id == "inline":
                    return True
                if isinstance(decorator, Attribute) and decorator.attr == "inline":
                    return True
            return False

        def make_inbound_conversion(annotation: expr, source_name: str):
            source = Name(id=source_name, ctx=Load())
            policy = annotation_policy(annotation)
            if policy == "box":
                return coerce_primitive(annotation, source)
            if policy == "construct":
                if is_passthrough_container_annotation(annotation):
                    return source
                return wrap_construct(annotation, source)
            if policy == "passthrough":
                return source
            return wrap_cast_or_construct(annotation, source)

        def detype_annassign_value(annotation: expr, value: expr | None):
            if value is None:
                return Constant(value=None)
            policy = annotation_policy(annotation)
            if policy == "box":
                return wrap_box(coerce_primitive(annotation, value))
            if policy == "construct":
                if is_passthrough_container_annotation(annotation):
                    return value
                return wrap_construct(annotation, value)
            if policy == "passthrough":
                return value
            return wrap_cast_or_construct(annotation, value)

        def ensure_static_imports(module: AST, names: set[str]):
            if len(names) == 0:
                return
            names = set(names)
            for child in module.body:
                if isinstance(child, ImportFrom) and child.module == "__static__":
                    existing = set(alias_node.name for alias_node in child.names)
                    for import_name in sorted(names):
                        if import_name not in existing:
                            child.names.append(alias(name=import_name, asname=None))
                    return

            insert_pos = 0
            for i, child in enumerate(module.body):
                if isinstance(child, (Import, ImportFrom)):
                    insert_pos = i + 1
            module.body.insert(
                insert_pos,
                ImportFrom(
                    module="__static__",
                    names=[alias(name=import_name, asname=None) for import_name in sorted(names)],
                    level=0,
                ),
            )

        def _detype_params_for_policy(
            fn: FunctionDef,
            used_names: set[str],
            conversion_stmts: list[AnnAssign],
            pass_state: PassState,
            expected_policy: str,
        ):
            changed = False
            for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs):
                annotation = a.annotation
                if annotation is None:
                    continue
                if a.arg in ("self", "cls"):
                    continue
                policy = annotation_policy(annotation)
                if policy != expected_policy:
                    continue
                pass_name = pass_name_for_annotation(PASS_SCOPE_PARAM, annotation)
                assert pass_name is not None, f"missing param pass for annotation: {annotation!r}"
                if not pass_state[pass_name]:
                    continue

                old_name = a.arg
                hidden_name = fresh_hidden_name(old_name, used_names)
                a.arg = hidden_name
                annotation_copy = deepcopy(annotation)
                conversion_stmts.append(
                    AnnAssign(
                        target=Name(id=old_name, ctx=Store()),
                        annotation=annotation_copy,
                        value=make_inbound_conversion(annotation_copy, hidden_name),
                        simple=1,
                    )
                )
                a.annotation = None
                changed = True
            return changed

        def _detype_params(fn: FunctionDef, pass_state: PassState):
            used_names = set(a.arg for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs))
            conversion_stmts: list[AnnAssign] = []
            changed = False
            changed = _detype_params_for_policy(fn, used_names, conversion_stmts, pass_state, "box") or changed
            changed = _detype_params_for_policy(fn, used_names, conversion_stmts, pass_state, "construct") or changed
            changed = _detype_params_for_policy(fn, used_names, conversion_stmts, pass_state, "cast") or changed
            return changed, conversion_stmts

        def _detype_body_for_policy(annotation: expr, value: expr | None, pass_state: PassState, expected_policy: str):
            policy = annotation_policy(annotation)
            assert policy == expected_policy, f"{expected_policy} body detype policy mismatch"
            pass_name = pass_name_for_annotation(PASS_SCOPE_BODY, annotation)
            assert pass_name is not None, f"missing body pass for annotation: {annotation!r}"
            if not pass_state[pass_name]:
                return None
            return detype_annassign_value(annotation, value)

        def _detype_body_primitive(annotation: expr, value: expr | None, pass_state: PassState):
            return _detype_body_for_policy(annotation, value, pass_state, "box")

        def _detype_body_construct(annotation: expr, value: expr | None, pass_state: PassState):
            return _detype_body_for_policy(annotation, value, pass_state, "construct")

        def _detype_body_cast(annotation: expr, value: expr | None, pass_state: PassState):
            return _detype_body_for_policy(annotation, value, pass_state, "cast")

        def _detype_return_for_policy(fn: FunctionDef, pass_state: PassState, expected_policy: str):
            if annotation_policy(fn.returns) != expected_policy:
                return False
            pass_name = pass_name_for_annotation(PASS_SCOPE_RETURN, fn.returns)
            assert pass_name is not None, f"missing return pass for annotation: {fn.returns!r}"
            if not pass_state[pass_name]:
                return False
            fn.returns = None
            return True

        def _detype_return(fn: FunctionDef, pass_state: PassState):
            changed = False
            changed = _detype_return_for_policy(fn, pass_state, "box") or changed
            changed = _detype_return_for_policy(fn, pass_state, "construct") or changed
            changed = _detype_return_for_policy(fn, pass_state, "cast") or changed
            return changed

        def pull_param_info(fn: FunctionDef, skip_implicit_first: bool, pass_state: PassState):
            args = fn.args
            positional = tuple(chain(args.posonlyargs, args.args))
            typed_pos: dict[int, expr] = {}
            blocked_pos: set[int] = set()
            call_index = 0
            for i, a in enumerate(positional):
                is_implicit = skip_implicit_first and i == 0 and a.arg in ("self", "cls")
                if is_implicit:
                    continue
                annotation = getattr(a, "annotation", None)
                pass_name = pass_name_for_annotation(PASS_SCOPE_PARAM, annotation)
                if pass_name is not None and pass_state[pass_name]:
                    typed_pos[call_index] = deepcopy(annotation)
                call_index += 1
            typed_kw: dict[str, expr] = {}
            blocked_kw: set[str] = set()
            for i, a in enumerate(chain(args.args, args.kwonlyargs)):
                is_implicit = skip_implicit_first and i == 0 and a.arg in ("self", "cls")
                if is_implicit:
                    continue
                annotation = getattr(a, "annotation", None)
                pass_name = pass_name_for_annotation(PASS_SCOPE_PARAM, annotation)
                if pass_name is not None and pass_state[pass_name]:
                    typed_kw[a.arg] = deepcopy(annotation)
            ret_ann = None
            ret_pass_name = pass_name_for_annotation(PASS_SCOPE_RETURN, fn.returns)
            if ret_pass_name is not None and pass_state[ret_pass_name]:
                ret_ann = None if is_none_annotation(fn.returns) else deepcopy(fn.returns)
            return typed_pos, blocked_pos, typed_kw, blocked_kw, ret_ann

        def merge_call_info(
            table: dict[GuideKey, dict], q_fun_name: GuideKey, fn: FunctionDef, skip_implicit_first: bool, pass_state: PassState
        ):
            typed_pos, blocked_pos, typed_kw, blocked_kw, ret_ann = pull_param_info(fn, skip_implicit_first, pass_state)
            if q_fun_name not in table:
                table[q_fun_name] = {
                    "typed_pos": {},
                    "blocked_pos": set(),
                    "typed_kw": {},
                    "blocked_kw": set(),
                    "ret_ann": None,
                    "ret_conflict": False,
                    "ret_seen_none": False,
                }

            entry = table[q_fun_name]

            for i in blocked_pos:
                entry["blocked_pos"].add(i)
                entry["typed_pos"].pop(i, None)
            for i, annotation in typed_pos.items():
                if i in entry["blocked_pos"]:
                    continue
                existing = entry["typed_pos"].get(i)
                if existing is None:
                    entry["typed_pos"][i] = annotation
                elif ast_dump(existing, include_attributes=False) != ast_dump(annotation, include_attributes=False):
                    entry["blocked_pos"].add(i)
                    entry["typed_pos"].pop(i, None)

            for name in blocked_kw:
                entry["blocked_kw"].add(name)
                entry["typed_kw"].pop(name, None)
            for name, annotation in typed_kw.items():
                if name in entry["blocked_kw"]:
                    continue
                existing = entry["typed_kw"].get(name)
                if existing is None:
                    entry["typed_kw"][name] = annotation
                elif ast_dump(existing, include_attributes=False) != ast_dump(annotation, include_attributes=False):
                    entry["blocked_kw"].add(name)
                    entry["typed_kw"].pop(name, None)

            if ret_ann is None:
                entry["ret_seen_none"] = True
                if entry["ret_ann"] is not None:
                    entry["ret_ann"] = None
                    entry["ret_conflict"] = True
            elif not entry["ret_conflict"]:
                if entry["ret_seen_none"]:
                    entry["ret_ann"] = None
                    entry["ret_conflict"] = True
                elif entry["ret_ann"] is None:
                    entry["ret_ann"] = ret_ann
                elif ast_dump(entry["ret_ann"], include_attributes=False) != ast_dump(ret_ann, include_attributes=False):
                    entry["ret_ann"] = None
                    entry["ret_conflict"] = True

        def consistent_method_return_annotations(table: dict[GuideKey, dict]):
            out: dict[str, expr] = {}
            blocked: set[str] = set()
            for q_fun_name, entry in table.items():
                antr_name, fun_name = q_fun_name
                if antr_name is None:
                    continue
                if fun_name in blocked:
                    continue
                ret_ann = entry["ret_ann"]
                if ret_ann is None:
                    blocked.add(fun_name)
                    out.pop(fun_name, None)
                    continue
                existing = out.get(fun_name)
                if existing is None:
                    out[fun_name] = deepcopy(ret_ann)
                    continue
                if ast_dump(existing, include_attributes=False) != ast_dump(ret_ann, include_attributes=False):
                    blocked.add(fun_name)
                    out.pop(fun_name, None)
            return out

        def collect_class_field_annotations(module_node: AST):
            field_map: dict[str, dict[str, expr]] = {}

            class FieldCollector(NodeTransformer):
                def __init__(self):
                    self.class_stack: list[str] = []

                def visit_ClassDef(self, node: ClassDef):
                    self.class_stack.append(node.name)
                    self.generic_visit(node)
                    self.class_stack.pop()
                    return node

                def visit_AnnAssign(self, node: AnnAssign):
                    if len(self.class_stack) == 0:
                        return node
                    target = node.target
                    if isinstance(target, Attribute) and isinstance(target.value, Name) and target.value.id == "self":
                        class_name = self.class_stack[-1]
                        field_name = target.attr
                        annotation = node.annotation
                        assert annotation is not None, "field annotation missing"
                        class_fields = field_map.setdefault(class_name, {})
                        existing = class_fields.get(field_name)
                        if existing is None:
                            class_fields[field_name] = deepcopy(annotation)
                        else:
                            assert (
                                ast_dump(existing, include_attributes=False)
                                == ast_dump(annotation, include_attributes=False)
                            ), f"conflicting field annotation for {class_name}.{field_name}"
                    return node

            FieldCollector().visit(deepcopy(module_node))
            return field_map

        class BodyDetyper(NodeTransformer):
            def __init__(
                self,
                pass_state: PassState,
                return_annotation: expr | None,
                class_name: str | None,
                field_annotations: dict[str, dict[str, expr]],
            ):
                self.pass_state = pass_state
                self.return_annotation = return_annotation
                self.class_name = class_name
                self.field_annotations = field_annotations
                self.body_variant_enabled = any(pass_state[pass_name] for pass_name in PASS_NAMES_BY_SCOPE[PASS_SCOPE_BODY])
                return_pass_name = pass_name_for_annotation(PASS_SCOPE_RETURN, return_annotation)
                self.box_returns = (
                    return_pass_name is not None
                    and pass_state[return_pass_name]
                    and annotation_policy(return_annotation) == "box"
                )
                self.primitive_reproject_annotations: dict[str, expr] = {}

            def _field_annotation(self, node: Attribute) -> expr | None:
                if self.class_name is None:
                    return None
                if not isinstance(node.value, Name):
                    return None
                if node.value.id != "self":
                    return None
                class_fields = self.field_annotations.get(self.class_name)
                if class_fields is None:
                    return None
                annotation = class_fields.get(node.attr)
                return deepcopy(annotation) if annotation is not None else None

            def _field_pass_enabled(self, annotation: expr | None):
                pass_name = pass_name_for_annotation(PASS_SCOPE_BODY, annotation)
                return pass_name is not None and self.pass_state[pass_name]

            def _embed_field_write(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    coerced = coerce_primitive(annotation, node)
                    if is_box_call(coerced):
                        return coerced
                    return wrap_box(coerced)
                if policy == "construct":
                    if is_passthrough_container_annotation(annotation):
                        return node
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def _project_field_read(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    return coerce_primitive(annotation, node)
                if policy == "construct":
                    if is_passthrough_container_annotation(annotation):
                        return node
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def visit_Assign(self, node: Assign):
                self.generic_visit(node)
                if len(node.targets) == 1 and isinstance(node.targets[0], Name):
                    target_name = node.targets[0].id
                    if target_name in self.primitive_reproject_annotations:
                        annotation = deepcopy(self.primitive_reproject_annotations[target_name])
                        value = node.value
                        assert value is not None, "assign without value unsupported"
                        coerced = coerce_primitive(annotation, value)
                        if not is_box_call(coerced):
                            node.value = wrap_box(coerced)
                        else:
                            node.value = coerced
                if len(node.targets) == 1 and isinstance(node.targets[0], Attribute):
                    field_ann = self._field_annotation(node.targets[0])
                    if field_ann is not None and self._field_pass_enabled(field_ann):
                        value = node.value
                        assert value is not None, "field assign without value unsupported"
                        node.value = self._embed_field_write(value, field_ann)
                node.type_comment = None
                return node

            def visit_Name(self, node: Name):
                # Primitive body detype can make a typed local dynamic; reproject on use.
                if isinstance(node.ctx, Load) and node.id in self.primitive_reproject_annotations:
                    return coerce_primitive(deepcopy(self.primitive_reproject_annotations[node.id]), node)
                return node

            def visit_Attribute(self, node: Attribute):
                self.generic_visit(node)
                if not isinstance(node.ctx, Load):
                    return node
                field_ann = self._field_annotation(node)
                if field_ann is None:
                    return node
                if not self._field_pass_enabled(field_ann):
                    return node
                return self._project_field_read(node, field_ann)

            def visit_AnnAssign(self, node: AnnAssign):
                self.generic_visit(node)
                annotation = node.annotation
                assert annotation is not None, "annassign without annotation unsupported"
                policy = annotation_policy(annotation)
                if policy == "passthrough":
                    return node
                if not self.body_variant_enabled:
                    return node

                if policy == "box":
                    value = _detype_body_primitive(annotation, node.value, self.pass_state)
                elif policy == "construct":
                    value = _detype_body_construct(annotation, node.value, self.pass_state)
                elif policy == "cast":
                    value = _detype_body_cast(annotation, node.value, self.pass_state)
                else:
                    assert False, f"unknown body detype policy: {policy}"

                if value is None:
                    return node
                if isinstance(node.target, Attribute):
                    field_ann = self._field_annotation(node.target)
                    if field_ann is not None:
                        assert (
                            ast_dump(field_ann, include_attributes=False)
                            == ast_dump(annotation, include_attributes=False)
                        ), f"field annotation mismatch on {self.class_name}.{node.target.attr}"
                        # Preserve class field declaration semantics; field read/write handling applies separately.
                        return node
                if policy == "box" and isinstance(node.target, Name) and node.value is not None:
                    self.primitive_reproject_annotations[node.target.id] = deepcopy(annotation)
                return Assign(targets=[node.target], value=value, type_comment=None)

            def visit_Return(self, node: Return):
                self.generic_visit(node)
                if self.box_returns and node.value is not None and not is_box_call(node.value):
                    node.value = wrap_box(coerce_primitive(self.return_annotation, node.value))
                return node

        class InlineArgInliner(NodeTransformer):
            def __init__(self, replacements: dict[str, expr]):
                self.replacements = replacements

            def visit_Name(self, node: Name):
                if isinstance(node.ctx, Load) and node.id in self.replacements:
                    return deepcopy(self.replacements[node.id])
                return node

            def visit_FunctionDef(self, node: FunctionDef):
                return node

            def visit_ClassDef(self, node: ClassDef):
                return node

        def cleanup_inline_function(fn: FunctionDef):
            if not is_inline_function(fn):
                return True
            if len(fn.body) == 0:
                return True

            replacement_map = {}
            body_index = 0
            while body_index < len(fn.body):
                stmt = fn.body[body_index]
                if not isinstance(stmt, AnnAssign):
                    break
                if not isinstance(stmt.target, Name):
                    break
                if stmt.value is None:
                    break
                replacement_map[stmt.target.id] = deepcopy(stmt.value)
                body_index += 1

            if body_index == 0:
                return len(fn.body) == 1 and isinstance(fn.body[0], Return)
            if body_index >= len(fn.body):
                return False
            if not isinstance(fn.body[body_index], Return):
                return False
            if len(fn.body) != body_index + 1:
                return False

            inliner = InlineArgInliner(replacement_map)
            cleaned_return = inliner.visit(fn.body[body_index])
            fn.body = [cleaned_return]
            return True

        class BoundaryCallRetyper(NodeTransformer):
            def __init__(self, call_info, class_names, method_ret_fallback):
                self.call_info = call_info
                self.class_names = class_names
                self.method_ret_fallback = method_ret_fallback
                self.class_stack: list[str] = []

            def visit_ClassDef(self, node: ClassDef):
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()
                return node

            def _embed_boundary_arg(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    if is_box_call(node):
                        return node
                    return wrap_box(coerce_primitive(annotation, node))
                if policy == "construct":
                    if is_passthrough_container_annotation(annotation):
                        return node
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def _project_boundary_return(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    return coerce_primitive(annotation, node)
                if policy == "construct":
                    if is_passthrough_container_annotation(annotation):
                        return node
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def _wrap_return(self, node: Call, annotation: expr | None):
                return self._project_boundary_return(node, annotation)

            def _is_unbound_class_method_call(self, node: Call) -> bool:
                if not isinstance(node.func, Attribute):
                    return False
                if not isinstance(node.func.value, Name):
                    return False
                return node.func.value.id in self.class_names

            def _resolve_method_q_name(self, node: Call):
                assert isinstance(node.func, Attribute), "method resolution on non-attribute call unsupported"
                method_name = node.func.attr
                owner_name = None
                if isinstance(node.func.value, Name):
                    receiver_name = node.func.value.id
                    if receiver_name in self.class_names:
                        owner_name = receiver_name
                    elif receiver_name in ("self", "cls") and len(self.class_stack) > 0:
                        owner_name = self.class_stack[-1]
                if owner_name is None:
                    matching_q_names = fun_q_names_by_method_name(method_name)
                    if len(matching_q_names) == 1:
                        return matching_q_names[0]
                    return None
                return get_fun_q_name(method_name, owner_name)

            def visit_Call(self, node: Call):
                self.generic_visit(node)
                info = None
                unbound_class_method = False
                if isinstance(node.func, Name):
                    if node.func.id in self.class_names:
                        q_fun_name = get_fun_q_name("__init__", node.func.id)
                        info = self.call_info.get(q_fun_name)
                    else:
                        q_fun_name = get_fun_q_name(node.func.id, None)
                        info = self.call_info.get(q_fun_name)
                elif isinstance(node.func, Attribute):
                    unbound_class_method = self._is_unbound_class_method_call(node)
                    q_fun_name = self._resolve_method_q_name(node)
                    if q_fun_name is None:
                        ret_fallback = self.method_ret_fallback.get(node.func.attr)
                        if ret_fallback is not None:
                            return self._wrap_return(node, ret_fallback)
                        return node
                    info = self.call_info.get(q_fun_name)

                if info is None:
                    return node

                typed_pos = info["typed_pos"]
                typed_kw = info["typed_kw"]
                ret_ann = info["ret_ann"]

                needs_arg_rewrite = len(typed_pos) > 0 or len(typed_kw) > 0
                if needs_arg_rewrite:
                    has_star_arg = any(isinstance(arg_node, Starred) for arg_node in node.args)
                    has_star_kw = any(kw.arg is None for kw in node.keywords)
                    assert not has_star_arg, "starred positional args not supported for boundary rewrite"
                    assert not has_star_kw, "starred keyword args not supported for boundary rewrite"

                arg_offset = 1 if unbound_class_method else 0
                if arg_offset == 1:
                    assert len(node.args) > 0, "unbound class method call missing explicit self/cls"

                new_args = []
                for i, arg_node in enumerate(node.args):
                    remapped_i = i - arg_offset
                    if remapped_i in typed_pos:
                        new_args.append(self._embed_boundary_arg(arg_node, typed_pos[remapped_i]))
                    else:
                        new_args.append(arg_node)
                node.args = new_args

                for kw in node.keywords:
                    if kw.arg in typed_kw:
                        kw.value = self._embed_boundary_arg(kw.value, typed_kw[kw.arg])

                return self._wrap_return(node, ret_ann)

        def detype_function(fn: FunctionDef, pass_state: PassState, class_name: str | None):
            return_annotation = deepcopy(fn.returns)
            changed_signature, conversion_stmts = _detype_params(fn, pass_state)

            if fn.args.vararg:
                vararg_ann = fn.args.vararg.annotation
                vararg_pass_name = pass_name_for_annotation(PASS_SCOPE_PARAM, vararg_ann)
                if vararg_pass_name is not None and pass_state[vararg_pass_name]:
                    assert False, f"detyping vararg annotation unsupported: {fn.name}"
            if fn.args.kwarg:
                kwarg_ann = fn.args.kwarg.annotation
                kwarg_pass_name = pass_name_for_annotation(PASS_SCOPE_PARAM, kwarg_ann)
                if kwarg_pass_name is not None and pass_state[kwarg_pass_name]:
                    assert False, f"detyping kwarg annotation unsupported: {fn.name}"

            changed_signature = _detype_return(fn, pass_state) or changed_signature

            if changed_signature:
                fn.type_comment = None

            body_detyper = BodyDetyper(
                pass_state=pass_state,
                return_annotation=return_annotation,
                class_name=class_name,
                field_annotations=class_field_annotations,
            )
            new_body = []
            for stmt in fn.body:
                new_stmt = body_detyper.visit(stmt)
                if new_stmt is None:
                    continue
                if isinstance(new_stmt, list):
                    new_body.extend(new_stmt)
                else:
                    new_body.append(new_stmt)
            fn.body = conversion_stmts + new_body
            assert cleanup_inline_function(fn), f"unsupported inline function body after detype: {fn.name}"

        def detype_top_level_body(module_node: AST):
            new_body = []
            for child in module_node.body:
                if isinstance(child, Assign):
                    child.type_comment = None
                    new_body.append(child)
                elif isinstance(child, AnnAssign):
                    annotation = child.annotation
                    assert annotation is not None, "top-level annassign without annotation unsupported"
                    policy = annotation_policy(annotation)
                    pass_name = pass_name_for_annotation(PASS_SCOPE_BODY, annotation)
                    if policy == "box":
                        if pass_name is not None and guide[TOP_LEVEL][pass_name]:
                            value = _detype_body_primitive(annotation, child.value, guide[TOP_LEVEL])
                            assert value is not None, "top-level primitive body pass enabled but no rewrite"
                            new_body.append(Assign(targets=[child.target], value=value, type_comment=None))
                        else:
                            new_body.append(child)
                    elif policy == "construct":
                        if pass_name is not None and guide[TOP_LEVEL][pass_name]:
                            value = _detype_body_construct(annotation, child.value, guide[TOP_LEVEL])
                            assert value is not None, "top-level construct body pass enabled but no rewrite"
                            new_body.append(Assign(targets=[child.target], value=value, type_comment=None))
                        else:
                            new_body.append(child)
                    elif policy == "cast":
                        if pass_name is not None and guide[TOP_LEVEL][pass_name]:
                            value = _detype_body_cast(annotation, child.value, guide[TOP_LEVEL])
                            assert value is not None, "top-level cast body pass enabled but no rewrite"
                            new_body.append(Assign(targets=[child.target], value=value, type_comment=None))
                        else:
                            new_body.append(child)
                    else:
                        new_body.append(child)
                else:
                    new_body.append(child)
            module_node.body = new_body

        def detype_walker(node: AST, antr_name: str | None = None):
            for child_node in iter_child_nodes(node):
                is_fun = isinstance(child_node, FunctionDef)
                is_class = isinstance(child_node, ClassDef)
                if is_fun:
                    if fun_has_any_pass_enabled(child_node.name, antr_name=antr_name):
                        q_fun_name = get_fun_q_name(child_node.name, antr_name=antr_name)
                        pass_state = guide[q_fun_name]
                        if antr_name is None:
                            merge_call_info(call_info, q_fun_name, child_node, skip_implicit_first=False, pass_state=pass_state)
                        else:
                            merge_call_info(call_info, q_fun_name, child_node, skip_implicit_first=True, pass_state=pass_state)
                        detype_function(child_node, pass_state, antr_name)
                elif is_class:
                    detype_walker(child_node, antr_name=child_node.name)

        call_info: dict[GuideKey, dict] = dict()
        class_field_annotations = collect_class_field_annotations(tree)
        class_names = set(class_name for class_name in self.class_antrs.keys() if class_name is not None)
        needed_imports: set[str] = set()

        detype_walker(tree)
        method_ret_fallback = consistent_method_return_annotations(call_info)
        tree = BoundaryCallRetyper(call_info, class_names, method_ret_fallback).visit(tree)
        if any(guide[TOP_LEVEL][pass_name] for pass_name in PASS_NAMES_BY_SCOPE[PASS_SCOPE_BODY]):
            detype_top_level_body(tree)
        ensure_static_imports(tree, needed_imports)
        fix_missing_locations(tree)
        return split_lines(unparse(tree))

    @cache
    @staticmethod
    def _perm_name(perm: Permutation) -> str:
        return hex(int("".join(str(int(b)) for b in perm), 2))

    @staticmethod
    def _all_pass_state(value: bool) -> PassState:
        return dict((pass_name, value) for pass_name in PASS_NAMES)

    @staticmethod
    def _pass_state_from_enabled_passes(enabled_pass_names: tuple[str, ...] | None) -> PassState:
        if enabled_pass_names is None:
            return CinderDetyperBoxUnbox._all_pass_state(True)
        unknown = tuple(pass_name for pass_name in enabled_pass_names if pass_name not in PASS_NAMES)
        assert len(unknown) == 0, f"unknown pass names: {unknown}"
        enabled_set = set(enabled_pass_names)
        return dict((pass_name, pass_name in enabled_set) for pass_name in PASS_NAMES)

    def _base_guide(self, value: bool) -> GuideType:
        return dict((q_fun_name, CinderDetyperBoxUnbox._all_pass_state(value)) for q_fun_name in self.fun_names)

    def _guide_from_fun_permutation(self, perm: Permutation, enabled_pass_names: tuple[str, ...] | None = None) -> GuideType:
        assert len(perm) == self.fun_count(), f"function permutation length mismatch: {len(perm)} vs {self.fun_count()}"
        disabled_pass_state = CinderDetyperBoxUnbox._all_pass_state(False)
        enabled_pass_state = CinderDetyperBoxUnbox._pass_state_from_enabled_passes(enabled_pass_names)
        return dict(
            (q_fun_name, dict(enabled_pass_state if perm[i] else disabled_pass_state))
            for i, q_fun_name in enumerate(self.fun_names)
        )

    def _guide_from_permutation(self, perm: Permutation, enabled_pass_names: tuple[str, ...] | None = None) -> GuideType:
        assert len(perm) == self.fun_count(), f"permutation length mismatch: {len(perm)} vs {self.fun_count()}"
        return self._guide_from_fun_permutation(perm, enabled_pass_names=enabled_pass_names)

    @cache
    def _perm_from_name(self, perm_str: str) -> Permutation:
        n = int(perm_str, 16)
        bits = bin(n)[2:].ljust(self.fun_count(), "0")
        return tuple(c == "1" for c in bits)

    @cache
    @staticmethod
    def file_trunc(file_name: str):
        trunc_name, *xts = file_name.split(".")
        assert len(xts) < 2, "multiple extensions not supported"
        assert len(xts) > 0, "extensionless files not supported"
        assert xts[0] == "py", "only '.py' files supported"
        return trunc_name

    @cache
    @staticmethod
    def q_file_trunc(perm: Permutation, file_name: str):
        trunc_name = CinderDetyperBoxUnbox.file_trunc(path.basename(file_name))
        bit_string = "".join("1" if b else "0" for b in perm)
        digest = sha1(bit_string.encode("ascii")).hexdigest()[:16]
        perm_string = f"p{len(perm)}_{digest}"
        return f"{trunc_name}_{perm_string}"

    def _detype_by_permutation(self, perm: Permutation, enabled_pass_names: tuple[str, ...] | None = None) -> str:
        tree = deepcopy(self._read_ast())
        guide: GuideType = self._guide_from_permutation(perm, enabled_pass_names=enabled_pass_names)
        return self._detype_funs(tree, guide)

    def perm_file_name(self, perm: Permutation):
        benchmark_name = path.basename(path.dirname(self.benchmark_dir))
        level_name = path.basename(self.benchmark_dir)
        file_stem = CinderDetyperBoxUnbox.q_file_trunc(perm, self.benchmark_file_name)
        return f"{self.scratch_dir}/{benchmark_name}_{level_name}_{file_stem}.py"

    def _ensure_scratch_dir(self):
        makedirs(self.scratch_dir, exist_ok=True)

    def write_permutation_hex(self, perm_str: str):
        self.write_permutation(self._perm_from_name(perm_str))

    def execute_permutation_hex(self, perm_str: str):
        return self.execute_permutation(self._perm_from_name(perm_str))

    def run_permutation_hex(self, perm_str: str):
        return self.run_permutation(self._perm_from_name(perm_str))

    def write_permutation(self, perm: Permutation, enabled_pass_names: tuple[str, ...] | None = None):
        self._ensure_scratch_dir()
        t1 = Timer("retype")
        new_file_string = self._detype_by_permutation(perm, enabled_pass_names=enabled_pass_names)
        new_file_name = self.perm_file_name(perm)
        t1.end()
        t2 = Timer("write file")
        with open(new_file_name, mode="w", encoding="utf-8") as f:
            f.write(new_file_string)
        t2.end()

    def _run_env(self):
        env = dict(environ)
        py_paths = [self.benchmark_dir]
        current = env.get("PYTHONPATH")
        if current:
            py_paths.append(current)
        env["PYTHONPATH"] = ":".join(py_paths)
        return env

    def execute_typecheck_permutation(self, perm: Permutation):
        file_name = self.perm_file_name(perm)
        t3 = Timer("run typecheck")
        cmd = " ".join(
            (
                self.python,
                "-m cinderx.compiler",
                "--static",
                "-c",
                file_name,
            )
        )
        result = run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            cwd=self.benchmark_dir,
            env=self._run_env(),
        )
        t3.end()
        return result

    def execute_permutation(self, perm: Permutation):
        file_name = self.perm_file_name(perm)
        t3 = Timer("run cmd")
        cmd = " ".join(
            (
                self.python,
                "-X jit",
                "-X jit-enable-jit-list-wildcards",
                "-X jit-shadow-frame",
                file_name,
                *self.params,
            )
        )
        result = run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            cwd=self.benchmark_dir,
            env=self._run_env(),
        )
        t3.end()
        return result

    def run_permutation(self, perm: Permutation, enabled_pass_names: tuple[str, ...] | None = None) -> CompletedProcess[str]:
        self.write_permutation(perm, enabled_pass_names=enabled_pass_names)
        typecheck_result = self.execute_typecheck_permutation(perm)
        if typecheck_result.returncode != 0:
            return typecheck_result
        return self.execute_permutation(perm)

    def _get_prep_perm(self, preportion: float):
        typed_count = self.fun_count() * preportion
        typed_count = round(typed_count)
        untyped_count = self.fun_count() - typed_count
        return tuple(sample((False, True), counts=(typed_count, untyped_count), k=self.fun_count()))

    @staticmethod
    def _collect_failure_stats(results: dict[str, CompletedProcess[str]]):
        counts = {}
        unknown = []
        success_count = 0
        successes = []
        failures = []

        for perm, result in results.items():
            stderr = result.stderr
            if result.returncode == 0:
                success_count += 1
                successes.append(perm)
                continue

            err_name = "<unknown>"
            if stderr:
                lines = stderr.strip().splitlines()
                err_name = lines[-1].split(":")[0] if lines else "<unknown>"
            if err_name == "<unknown>":
                unknown.append(stderr)

            counts[err_name] = counts.get(err_name, 0) + 1
            failures.append(str((perm, err_name)))

        return unknown, counts, success_count, successes, failures

    @staticmethod
    def _make_info_dump(results: dict[str, CompletedProcess[str]]):
        return dict(
            (perm_name, {"returncode": out.returncode, "stdout": out.stdout, "stderr": out.stderr})
            for perm_name, out in results.items()
        )

    def find_permutation_errors(self, samples=2, granularity=1):
        self._ensure_scratch_dir()

        def xp_gen():
            yield self._get_prep_perm(1.0)
            yield self._get_prep_perm(0.0)
            for i in range(1, self.fun_count(), granularity):
                preportion = i / self.fun_count()
                for _ in range(samples):
                    yield self._get_prep_perm(preportion)

        xps = tuple(xp_gen())
        task_args = tuple(reversed(tuple(map(tuple, zip((self,) * len(xps), xps)))))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trunk = CinderDetyperBoxUnbox.file_trunc(path.basename(self.benchmark_file_name))
        results_file_name = f"results_BOXUNBOX_{trunk}_{stamp}_samples_{samples}_granularity_{granularity}.json"

        total = len(task_args)
        counter = Value("i", 0)
        lock = Lock()
        print(f"deploying {total} tasks")
        results = dict()

        with Pool(cpu_count(), initializer=PermutationLocker._init_worker, initargs=(counter, lock)) as pool:
            async_res = pool.map_async(PermutationLocker.run_perm_worker, task_args)
            bar_width = 10
            while not async_res.ready():
                done = counter.value
                frac = done / total
                filled = int(frac * bar_width)
                print(
                    f"\r[{'#' * filled}{'-' * (bar_width - filled)}] {done}/{total} {round(10000 * (done/total)) / 100}%",
                    end="",
                    flush=True,
                )
                sleep(0.1)
            print(f"\r[{'#' * bar_width}] {total}/{total} 100%", end="", flush=True)
            print()
            results = dict(async_res.get())

        unknowns, failure_stats, success_count, successes, failures = CinderDetyperBoxUnbox._collect_failure_stats(results)

        total = len(results)
        failure_count = total - success_count
        print(f"{success_count} successes, {failure_count} failures out of {total}")

        with open(results_file_name, "w", encoding="utf-8") as f:
            dump(
                {
                    "source": self.benchmark_file_name,
                    "success count": success_count,
                    "failure count": failure_count,
                    "failure stats": failure_stats,
                    "successes": successes,
                    "failures": failures,
                    "unknowns": unknowns,
                    "info_dump": CinderDetyperBoxUnbox._make_info_dump(results),
                },
                f,
                indent=2,
            )
        print(f"results in {results_file_name}")

    def get_fully_typed_perm(self):
        return tuple(False for _ in range(self.fun_count()))

    def get_fully_detyped_perm(self):
        return tuple(True for _ in range(self.fun_count()))

    def test_correctness(self):
        typed = self.run_permutation(self.get_fully_typed_perm())
        detyped = self.run_permutation(self.get_fully_detyped_perm())
        return typed, detyped

    def run_pass_only_signals(self):
        fully_detyped = self.get_fully_detyped_perm()
        cases: list[tuple[str, Permutation, tuple[str, ...] | None]] = [
            ("fully_typed", self.get_fully_typed_perm(), None),
            ("fully_detyped", fully_detyped, None),
        ]
        for pass_name in PASS_NAMES:
            cases.append((pass_name, fully_detyped, (pass_name,)))

        out: list[tuple[str, str, int, str]] = []
        for case_name, perm, enabled_pass_names in cases:
            result = self.run_permutation(perm, enabled_pass_names=enabled_pass_names)
            stderr_lines = result.stderr.strip().splitlines()
            detail = stderr_lines[-1] if len(stderr_lines) > 0 else "<no stderr>"
            if enabled_pass_names is None:
                perm_name = CinderDetyperBoxUnbox._perm_name(perm)
            else:
                perm_name = f"{CinderDetyperBoxUnbox._perm_name(perm)}+{enabled_pass_names[0]}"
            out.append((case_name, perm_name, result.returncode, detail))
        return out


def resolve_benchmark_path(benchmark_or_path: str, level: str, root: str):
    if path.isfile(benchmark_or_path):
        return benchmark_or_path
    return f"{root}/{benchmark_or_path}/{level}/main.py"


def main():
    parser = ArgumentParser()
    parser.add_argument("benchmark", help="benchmark name (e.g. deltablue) or path to main.py")
    parser.add_argument("--level", default="advanced", choices=("advanced", "shallow", "untyped"))
    parser.add_argument("--root", default="/root/static-python-perf/Benchmark")
    parser.add_argument("--python", default="/cinder/python")
    parser.add_argument("--scratch", default="/tmp/detype_boxunbox")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--granularity", type=int, default=1)
    parser.add_argument("--param", action="append", default=[], help="benchmark runtime arg (repeatable)")
    parser.add_argument("--show-perm", type=str, default=None, help="hex permutation id to print transformed source")
    parser.add_argument("--test", action="store_true", help="run typed and fully detyped once")
    parser.add_argument("--signals", action="store_true", help="run deterministic pass-only signal report")
    args = parser.parse_args()

    benchmark_path = resolve_benchmark_path(args.benchmark, args.level, args.root)
    if not path.exists(benchmark_path):
        raise FileNotFoundError(f"benchmark file not found: {benchmark_path}")

    detyper = CinderDetyperBoxUnbox(
        benchmark_file_name=benchmark_path,
        python=args.python,
        scratch_dir=args.scratch,
        params=tuple(args.param),
    )

    print(f"benchmark: {benchmark_path}")
    print(f"functions: {detyper.fun_count()}")
    print(f"passes: {PASS_COUNT} ({', '.join(PASS_NAMES)})")
    print(f"plan bits: {detyper.plan_bit_count()}")

    if args.show_perm is not None:
        perm = detyper._perm_from_name(args.show_perm)
        print(detyper._detype_by_permutation(perm))
        return

    if args.test:
        typed, detyped = detyper.test_correctness()
        print(f"typed return code: {typed.returncode}")
        print(f"detyped return code: {detyped.returncode}")
        if typed.returncode != 0:
            print(typed.stderr)
        if detyped.returncode != 0:
            print(detyped.stderr)
        return

    if args.signals:
        for case_name, perm_name, return_code, detail in detyper.run_pass_only_signals():
            state = "ok" if return_code == 0 else "fail"
            print(f"{case_name}: {state} perm={perm_name}")
            if return_code != 0:
                print(f"  {detail}")
        return

    detyper.find_permutation_errors(samples=args.samples, granularity=args.granularity)


if __name__ == "__main__":
    main()
