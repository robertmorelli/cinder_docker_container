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
    Tuple as AstTuple,
    BinOp,
    BitOr,
)
from functools import cache
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
from typing import Tuple as TypingTuple
from argparse import ArgumentParser
from metadata.detype_metadata import DetypeMetadata
from passes import PARAM_PASS_APPLIERS, BODY_PASS_APPLIERS, RETURN_PASS_APPLIERS
from passes.context import PassContext
from passes.cleanup_inline import cleanup_inline_function
from passes.cleanup_wrappers import RedundantWrapperCleaner
from passes.decider import (
    decide_body_strategy,
    decide_param_strategy,
    decide_return_strategy,
    annotation_policy,
    pass_name_for_annotation,
    is_optional_or_union_annotation,
    is_passthrough_container_annotation,
    is_primitive_annotation,
    is_dynamic_annotation,
    is_none_annotation,
    is_constructor_annotation,
)
from passes.bins import (
    box_primitive_type_order,
    box_primitive_types,
    explicit_cast_type_order,
    scalar_construct_type_order,
    scalar_construct_types,
    container_construct_type_order,
    container_passthrough_type_order,
    container_construct_types,
    container_passthrough_types,
    BOX_LOGIC_BIN,
    CONSTRUCT_LOGIC_BIN,
    CAST_LOGIC_BIN,
    CAST_CONTAINER_PASSTHROUGH_LOGIC_BIN,
)

try:
    import black
except Exception:
    black = None


def split_lines(src: str) -> str:
    if black is None:
        return src
    return black.format_str(src, mode=black.Mode(line_length=1))


Permutation = TypingTuple[bool, ...]
GuideKey = tuple[str | None, str | None]
PassState = dict[str, bool]
GuideType = dict[GuideKey, PassState]

PASS_SCOPE_PARAM = "param"
PASS_SCOPE_BODY = "body"
PASS_SCOPE_RETURN = "return"
PASS_SCOPES = (PASS_SCOPE_PARAM, PASS_SCOPE_BODY, PASS_SCOPE_RETURN)

# Known limitations (intentional guardrails for deterministic correctness checks).
TRANSFORM_LIMITATIONS = (
    "class_members: body detyping never rewrites self.<field> declarations, reads, or writes",
    "top_level: module AnnAssign body detyping is always forced typed (plan keeps TOP_LEVEL bit)",
    "decl_only_annassign: body detyping skips declaration-only locals (`x: T`) to avoid dynamic seed / undefined-name edge cases",
    "vararg_param: detyping annotated *args/**kwargs parameters is unsupported",
    "starred_boundary_call: call-boundary rewrite does not support *args/**kwargs expansion at rewritten call sites",
    "body_cast_all: Optional[...] and union annotations are skipped (flow narrowing unsupported)",
    "nogo_types: Iterator/Iterable/Generator/Async*/Coroutine/Protocol/Callable annotations are never detyped",
    "imported_nominal_cast: cast(Name|module.Type, dyn) for imported nominal types is not reliably supported by the static compiler",
    "ambiguous_method_names: unresolved attribute calls are not boundary-rewritten when a method name appears in multiple unrelated class hierarchies",
)

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


TYPE_BIN_SPECS: tuple[tuple[str, str], ...] = (
    ("box", BOX_LOGIC_BIN),
    ("construct", CONSTRUCT_LOGIC_BIN),
    ("cast", CAST_LOGIC_BIN),
    ("cast", CAST_CONTAINER_PASSTHROUGH_LOGIC_BIN),
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
        strict_limitations: bool = False,
    ):
        t1 = Timer("init")
        self.params = params
        self.python = python
        self.benchmark_file_name = benchmark_file_name
        self.scratch_dir = scratch_dir
        self.strict_limitations = strict_limitations
        self.benchmark_dir = path.dirname(path.abspath(benchmark_file_name))

        t2 = Timer("enumerate")
        self.metadata = DetypeMetadata.build(self._read_ast(), TOP_LEVEL)
        self.class_antrs = self.metadata.class_antrs
        self.fun_names = self.metadata.fun_names
        self.ambiguous_method_names = self.metadata.ambiguous_method_names
        self.plan_names = self.metadata.plan_names
        assert len(self.plan_names) <= len(self.fun_names), "plan name enumeration mismatch"
        t2.end()
        t1.end()

    def _strict_fail(self, limitation_name: str, detail: str):
        if not self.strict_limitations:
            return
        assert False, f"strict limitation hit [{limitation_name}]: {detail}"

    def fun_count(self):
        return len(self.fun_names)

    def pass_count(self):
        return PASS_COUNT

    def plan_bit_count(self):
        return len(self.plan_names)

    @staticmethod
    @cache
    def read_text(file_name):
        with open(file_name, encoding="utf-8") as f:
            return f.read()
        raise RuntimeError("python file not found")

    @cache
    def _read_ast(self):
        return parse(CinderDetyperBoxUnbox.read_text(self.benchmark_file_name), type_comments=True)

    def _detype_funs(self, tree: AST, guide: GuideType) -> str:
        def strict_fail(limitation_name: str, detail: str):
            self._strict_fail(limitation_name, detail)

        def is_simple_type_expr(node: expr):
            if isinstance(node, Name):
                return True
            if isinstance(node, Attribute):
                return is_simple_type_expr(node.value)
            if isinstance(node, Subscript):
                return is_simple_type_expr(node.value) and is_simple_type_expr(node.slice)
            if isinstance(node, AstTuple):
                return all(is_simple_type_expr(item) for item in node.elts)
            if isinstance(node, Constant):
                return node.value is None
            return False

        def collect_one_hop_cast_aliases(module_node: AST):
            assert hasattr(module_node, "body"), "module body missing for alias collection"
            aliases: dict[str, expr] = {}
            blocked: set[str] = set()
            for stmt in module_node.body:
                if not (isinstance(stmt, Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], Name)):
                    continue
                alias_name = stmt.targets[0].id
                if alias_name in blocked:
                    continue
                if alias_name in aliases:
                    aliases.pop(alias_name, None)
                    blocked.add(alias_name)
                    continue
                if not isinstance(stmt.value, expr):
                    blocked.add(alias_name)
                    continue
                if not is_simple_type_expr(stmt.value):
                    blocked.add(alias_name)
                    continue
                aliases[alias_name] = deepcopy(stmt.value)
            return aliases

        def collect_imported_symbols(module_node: AST):
            imported_names: set[str] = set()
            imported_module_aliases: set[str] = set()
            assert hasattr(module_node, "body"), "module body missing for import collection"
            for stmt in module_node.body:
                if isinstance(stmt, ImportFrom):
                    for imported in stmt.names:
                        imported_names.add(imported.asname or imported.name)
                elif isinstance(stmt, Import):
                    for imported in stmt.names:
                        imported_module_aliases.add(imported.asname or imported.name.split(".")[0])
            return imported_names, imported_module_aliases

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

        def is_box_call(node: expr) -> bool:
            return isinstance(node, Call) and isinstance(node.func, Name) and node.func.id == "box"

        def wrap_box(node: expr):
            needed_imports.add("box")
            return Call(func=Name(id="box", ctx=Load()), args=[node], keywords=[])

        one_hop_cast_aliases = collect_one_hop_cast_aliases(tree)
        imported_symbols, imported_module_aliases = collect_imported_symbols(tree)

        def wrap_cast(annotation: expr, node: expr):
            def normalize_cast_annotation(annotation_expr: expr):
                if isinstance(annotation_expr, Name) and annotation_expr.id in one_hop_cast_aliases:
                    annotation_expr = deepcopy(one_hop_cast_aliases[annotation_expr.id])
                if isinstance(annotation_expr, Name) and annotation_expr.id in imported_symbols:
                    strict_fail(
                        "imported_nominal_cast",
                        f"cast target uses imported name {annotation_expr.id} in {self.benchmark_file_name}",
                    )
                if (
                    isinstance(annotation_expr, Attribute)
                    and isinstance(annotation_expr.value, Name)
                    and annotation_expr.value.id in imported_module_aliases
                ):
                    strict_fail(
                        "imported_nominal_cast",
                        f"cast target uses imported module attribute {annotation_expr.value.id}.{annotation_expr.attr} in {self.benchmark_file_name}",
                    )
                if isinstance(annotation_expr, AstTuple):
                    return Subscript(
                        value=Name(id="tuple", ctx=Load()),
                        slice=AstTuple(elts=[deepcopy(e) for e in annotation_expr.elts], ctx=Load()),
                        ctx=Load(),
                    )
                return deepcopy(annotation_expr)

            needed_imports.add("cast")
            return Call(
                func=Name(id="cast", ctx=Load()),
                args=[normalize_cast_annotation(annotation), node],
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
            if isinstance(annotation, AstTuple):
                return node
            if isinstance(annotation, Name) and annotation.id in scalar_construct_types:
                return wrap_scalar_construct(annotation, node)
            return wrap_cast(annotation, node)

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

        pass_ctx = PassContext(
            annotation_policy=annotation_policy,
            pass_name_for_annotation=pass_name_for_annotation,
            coerce_primitive=coerce_primitive,
            wrap_construct=wrap_construct,
            wrap_cast_or_construct=wrap_cast_or_construct,
            wrap_box=wrap_box,
            is_passthrough_container_annotation=is_passthrough_container_annotation,
            is_optional_or_union_annotation=is_optional_or_union_annotation,
        )

        def _detype_params(fn: FunctionDef, pass_state: PassState):
            used_names = set(a.arg for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs))
            conversion_stmts: list[AnnAssign] = []
            param_pass_enabled = any(pass_state[pass_name] for pass_name in PASS_NAMES_BY_SCOPE[PASS_SCOPE_PARAM])
            if param_pass_enabled:
                for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs):
                    annotation = getattr(a, "annotation", None)
                    decision = decide_param_strategy(annotation)
                    if decision.strategy == "nogo":
                        strict_fail(
                            "nogo_types",
                            f"param annotation unsupported in {fn.name}.{a.arg}: {ast_dump(annotation, include_attributes=False)}",
                        )
            changed = False
            for pass_apply in PARAM_PASS_APPLIERS:
                changed = pass_apply(fn, pass_state, used_names, conversion_stmts, pass_ctx) or changed
            return changed, conversion_stmts

        def _detype_body(annotation: expr, value: expr | None, pass_state: PassState):
            for pass_apply in BODY_PASS_APPLIERS:
                out = pass_apply(annotation, value, pass_state, pass_ctx)
                if out is not None:
                    return out
            return None

        def _detype_return(fn: FunctionDef, pass_state: PassState):
            return_pass_enabled = any(pass_state[pass_name] for pass_name in PASS_NAMES_BY_SCOPE[PASS_SCOPE_RETURN])
            if return_pass_enabled:
                decision = decide_return_strategy(fn.returns)
                if decision.strategy == "nogo":
                    strict_fail(
                        "nogo_types",
                        f"return annotation unsupported in {fn.name}: {ast_dump(fn.returns, include_attributes=False)}",
                    )
            changed = False
            for pass_apply in RETURN_PASS_APPLIERS:
                changed = pass_apply(fn, pass_state, pass_ctx) or changed
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
                pass_name = decide_param_strategy(annotation).pass_name
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
                pass_name = decide_param_strategy(annotation).pass_name
                if pass_name is not None and pass_state[pass_name]:
                    typed_kw[a.arg] = deepcopy(annotation)
            ret_ann = None
            ret_pass_name = decide_return_strategy(fn.returns).pass_name
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

        def consistent_method_param_annotations(table: dict[GuideKey, dict]):
            methods: dict[str, list[dict]] = {}
            for q_fun_name in self.fun_names:
                antr_name, fun_name = q_fun_name
                if antr_name is None:
                    continue
                methods.setdefault(fun_name, []).append(
                    table.get(
                        q_fun_name,
                        {
                            "typed_pos": {},
                            "blocked_pos": set(),
                            "typed_kw": {},
                            "blocked_kw": set(),
                        },
                    )
                )

            out: dict[str, dict[str, dict]] = {}
            for method_name, entries in methods.items():
                pos_indexes: set[int] = set()
                kw_names: set[str] = set()
                for entry in entries:
                    pos_indexes.update(entry["typed_pos"].keys())
                    pos_indexes.update(entry["blocked_pos"])
                    kw_names.update(entry["typed_kw"].keys())
                    kw_names.update(entry["blocked_kw"])

                typed_pos: dict[int, expr] = {}
                for i in sorted(pos_indexes):
                    if any(i in entry["blocked_pos"] for entry in entries):
                        continue
                    annotations = tuple(entry["typed_pos"].get(i) for entry in entries)
                    if any(annotation is None for annotation in annotations):
                        continue
                    first = annotations[0]
                    assert first is not None, "expected annotation for consistent positional fallback"
                    if all(
                        ast_dump(first, include_attributes=False) == ast_dump(annotation, include_attributes=False)
                        for annotation in annotations[1:]
                    ):
                        typed_pos[i] = deepcopy(first)

                typed_kw: dict[str, expr] = {}
                for name in sorted(kw_names):
                    if any(name in entry["blocked_kw"] for entry in entries):
                        continue
                    annotations = tuple(entry["typed_kw"].get(name) for entry in entries)
                    if any(annotation is None for annotation in annotations):
                        continue
                    first = annotations[0]
                    assert first is not None, "expected annotation for consistent keyword fallback"
                    if all(
                        ast_dump(first, include_attributes=False) == ast_dump(annotation, include_attributes=False)
                        for annotation in annotations[1:]
                    ):
                        typed_kw[name] = deepcopy(first)

                if len(typed_pos) > 0 or len(typed_kw) > 0:
                    out[method_name] = {"typed_pos": typed_pos, "typed_kw": typed_kw}

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
                function_name: str,
                field_annotations: dict[str, dict[str, expr]],
            ):
                self.pass_state = pass_state
                self.return_annotation = return_annotation
                self.class_name = class_name
                self.function_name = function_name
                self.in_init = function_name == "__init__"
                self.field_annotations = field_annotations
                self.body_variant_enabled = any(pass_state[pass_name] for pass_name in PASS_NAMES_BY_SCOPE[PASS_SCOPE_BODY])
                return_decision = decide_return_strategy(return_annotation)
                return_pass_name = return_decision.pass_name
                self.box_returns = (
                    return_pass_name is not None
                    and pass_state[return_pass_name]
                    and return_decision.strategy == "box"
                )
                self.local_reproject_annotations: dict[str, expr] = {}
                self.skip_self_property_detype = True

            def _is_self_attribute(self, node: expr):
                return isinstance(node, Attribute) and isinstance(node.value, Name) and node.value.id == "self"

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
                pass_name = decide_body_strategy(annotation).pass_name
                return pass_name is not None and self.pass_state[pass_name]

            def _embed_field_write(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    coerced = coerce_primitive(annotation, node)
                    if is_box_call(coerced):
                        return coerced
                    return wrap_box(coerced)
                if policy == "construct":
                    assert not is_passthrough_container_annotation(
                        annotation
                    ), "passthrough containers must use cast passthrough pass"
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def _project_field_read(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    return coerce_primitive(annotation, node)
                if policy == "construct":
                    assert not is_passthrough_container_annotation(
                        annotation
                    ), "passthrough containers must use cast passthrough pass"
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def _embed_local_write(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    coerced = coerce_primitive(annotation, node)
                    if is_box_call(coerced):
                        return coerced
                    return wrap_box(coerced)
                if policy == "construct":
                    assert not is_passthrough_container_annotation(
                        annotation
                    ), "passthrough containers must use cast passthrough pass"
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def _project_local_read(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    return coerce_primitive(annotation, node)
                if policy == "construct":
                    assert not is_passthrough_container_annotation(
                        annotation
                    ), "passthrough containers must use cast passthrough pass"
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def visit_Assign(self, node: Assign):
                self.generic_visit(node)
                if len(node.targets) == 1 and isinstance(node.targets[0], Name):
                    target_name = node.targets[0].id
                    if target_name in self.local_reproject_annotations:
                        annotation = deepcopy(self.local_reproject_annotations[target_name])
                        value = node.value
                        assert value is not None, "assign without value unsupported"
                        node.value = self._embed_local_write(value, annotation)
                if len(node.targets) == 1 and isinstance(node.targets[0], Attribute):
                    if self.skip_self_property_detype and self._is_self_attribute(node.targets[0]):
                        if self.body_variant_enabled:
                            strict_fail(
                                "class_members",
                                f"self field assign skipped in {self.function_name} at line {getattr(node, 'lineno', '?')}",
                            )
                        node.type_comment = None
                        return node
                node.type_comment = None
                return node

            def visit_Name(self, node: Name):
                # Body detype can make a typed local dynamic; reproject on use.
                if isinstance(node.ctx, Load) and node.id in self.local_reproject_annotations:
                    return self._project_local_read(node, deepcopy(self.local_reproject_annotations[node.id]))
                return node

            def visit_Attribute(self, node: Attribute):
                self.generic_visit(node)
                if self.skip_self_property_detype and self._is_self_attribute(node):
                    if self.body_variant_enabled:
                        strict_fail(
                            "class_members",
                            f"self field read skipped in {self.function_name} at line {getattr(node, 'lineno', '?')}",
                        )
                    return node
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
                if self.skip_self_property_detype and self._is_self_attribute(node.target):
                    if self.body_variant_enabled:
                        strict_fail(
                            "class_members",
                            f"self field annassign skipped in {self.function_name} at line {getattr(node, 'lineno', '?')}",
                        )
                    return node
                # Declaration-only local annotations are intentionally kept typed.
                # Replacing them with dynamic placeholders (`x = None`) introduces
                # invalid flow for primitive locals and can also change name-binding behavior.
                if node.value is None:
                    if self.body_variant_enabled:
                        strict_fail(
                            "decl_only_annassign",
                            f"declaration-only AnnAssign skipped in {self.function_name} at line {getattr(node, 'lineno', '?')}",
                        )
                    return node
                annotation = node.annotation
                assert annotation is not None, "annassign without annotation unsupported"
                decision = decide_body_strategy(annotation)
                policy = decision.strategy
                pass_name = decision.pass_name
                if policy == "nogo" and self.body_variant_enabled:
                    strict_fail(
                        "nogo_types",
                        f"body annotation unsupported in {self.function_name} at line {getattr(node, 'lineno', '?')}: {ast_dump(annotation, include_attributes=False)}",
                    )
                if policy in ("passthrough", "nogo"):
                    return node
                if not self.body_variant_enabled:
                    return node
                if (
                    pass_name == "body_cast_all"
                    and self.pass_state[pass_name]
                    and is_optional_or_union_annotation(annotation)
                ):
                    strict_fail(
                        "body_cast_all_optional_union",
                        f"optional/union body annotation skipped in {self.function_name} at line {getattr(node, 'lineno', '?')}",
                    )

                if policy in ("box", "construct", "cast"):
                    value = _detype_body(annotation, node.value, self.pass_state)
                else:
                    assert False, f"unknown body detype policy: {policy}"

                if value is None:
                    if pass_name is not None and self.pass_state[pass_name]:
                        strict_fail(
                            "strategy_resolution",
                            f"no body rewrite produced for {self.function_name} at line {getattr(node, 'lineno', '?')}",
                        )
                    return node
                if isinstance(node.target, Name):
                    self.local_reproject_annotations[node.target.id] = deepcopy(annotation)
                return Assign(targets=[node.target], value=value, type_comment=None)

            def visit_Return(self, node: Return):
                self.generic_visit(node)
                if self.box_returns and node.value is not None and not is_box_call(node.value):
                    node.value = wrap_box(coerce_primitive(self.return_annotation, node.value))
                return node

        def collect_primitive_name_annotations(fn: FunctionDef):
            out: dict[str, expr] = {}

            for a in chain(fn.args.posonlyargs, fn.args.args, fn.args.kwonlyargs):
                if is_primitive_annotation(a.annotation):
                    out[a.arg] = deepcopy(a.annotation)

            def walk(node: AST):
                if isinstance(node, AnnAssign):
                    if isinstance(node.target, Name) and is_primitive_annotation(node.annotation):
                        out[node.target.id] = deepcopy(node.annotation)
                for child in iter_child_nodes(node):
                    if isinstance(child, (FunctionDef, ClassDef)):
                        continue
                    walk(child)

            for stmt in fn.body:
                walk(stmt)
            return out

        class BoundaryCallRetyper(NodeTransformer):
            def __init__(
                self,
                call_info,
                class_names,
                method_ret_fallback,
                method_arg_fallback,
                field_annotations,
                ambiguous_method_names,
                strict_limitations: bool,
            ):
                self.call_info = call_info
                self.class_names = class_names
                self.method_ret_fallback = method_ret_fallback
                self.method_arg_fallback = method_arg_fallback
                self.field_annotations = field_annotations
                self.ambiguous_method_names = ambiguous_method_names
                self.strict_limitations = strict_limitations
                self.class_stack: list[str] = []
                self.primitive_name_stack: list[dict[str, expr]] = []

            def visit_ClassDef(self, node: ClassDef):
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()
                return node

            def visit_FunctionDef(self, node: FunctionDef):
                self.primitive_name_stack.append(collect_primitive_name_annotations(node))
                self.generic_visit(node)
                self.primitive_name_stack.pop()
                return node

            def _matching_primitive_name_annotation(self, node: expr, annotation: expr | None):
                if not (isinstance(node, Name) and isinstance(annotation, Name) and is_primitive_annotation(annotation)):
                    return False
                if len(self.primitive_name_stack) == 0:
                    return False
                known = self.primitive_name_stack[-1].get(node.id)
                return isinstance(known, Name) and known.id == annotation.id

            def _embed_boundary_arg(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    if is_box_call(node):
                        return node
                    # If this name is already known to be the same primitive, avoid redundant T(x) wrapping.
                    if self._matching_primitive_name_annotation(node, annotation):
                        return wrap_box(node)
                    return wrap_box(coerce_primitive(annotation, node))
                if policy == "construct":
                    assert not is_passthrough_container_annotation(
                        annotation
                    ), "passthrough containers must use cast passthrough pass"
                    return wrap_construct(annotation, node)
                if policy == "cast":
                    return wrap_cast_or_construct(annotation, node)
                return node

            def _project_boundary_return(self, node: expr, annotation: expr | None):
                policy = annotation_policy(annotation)
                if policy == "box":
                    return coerce_primitive(annotation, node)
                if policy == "construct":
                    assert not is_passthrough_container_annotation(
                        annotation
                    ), "passthrough containers must use cast passthrough pass"
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

            def _resolve_receiver_owner_name(self, receiver: expr):
                if isinstance(receiver, Name):
                    if receiver.id in self.class_names:
                        return receiver.id
                    if receiver.id in ("self", "cls") and len(self.class_stack) > 0:
                        return self.class_stack[-1]
                    return None

                if isinstance(receiver, Attribute):
                    parent_owner = self._resolve_receiver_owner_name(receiver.value)
                    if parent_owner is None:
                        return None
                    class_fields = self.field_annotations.get(parent_owner)
                    if class_fields is None:
                        return None
                    field_annotation = class_fields.get(receiver.attr)
                    if isinstance(field_annotation, Name) and field_annotation.id in self.class_names:
                        return field_annotation.id
                    return None

                return None

            def _resolve_method_q_name(self, node: Call):
                assert isinstance(node.func, Attribute), "method resolution on non-attribute call unsupported"
                method_name = node.func.attr
                owner_name = self._resolve_receiver_owner_name(node.func.value)
                if owner_name is None:
                    if method_name in self.ambiguous_method_names:
                        return None
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
                        if node.func.attr in self.ambiguous_method_names:
                            if self.strict_limitations:
                                strict_fail(
                                    "ambiguous_method_names",
                                    f"unresolved ambiguous method call: {node.func.attr} at line {getattr(node, 'lineno', '?')}",
                                )
                            return node
                        arg_fallback = self.method_arg_fallback.get(node.func.attr)
                        ret_fallback = self.method_ret_fallback.get(node.func.attr)
                        if arg_fallback is None and ret_fallback is None:
                            if self.strict_limitations:
                                strict_fail(
                                    "strategy_resolution",
                                    f"unresolved call target without fallback: {node.func.attr} at line {getattr(node, 'lineno', '?')}",
                                )
                            return node
                        info = {
                            "typed_pos": arg_fallback["typed_pos"] if arg_fallback is not None else {},
                            "typed_kw": arg_fallback["typed_kw"] if arg_fallback is not None else {},
                            "ret_ann": ret_fallback,
                        }
                    else:
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
                vararg_pass_name = decide_param_strategy(vararg_ann).pass_name
                if vararg_pass_name is not None and pass_state[vararg_pass_name]:
                    assert False, f"detyping vararg annotation unsupported: {fn.name}"
            if fn.args.kwarg:
                kwarg_ann = fn.args.kwarg.annotation
                kwarg_pass_name = decide_param_strategy(kwarg_ann).pass_name
                if kwarg_pass_name is not None and pass_state[kwarg_pass_name]:
                    assert False, f"detyping kwarg annotation unsupported: {fn.name}"

            changed_signature = _detype_return(fn, pass_state) or changed_signature

            if changed_signature:
                fn.type_comment = None

            body_detyper = BodyDetyper(
                pass_state=pass_state,
                return_annotation=return_annotation,
                class_name=class_name,
                function_name=fn.name,
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
                    pass_name = decide_body_strategy(annotation).pass_name
                    if policy in ("box", "construct", "cast"):
                        if pass_name is not None and guide[TOP_LEVEL][pass_name]:
                            value = _detype_body(annotation, child.value, guide[TOP_LEVEL])
                            assert value is not None, "top-level body pass enabled but no rewrite"
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
        method_arg_fallback = consistent_method_param_annotations(call_info)
        tree = BoundaryCallRetyper(
            call_info,
            class_names,
            method_ret_fallback,
            method_arg_fallback,
            class_field_annotations,
            self.ambiguous_method_names,
            self.strict_limitations,
        ).visit(tree)
        if any(guide[TOP_LEVEL][pass_name] for pass_name in PASS_NAMES_BY_SCOPE[PASS_SCOPE_BODY]):
            detype_top_level_body(tree)
        tree = RedundantWrapperCleaner().visit(tree)
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
        plan_enabled = dict((q_fun_name, perm[i]) for i, q_fun_name in enumerate(self.fun_names))
        guide = {}
        for q_fun_name in self.fun_names:
            antr_name, fun_name = q_fun_name
            is_ambiguous_method = antr_name is not None and fun_name in self.ambiguous_method_names
            if is_ambiguous_method:
                if plan_enabled[q_fun_name]:
                    self._strict_fail("ambiguous_method_names", f"plan selected ambiguous method: {q_fun_name}")
                guide[q_fun_name] = dict(disabled_pass_state)
            else:
                guide[q_fun_name] = dict(enabled_pass_state if plan_enabled[q_fun_name] else disabled_pass_state)
        # Keep top-level in the plan bitset, but force module body detyping off.
        if TOP_LEVEL in guide:
            guide[TOP_LEVEL] = dict(disabled_pass_state)
        return guide

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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="assert when transform limitations are encountered instead of silently skipping",
    )
    args = parser.parse_args()

    benchmark_path = resolve_benchmark_path(args.benchmark, args.level, args.root)
    if not path.exists(benchmark_path):
        raise FileNotFoundError(f"benchmark file not found: {benchmark_path}")

    detyper = CinderDetyperBoxUnbox(
        benchmark_file_name=benchmark_path,
        python=args.python,
        scratch_dir=args.scratch,
        params=tuple(args.param),
        strict_limitations=args.strict,
    )

    print(f"benchmark: {benchmark_path}")
    print(f"functions: {detyper.fun_count()}")
    print(f"passes: {PASS_COUNT} ({', '.join(PASS_NAMES)})")
    print(f"plan bits: {detyper.plan_bit_count()}")
    print(f"strict: {args.strict}")

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
