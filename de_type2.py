from ast import (
    parse,
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
    Constant,
)
from functools import cache, reduce
from itertools import chain
from random import sample
from subprocess import run, CompletedProcess
from time import perf_counter
from multiprocessing import Pool, cpu_count
from datetime import datetime
from json import dump
from multiprocessing import Value, Lock
from time import sleep
from os import path, makedirs, environ
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
GuideType = dict[GuideKey, bool]
QNameType = set[GuideKey]

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
        result = me.run_permutation(perm)
        with PermutationLocker._g_lock:
            PermutationLocker._g_counter.value += 1
        return CinderDetyper2._perm_name(perm), result


class CinderDetyper2:
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
        t2.end()
        t1.end()

    def fun_count(self):
        return len(self.fun_names)

    @staticmethod
    @cache
    def read_text(file_name):
        with open(file_name, encoding="utf-8") as f:
            return f.read()
        raise RuntimeError("python file not found")

    def _read_ast(self):
        return parse(CinderDetyper2.read_text(self.benchmark_file_name), type_comments=True)

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
            assert antr_name in antr_graph, "function before class ancestors"
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

        def get_fun_q_name(fun_name: str, antr_name=None) -> GuideKey:
            q_fun_name, *alternate_names = get_fun_q_names(fun_name, antr_name=antr_name)
            assert len(alternate_names) <= 1, f"function identity failure: one of {alternate_names}"
            return alternate_names[0] if alternate_names else q_fun_name

        def fun_should_be_detyped(fun_name: str, antr_name: str | None = None):
            q_fun_name: GuideKey = get_fun_q_name(fun_name, antr_name=antr_name)
            assert q_fun_name in guide, "function type status unspecified"
            return guide[q_fun_name]

        def detype_fun_params(fn: FunctionDef):
            args = fn.args
            args_to_clear = chain(args.posonlyargs, args.args, args.kwonlyargs, (args.vararg, args.kwarg))
            args_to_clear_filtered = filter(bool, args_to_clear)
            for a in args_to_clear_filtered:
                a.annotation = None

        def detype_fun_ret(fn: FunctionDef):
            fn.returns = None
            fn.type_comment = None

        def detype_body(node: AST, antr_name: str | None = None, fun_name: str | None = None):
            inside_fun = fun_name is not None
            inside_class = antr_name is not None
            is_toplevel = (not inside_fun) and (not inside_class)
            for child_node in iter_child_nodes(node):
                is_fun = isinstance(child_node, FunctionDef)
                is_class = isinstance(child_node, ClassDef)
                is_assign = isinstance(child_node, Assign)
                is_annassign = isinstance(child_node, AnnAssign)

                if (not is_toplevel) and is_fun:
                    detype_body(child_node, antr_name=antr_name, fun_name=child_node.name)
                    detype_fun_params(child_node)
                    detype_fun_ret(child_node)
                elif (not is_toplevel) and is_class:
                    detype_body(child_node, antr_name=child_node.name, fun_name=fun_name)
                elif is_assign:
                    child_node.type_comment = None
                elif is_annassign:
                    child_node.__class__ = Assign
                    child_node.targets = [child_node.target]
                    if getattr(child_node, "value", None) is None:
                        child_node.value = Constant(value=None)
                    if hasattr(child_node, "annotation"):
                        del child_node.annotation
                    child_node.type_comment = None

        def detype_walker(node: AST, antr_name: str | None = None):
            for child_node in iter_child_nodes(node):
                is_fun = isinstance(child_node, FunctionDef)
                is_class = isinstance(child_node, ClassDef)
                if is_fun:
                    if fun_should_be_detyped(child_node.name, antr_name=antr_name):
                        detype_body(child_node)
                        detype_fun_params(child_node)
                        detype_fun_ret(child_node)
                elif is_class:
                    detype_walker(child_node, antr_name=child_node.name)

        detype_walker(tree)
        if guide[TOP_LEVEL]:
            detype_body(tree)
        fix_missing_locations(tree)
        return split_lines(unparse(tree))

    @cache
    @staticmethod
    def _perm_name(perm: Permutation) -> str:
        return hex(int("".join(str(int(b)) for b in perm), 2))

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
        trunc_name = CinderDetyper2.file_trunc(path.basename(file_name))
        perm_string = CinderDetyper2._perm_name(perm)
        return f"{trunc_name}_{perm_string}"

    def _detype_by_permutation(self, perm: Permutation) -> str:
        tree = self._read_ast()
        guide: GuideType = dict(zip(self.fun_names, perm))
        return self._detype_funs(tree, guide)

    def perm_file_name(self, perm: Permutation):
        benchmark_name = path.basename(path.dirname(self.benchmark_dir))
        level_name = path.basename(self.benchmark_dir)
        file_stem = CinderDetyper2.q_file_trunc(perm, self.benchmark_file_name)
        return f"{self.scratch_dir}/{benchmark_name}_{level_name}_{file_stem}.py"

    def _ensure_scratch_dir(self):
        makedirs(self.scratch_dir, exist_ok=True)

    def _run_env(self):
        env = dict(environ)
        py_paths = [self.benchmark_dir]
        current = env.get("PYTHONPATH")
        if current:
            py_paths.append(current)
        env["PYTHONPATH"] = ":".join(py_paths)
        return env

    def write_permutation(self, perm: Permutation):
        self._ensure_scratch_dir()
        t1 = Timer("retype")
        new_file_string = self._detype_by_permutation(perm)
        new_file_name = self.perm_file_name(perm)
        t1.end()
        t2 = Timer("write file")
        with open(new_file_name, mode="w", encoding="utf-8") as f:
            f.write(new_file_string)
        t2.end()

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

    def run_permutation(self, perm: Permutation) -> CompletedProcess[str]:
        self.write_permutation(perm)
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
        trunk = CinderDetyper2.file_trunc(path.basename(self.benchmark_file_name))
        results_file_name = f"results_DETYPE2_{trunk}_{stamp}_samples_{samples}_granularity_{granularity}.json"

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

        unknowns, failure_stats, success_count, successes, failures = CinderDetyper2._collect_failure_stats(results)

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
                    "info_dump": CinderDetyper2._make_info_dump(results),
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
    parser.add_argument("--scratch", default="/tmp/detype2")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--granularity", type=int, default=1)
    parser.add_argument("--param", action="append", default=[], help="benchmark runtime arg (repeatable)")
    parser.add_argument("--show-perm", type=str, default=None, help="hex permutation id to print transformed source")
    parser.add_argument("--test", action="store_true", help="run typed and fully detyped once")
    args = parser.parse_args()

    benchmark_path = resolve_benchmark_path(args.benchmark, args.level, args.root)
    if not path.exists(benchmark_path):
        raise FileNotFoundError(f"benchmark file not found: {benchmark_path}")

    detyper = CinderDetyper2(
        benchmark_file_name=benchmark_path,
        python=args.python,
        scratch_dir=args.scratch,
        params=tuple(args.param),
    )

    print(f"benchmark: {benchmark_path}")
    print(f"functions: {detyper.fun_count()}")

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

    detyper.find_permutation_errors(samples=args.samples, granularity=args.granularity)


if __name__ == "__main__":
    main()
