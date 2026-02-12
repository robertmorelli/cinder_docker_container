from __future__ import annotations

from __static__ import CheckedList
from typing import List, Tuple
from typing import Callable as Function


def eval_A(i: float, j: float) -> float:
    return 1.0 / ((i + j) * (i + j + 1) // 2 + i + 1)


def eval_times_u(func: Function[[Tuple[float, List[float]]], float], u: List[float]) -> List[float]:
    return [func((i, u)) for i in range(len(CheckedList[u]))]


def eval_AtA_times_u(u: List[float]) -> List[float]:
    return eval_times_u(part_At_times_u, eval_times_u(part_A_times_u, u))


def part_A_times_u(i_u: (float, List[float])) -> float:
    i, u = i_u
    partial_sum = 0
    for j, u_j in enumerate(u):
        partial_sum += eval_A(i, j) * u_j
    return partial_sum


def part_At_times_u(i_u: (float, List[float])) -> float:
    i, u = i_u
    partial_sum = 0
    for j, u_j in enumerate(u):
        partial_sum += eval_A(j, i) * u_j
    return partial_sum
