"""
Simple, brute-force N-Queens solver. Using static python
Made by sebastiancr@fb.com(Sebastian Chaves) based on main.py made by collinwinter@google.com (Collin Winter)
"""
from __future__ import annotations
import __static__
from __static__ import int64, box, Array, cbool, clen, cast
from typing import List, Generator, Iterator
import time

# detyper-status: types_kept
def static_abs(v: int64) -> int64:
    if v < 0:
        return -v
    return v

# detyper-status: types_kept
def create_array(start: int64, end: int64, step: int64) -> Array[int64]:
    """
    Function that creates an array that contains elements from start (inclusive) to end (non-inclusve) increasing the given steps
    Note: if It is not possible to go from start to end, an empty array will be returned.
    For example: create_array(2,7,2) -> (2,4,6) ; create_array(1,4,1)->(1,2,3)
    """
    c: int64 = start
    i: int64 = 0
    if (end - start) * step <= 0:
        return Array[int64](0)
    size: int64 = int64((static_abs(end - start) - 1) / static_abs(step) + 1)
    a: Array[int64] = Array[int64](box(size))
    while i < size:
        a[i] = c
        c = c + step
        i = i + 1
    return a

# detyper-status: types_removed
def permutations(_pool, _r=-1) -> Iterator[Array[int64]]:
    r: int64 = int64(_r)
    pool: Array[int64] = cast(Array[int64], _pool)
    n = clen(pool)
    if r == -1:
        r = n
    rb = box(r)
    indices = cast(Array[int64], create_array(0, n, 1))
    cycles = cast(Array[int64], create_array(n, n - r, -1))
    per = cast(Array[int64], Array[int64](rb))
    idx = box(int64(0))
    while int64(idx) < r:
        cast(Array[int64], per)[int64(idx)] = pool[cast(Array[int64], indices)[int64(idx)]]
        idx += 1
    yield cast(Array[int64], per)
    while n:
        i = rb - 1
        while i >= 0:
            cast(Array[int64], cycles)[i] -= 1
            if cast(Array[int64], cycles)[i] == 0:
                lastN = box(int64(cast(Array[int64], indices)[i]))
                for ii in range(i + 1, len(cast(Array[int64], indices))):
                    cast(Array[int64], indices)[ii - 1] = cast(Array[int64], indices)[ii]
                cast(Array[int64], indices)[len(cast(Array[int64], indices)) - 1] = int64(lastN)
                cast(Array[int64], cycles)[i] = n - int64(i)
            else:
                j = cast(Array[int64], cycles)[i]
                tmp = box(int64(cast(Array[int64], indices)[-j]))
                cast(Array[int64], indices)[-j] = cast(Array[int64], indices)[i]
                cast(Array[int64], indices)[i] = int64(tmp)
                idx = box(int64(0))
                while int64(idx) < r:
                    cast(Array[int64], per)[int64(idx)] = pool[cast(Array[int64], indices)[int64(idx)]]
                    idx += 1
                yield cast(Array[int64], per)
                break
            i -= 1
        if i == -1:
            return

# detyper-status: types_kept
def solve(queen_count: int) -> Iterator[Array[int64]]:
    """N-Queens solver.

    Args:
        queen_count: the number of queens to solve for. This is also the
            board size.

    Yields:
        Solutions to the problem. Each yielded value is looks like
        (3, 8, 2, 1, 4, ..., 6) where each number is the column position for the
        queen, and the index into the tuple indicates the row.
    """
    cols: Iterator[int] = range(queen_count)
    static_cols: Array[int64] = create_array(0, int64(queen_count), 1)
    for vec in permutations(cast(Array[int64], static_cols)):
        if queen_count == len(set((vec[i] + i for i in cols))) == len(set((vec[i] - i for i in cols))):
            yield vec

# detyper-status: types_removed
def bench_n_queens(_queen_count):
    queen_count: int = cast(int, _queen_count)
    '\n    Return all the possible valid configurations of the queens\n    in a board of size queen_count.\n    See solve method to understand it better\n    '
    return list(solve(queen_count))
if __name__ == '__main__':
    import sys
    num_iterations = 1
    if len(sys.argv) > 1:
        num_iterations = int(sys.argv[1])
    queen_count = 8
    startTime = time.time()
    for _ in range(num_iterations):
        res = cast(List[Array[int64]], bench_n_queens(cast(int, queen_count)))
    endTime = time.time()
    runtime = endTime - startTime
    print(runtime)
