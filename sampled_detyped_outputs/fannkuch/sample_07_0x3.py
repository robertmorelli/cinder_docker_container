"""
The Computer Language Benchmarks Game
http://benchmarksgame.alioth.debian.org/
Contributed by Sokolov Yura, modified by Tupteq.
"""
from __future__ import annotations
import __static__
from __static__ import box, int64, Array, cast
from typing import Callable, List
import sys
import time
DEFAULT_ARG = 9

def fannkuch(_nb):
    nb: int = cast(int, _nb)
    n = box(int64(nb))
    count = cast(Array[int64], Array[int64](nb))
    i = box(int64(0))
    while int64(i) < int64(n):
        cast(Array[int64], count)[int64(i)] = int64(i) + 1
        i += 1
    max_flips = box(int64(0))
    m = box(int64(int64(n) - 1))
    r = box(int64(n))
    perm1 = cast(Array[int64], Array[int64](nb))
    perm = cast(Array[int64], Array[int64](nb))
    i = box(int64(0))
    while int64(i) < int64(n):
        cast(Array[int64], perm1)[int64(i)] = int64(i)
        cast(Array[int64], perm)[int64(i)] = int64(i)
        i += 1
    perm0 = cast(Array[int64], Array[int64](nb))
    while 1:
        while int64(r) != 1:
            cast(Array[int64], count)[int64(r) - 1] = int64(r)
            r -= 1
        if cast(Array[int64], perm1)[0] != 0 and cast(Array[int64], perm1)[int64(m)] != int64(m):
            i = box(int64(0))
            while int64(i) < int64(n):
                cast(Array[int64], perm)[int64(i)] = cast(Array[int64], perm1)[int64(i)]
                i += 1
            flips_count = box(int64(0))
            k = box(int64(cast(Array[int64], perm)[0]))
            while int64(k):
                i = box(int64(k))
                while int64(i) >= 0:
                    cast(Array[int64], perm0)[int64(i)] = cast(Array[int64], perm)[int64(k) - int64(i)]
                    i -= 1
                i = box(int64(k))
                while int64(i) >= 0:
                    cast(Array[int64], perm)[int64(i)] = cast(Array[int64], perm0)[int64(i)]
                    i -= 1
                flips_count += 1
                k = box(int64(cast(Array[int64], perm)[0]))
            if int64(flips_count) > int64(max_flips):
                max_flips = box(int64(flips_count))
        while int64(r) != int64(n):
            first = box(int64(cast(Array[int64], perm1)[0]))
            i = box(int64(1))
            while int64(i) <= int64(r):
                cast(Array[int64], perm1)[int64(i) - 1] = cast(Array[int64], perm1)[int64(i)]
                i += 1
            cast(Array[int64], perm1)[int64(r)] = int64(first)
            cast(Array[int64], count)[int64(r)] -= 1
            if cast(Array[int64], count)[int64(r)] > 0:
                break
            r += 1
        else:
            return box(int64(max_flips))
    return 0
if __name__ == '__main__':
    num_iterations = 1
    if len(sys.argv) > 1:
        num_iterations = int(sys.argv[1])
    start_time = time.time()
    for _ in range(num_iterations):
        res = cast(int, fannkuch(cast(int, DEFAULT_ARG)))
        assert res == 30
    end_time = time.time()
    runtime = end_time - start_time
    print(runtime / num_iterations)
