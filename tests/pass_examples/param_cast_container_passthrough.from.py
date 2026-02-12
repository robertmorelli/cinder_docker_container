from __static__ import Array, int64


def first(xs: Array[int64]) -> int64:
    return xs[int64(0)]


def use(src: Array[int64]) -> int64:
    out: int64 = first(src)
    return out
