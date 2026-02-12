from __static__ import Array, int64


def first_value(xs: Array[int64]) -> int64:
    arr: Array[int64] = xs
    return arr[int64(0)]
