from __static__ import Array, box, int64


def make_array(n: int64) -> Array[int64]:
    return Array[int64](box(n))


def read_first(n: int64) -> int64:
    arr: Array[int64] = make_array(n)
    idx: int64 = 0
    return arr[idx]
