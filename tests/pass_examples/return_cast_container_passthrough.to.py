from __static__ import Array, int64, cast


def make_items(xs: Array[int64]):
    return xs


def use(v: Array[int64]) -> int64:
    out: Array[int64] = cast(Array[int64], make_items(v))
    return out[int64(0)]
