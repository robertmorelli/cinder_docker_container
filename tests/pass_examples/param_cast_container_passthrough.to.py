from __static__ import Array, int64, cast

# detyper-status: types_removed
def first(_xs) -> int64:
    xs: Array[int64] = cast(Array[int64], _xs)
    return xs[int64(0)]

# detyper-status: types_kept
def use(src: Array[int64]) -> int64:
    out: int64 = first(cast(Array[int64], src))
    return out