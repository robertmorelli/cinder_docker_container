from __static__ import Array, int64, cast

# detyper-status: types_removed
def first_value(xs: Array[int64]) -> int64:
    arr = cast(Array[int64], xs)
    return cast(Array[int64], arr)[int64(0)]