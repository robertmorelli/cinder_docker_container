from __static__ import int64, CheckedList


def first(xs: CheckedList[int64]) -> int64:
    return xs[0]


def use(src: CheckedList[int64]) -> int64:
    out: int64 = first(src)
    return out
