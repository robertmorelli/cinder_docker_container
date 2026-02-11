from __static__ import int64, CheckedList


def first(_xs) -> int64:
    xs: CheckedList[int64] = CheckedList[int64](_xs)
    return xs[0]


def use(src: CheckedList[int64]) -> int64:
    out: int64 = first(CheckedList[int64](src))
    return out
