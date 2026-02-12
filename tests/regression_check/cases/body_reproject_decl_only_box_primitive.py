from __static__ import CheckedList, int64


VALUES = CheckedList[int64]([1, 2, 3])


def inc_first(values=VALUES) -> int64:
    x: int64
    for x in values:
        return x + 1
    return 0
