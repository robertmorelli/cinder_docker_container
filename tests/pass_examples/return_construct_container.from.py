from __static__ import int64, CheckedList


def make_items() -> CheckedList[int64]:
    return CheckedList[int64]([1, 2, 3])


def use() -> int64:
    out: CheckedList[int64] = make_items()
    return out[0]
