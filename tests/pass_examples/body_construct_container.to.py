from __static__ import int64, CheckedList


def first_value() -> int64:
    xs = CheckedList[int64](CheckedList[int64]([1, 2, 3]))
    return CheckedList[int64](xs)[0]
