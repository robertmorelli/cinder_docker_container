from __static__ import int64, CheckedList

X: CheckedList[int64] = CheckedList[int64]([7])


def get_x() -> int64:
    return X[0]
