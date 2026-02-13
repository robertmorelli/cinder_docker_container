from __static__ import int64, CheckedList

# detyper-status: types_removed
def make_items():
    return CheckedList[int64]([1, 2, 3])

# detyper-status: types_kept
def use() -> int64:
    out: CheckedList[int64] = CheckedList[int64](make_items())
    return out[0]