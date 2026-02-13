from __static__ import int64, box

# detyper-status: types_removed
def add_one(_x) -> int64:
    x: int64 = int64(_x)
    return x + 1

# detyper-status: types_kept
def use(v: int64) -> int64:
    out: int64 = add_one(box(v))
    return out