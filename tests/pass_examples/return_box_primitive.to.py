from __static__ import int64, box

# detyper-status: types_removed
def inc(x: int64):
    return box(int64(x + 1))

# detyper-status: types_kept
def use(v: int64) -> int64:
    out: int64 = int64(inc(v))
    return out