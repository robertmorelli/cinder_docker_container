from __static__ import int64, box

# detyper-status: types_removed
def bump() -> int64:
    n = box(int64(41))
    return int64(n) + 1