from __static__ import int64, box


def bump() -> int64:
    n = box(int64(41))
    return int64(n) + 1
