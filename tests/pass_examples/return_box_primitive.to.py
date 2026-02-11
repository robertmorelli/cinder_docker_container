from __static__ import int64, box


def inc(x: int64):
    return box(int64(x + 1))


def use(v: int64) -> int64:
    out: int64 = int64(inc(v))
    return out
