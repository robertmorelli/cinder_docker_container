from __static__ import int64


def inc(x: int64) -> int64:
    return x + 1


def use(v: int64) -> int64:
    out: int64 = inc(v)
    return out
