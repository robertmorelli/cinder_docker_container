from __static__ import int64


def add_one(x: int64) -> int64:
    return x + 1


def use(v: int64) -> int64:
    out: int64 = add_one(v)
    return out
