from __static__ import int64


def total(*xs: int64) -> int64:
    out: int64 = 0
    for x in xs:
        out += x
    return out
