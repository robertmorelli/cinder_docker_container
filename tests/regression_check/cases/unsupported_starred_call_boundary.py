from __static__ import int64


def callee(x: int64) -> int64:
    return x + 1


def caller(values) -> int64:
    return callee(*values)
