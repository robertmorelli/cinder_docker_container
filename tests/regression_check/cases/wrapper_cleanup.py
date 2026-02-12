from __static__ import cast, int64


class Foo:
    pass


def demo(x: Foo) -> int64:
    y: Foo = cast(Foo, cast(Foo, x))
    n: int64 = int64(int64(1))
    _ = int64(int64(n))
    if y is None:
        return 0
    return n
