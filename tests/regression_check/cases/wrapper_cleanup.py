from __static__ import cast, int64, box


class Foo:
    pass


def demo(x: Foo) -> int64:
    y: Foo = cast(Foo, cast(Foo, x))
    n: int64 = int64(int64(1))
    _ = box(box(n))
    if y is None:
        return 0
    return n
