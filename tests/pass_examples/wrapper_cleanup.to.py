from __static__ import cast, int64, box


class Foo:
    pass


def demo(x):
    a = cast(Foo, x)
    b = int64(1)
    c = box(int64(2))
    return a, b, c
