from __static__ import cast, int64, box


class Foo:
    pass


def demo(x):
    a = cast(Foo, cast(Foo, x))
    b = int64(int64(1))
    c = box(box(int64(2)))
    return a, b, c
