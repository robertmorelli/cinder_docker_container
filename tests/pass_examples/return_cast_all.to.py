from __static__ import cast


class Foo:
    pass


def make_foo():
    return Foo()


def use():
    out: Foo = cast(Foo, make_foo())
    return out
