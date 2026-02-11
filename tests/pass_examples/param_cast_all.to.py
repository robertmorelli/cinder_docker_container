from __static__ import cast


class Foo:
    pass


def echo(_foo) -> Foo:
    foo: Foo = cast(Foo, _foo)
    return foo


def use(_x) -> Foo:
    x: Foo = cast(Foo, _x)
    out: Foo = echo(cast(Foo, x))
    return out
