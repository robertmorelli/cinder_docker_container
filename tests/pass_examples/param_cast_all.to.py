from __static__ import cast


class Foo:
    pass


def echo(_foo) -> Foo:
    foo: Foo = cast(Foo, _foo)
    return foo


def use(x: Foo) -> Foo:
    out: Foo = echo(cast(Foo, x))
    return out
