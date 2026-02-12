from __static__ import cast


class Foo:
    pass


def echo(foo: Foo) -> Foo:
    return foo


def use(x: Foo) -> Foo:
    out: Foo = echo(x)
    return out
