from __static__ import cast


class Foo:
    pass


def make_foo() -> Foo:
    return Foo()


def use() -> Foo:
    out: Foo = make_foo()
    return out
