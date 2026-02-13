from __static__ import cast

class Foo:
    pass

# detyper-status: types_removed
def echo(_foo) -> Foo:
    foo: Foo = cast(Foo, _foo)
    return foo

# detyper-status: types_kept
def use(x: Foo) -> Foo:
    out: Foo = echo(cast(Foo, x))
    return out