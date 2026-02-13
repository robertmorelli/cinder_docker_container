from __static__ import cast

class Foo:
    pass

# detyper-status: types_removed
def make_foo():
    return Foo()

# detyper-status: types_kept
def use() -> Foo:
    out: Foo = cast(Foo, make_foo())
    return out