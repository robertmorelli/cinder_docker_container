from __static__ import inline


class Foo:
    pass


@inline
def project(x: Foo) -> Foo:
    return x
