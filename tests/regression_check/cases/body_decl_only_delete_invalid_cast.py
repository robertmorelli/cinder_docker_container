class Foo:
    pass


def read_decl_only_cast() -> Foo:
    x: Foo
    return x
