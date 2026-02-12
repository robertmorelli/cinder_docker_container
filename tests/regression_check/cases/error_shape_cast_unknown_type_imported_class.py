from imported_type_support import Population


def build() -> Population:
    return Population(1)


def evolve(p: Population) -> Population:
    return p


def run() -> Population:
    return evolve(build())
