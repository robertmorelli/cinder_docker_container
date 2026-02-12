from typing import List, Tuple

Population = List[float]


def build() -> Population:
    return [1.0, 2.0]


def evolve(p: Population) -> Population:
    return p


def simulation_to_lines(p: Population) -> List[Tuple[int, float]]:
    return [(0, p[0])]


def run() -> None:
    simulation_to_lines(evolve(build()))
