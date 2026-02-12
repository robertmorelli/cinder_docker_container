from __future__ import annotations
from math import sin, cos, sqrt
from __static__ import CheckedList, inline
from typing import final


@final
class Point(object):
    def __init__(self: Point, i: float) -> None:
        x: float = sin(i)
        self.x: float = x
        self.y: float = cos(i) * 3
        self.z: float = (x * x) / 2

    def normalize(self: Point) -> None:
        x: float = self.x
        y: float = self.y
        z: float = self.z
        norm: float = sqrt(x * x + y * y + z * z)
        self.x /= norm
        self.y /= norm
        self.z /= norm

    def maximize(self: Point, other: Point) -> Point:
        self.x = self.x if self.x > other.x else other.x
        self.y = self.y if self.y > other.y else other.y
        self.z = self.z if self.z > other.z else other.z
        return self


def maximize(points: CheckedList[Point]) -> Point:
    nxt: Point = points[0]
    for p in points[1:]:
        nxt = nxt.maximize(p)
    return nxt


@inline
def benchmark(n: int) -> Point:
    points: CheckedList[Point] = CheckedList[Point]([Point(i) for i in range(n)])
    for p in points:
        p.normalize()
    return maximize(points)
