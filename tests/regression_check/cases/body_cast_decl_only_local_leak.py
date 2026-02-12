from __static__ import CheckedList, double


class Point:
    def __init__(self, x: double):
        self.x: double = x


POINTS = CheckedList[Point]([Point(1.0)])


def leak(values=POINTS) -> double:
    p: Point
    for p in values:
        out: double = p.x
        return out
    return 0.0
