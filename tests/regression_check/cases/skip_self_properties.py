from __static__ import int64


class Counter:
    def __init__(self, v: int64) -> None:
        self.value: int64 = v

    def set_value(self, v: int64) -> None:
        self.value = v

    def get_value(self) -> int64:
        return self.value


def run(x: int64) -> int64:
    c = Counter(x)
    c.set_value(x + 1)
    return c.get_value()
