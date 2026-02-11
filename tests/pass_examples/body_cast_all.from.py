from __static__ import cast


class Widget:
    def tick(self) -> None:
        return None


def make_widget() -> Widget:
    return Widget()


def run() -> None:
    w: Widget = make_widget()
    w.tick()
