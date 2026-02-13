from __static__ import cast

class Widget:

    # detyper-status: types_kept
    def tick(self) -> None:
        return None

# detyper-status: types_kept
def make_widget() -> Widget:
    return Widget()

# detyper-status: types_removed
def run() -> None:
    w = cast(Widget, make_widget())
    cast(Widget, w).tick()