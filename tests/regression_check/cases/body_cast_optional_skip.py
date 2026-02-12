from __future__ import annotations

from __static__ import cast
from typing import Optional


class Node:
    def __init__(self, nxt: Optional[Node]) -> None:
        self.nxt = nxt


def hop(n: Optional[Node]) -> Optional[Node]:
    cur: Optional[Node] = n
    if cur is not None:
        return cur.nxt
    return None
