from __future__ import annotations
import random
import math
from typing import final, Set
from __static__ import int64, Array, CheckedList, cbool, box, cast
import time
GAMES: int = 200
MAXMOVES: int = 9 * 9 * 3
TIMESTAMP: int = 0
MOVES: int = 0

# detyper-status: types_kept
def to_pos(x: int64, y: int64) -> int64:
    return y * 9 + x

@final
class Square:

    # detyper-status: types_kept
    def __init__(self: Square, board: Board, pos: int) -> None:
        self.board: Board = board
        self.pos: int64 = int64(pos)
        self.timestamp: int = TIMESTAMP
        self.removestamp: int = TIMESTAMP
        self.zobrist_strings: Array[int64] = Array[int64](3)
        for ii in range(3):
            self.zobrist_strings[ii] = int64(random.randrange(9223372036854775807))
        self.color: int64 = 0
        self.reference: Square = self
        self.ledges: int64 = 0
        self.used: cbool = False
        self.neighbours: CheckedList[Square] = CheckedList[Square]([])
        self.temp_ledges: int64 = 0

    # detyper-status: types_kept
    def set_neighbours(self: Square) -> None:
        x: int64 = self.pos % 9
        y: int64 = self.pos // 9
        self.neighbours = []
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            newx: int64 = x + int64(dx)
            newy: int64 = y + int64(dy)
            if 0 <= newx < 9 and 0 <= newy < 9:
                bbb: Square = self.board.squares[to_pos(newx, newy)]
                self.neighbours.append(bbb)

    # detyper-status: types_kept
    def move(self: Square, color: int64) -> None:
        global TIMESTAMP, MOVES
        TIMESTAMP += 1
        MOVES += 1
        self.board.zobrist.update(cast(Square, self), box(color))
        self.color = color
        self.reference = self
        self.ledges = int64(0)
        self.used = True
        for neighbour in self.neighbours:
            neighcolor: int64 = neighbour.color
            if neighcolor == 0:
                self.ledges += 1
            else:
                neighbour_ref: Square = neighbour.find(True)
                if neighcolor == self.color:
                    if neighbour_ref.reference.pos != self.pos:
                        self.ledges += neighbour_ref.ledges
                        neighbour_ref.reference = self
                    self.ledges -= 1
                else:
                    neighbour_ref.ledges -= 1
                    if neighbour_ref.ledges == 0:
                        neighbour.remove(neighbour_ref, True)
        self.board.zobrist.add()

    # detyper-status: types_kept
    def remove(self: Square, reference: Square, update: bool) -> None:
        self.board.zobrist.update(cast(Square, self), box(int64(0)))
        self.removestamp = TIMESTAMP
        if update:
            self.color = 0
            self.board.emptyset.add(self.pos)
        for neighbour in self.neighbours:
            if neighbour.color != 0 and cbool(neighbour.removestamp != TIMESTAMP):
                neighbour_ref: Square = neighbour.find(update)
                if neighbour_ref.pos == reference.pos:
                    neighbour.remove(reference, update)
                elif update:
                    neighbour_ref.ledges += 1

    # detyper-status: types_kept
    def find(self: Square, update: bool) -> Square:
        reference: Square = self.reference
        if reference.pos != self.pos:
            reference = reference.find(update)
            if update:
                self.reference = reference
        return reference

@final
class EmptySet:

    # detyper-status: types_kept
    def __init__(self, board: Board) -> None:
        self.board: Board = board
        S2 = 9 * 9
        self.empties: Array[int64] = Array[int64](S2)
        self.empty_pos: Array[int64] = Array[int64](S2)
        for kk in range(S2):
            ii: int64 = int64(kk)
            self.empties[ii] = ii
            self.empty_pos[ii] = ii

    # detyper-status: types_kept
    def random_choice(self) -> int64:
        choices: int64 = int64(len(self.empties))
        while choices:
            i: int64 = int64(int(random.random() * box(choices)))
            pos = self.empties[i]
            if cast(int, self.board.useful(box(int64(pos)))):
                return pos
            choices -= 1
            self.set(box(i), box(int64(self.empties[choices])))
            self.set(box(choices), box(int64(pos)))
        return -1

    # detyper-status: types_kept
    def add(self, pos: int64) -> None:
        self.empty_pos[pos] = int64(len(self.empties))
        self.empties[pos] = pos

    # detyper-status: types_kept
    def remove(self, pos: int64, update: bool) -> None:
        self.set(box(int64(self.empty_pos[pos])), box(int64(self.empties[len(self.empties) - 1])))

    # detyper-status: types_removed
    def set(self, _i, _pos) -> None:
        i: int64 = int64(_i)
        pos: int64 = int64(_pos)
        self.empties[i] = pos
        self.empty_pos[pos] = i

@final
class ZobristHash:

    # detyper-status: types_kept
    def __init__(self, board: Board) -> None:
        self.hash_set: Set[int] = set()
        self.hash: int = 0
        for square in board.squares:
            self.hash ^= square.zobrist_strings[0]
        self.hash_set.clear()
        self.hash_set.add(self.hash)

    # detyper-status: types_removed
    def update(self, _square, _color) -> None:
        color: int64 = int64(_color)
        square: Square = cast(Square, _square)
        self.hash ^= square.zobrist_strings[square.color]
        self.hash ^= square.zobrist_strings[color]

    # detyper-status: types_kept
    def add(self) -> None:
        self.hash_set.add(self.hash)

    # detyper-status: types_removed
    def dupe(self):
        return self.hash in self.hash_set

@final
class Board:

    # detyper-status: types_kept
    def __init__(self) -> None:
        self.squares: CheckedList[Square] = CheckedList[Square]([])
        self.emptyset: EmptySet = EmptySet(self)
        self.zobrist: ZobristHash = ZobristHash(self)
        self.color: int64 = 2
        self.finished: bool = False
        self.lastmove: int64 = -2
        self.history: CheckedList[int] = CheckedList[int]([])
        self.white_dead: int = 0
        self.black_dead: int = 0
        for pos in range(9 * 9):
            self.squares.append(Square(self, pos))
        for square in self.squares:
            square.set_neighbours()
            square.color = 0
            square.used = False

    # detyper-status: types_removed
    def reset(self) -> None:
        for square in self.squares:
            square.color = 0
            square.used = False
        self.emptyset = EmptySet(self)
        self.zobrist = ZobristHash(self)
        self.color = 2
        self.finished = False
        self.lastmove = -2
        self.history = []
        self.white_dead = 0
        self.black_dead = 0

    # detyper-status: types_kept
    def move(self, pos: int64) -> None:
        square = self.squares[pos]
        if pos != -1:
            square.move(self.color)
            self.emptyset.remove(square.pos, True)
        elif self.lastmove == -1:
            self.finished = True
        if self.color == 2:
            self.color = 1
        else:
            self.color = 2
        self.lastmove = pos
        self.history.append(box(pos))

    # detyper-status: types_removed
    def random_move(self):
        return box(int64(self.emptyset.random_choice()))

    # detyper-status: types_kept
    def useful_fast(self, square: Square) -> bool:
        if not square.used:
            for neighbour in square.neighbours:
                if neighbour.color == 0:
                    return True
        return False

    # detyper-status: types_removed
    def useful(self, _pos):
        pos: int64 = int64(_pos)
        global TIMESTAMP
        TIMESTAMP += 1
        square = self.squares[pos]
        if self.useful_fast(square):
            return True
        old_hash = self.zobrist.hash
        self.zobrist.update(cast(Square, square), box(int64(self.color)))
        empties = opps = weak_opps = neighs = weak_neighs = 0
        for neighbour in square.neighbours:
            neighcolor = neighbour.color
            if neighcolor == 0:
                empties += 1
                continue
            neighbour_ref = neighbour.find(False)
            if neighbour_ref.timestamp != TIMESTAMP:
                if neighcolor == self.color:
                    neighs += 1
                else:
                    opps += 1
                neighbour_ref.timestamp = TIMESTAMP
                neighbour_ref.temp_ledges = neighbour_ref.ledges
            neighbour_ref.temp_ledges -= 1
            if neighbour_ref.temp_ledges == 0:
                if neighcolor == self.color:
                    weak_neighs += 1
                else:
                    weak_opps += 1
                    neighbour_ref.remove(neighbour_ref, False)
        dupe = cast(bool, self.zobrist.dupe())
        self.zobrist.hash = old_hash
        strong_neighs = neighs - weak_neighs
        strong_opps = opps - weak_opps
        return not dupe and (empties or weak_opps or (strong_neighs and (strong_opps or weak_neighs)))

    # detyper-status: types_removed
    def useful_moves(self):
        return CheckedList[int]([pos for pos in self.emptyset.empties if cast(int, self.useful(box(int64(pos))))])

    # detyper-status: types_kept
    def replay(self, history: Array[int64]) -> None:
        for pos in history:
            self.move(pos)

    # detyper-status: types_kept
    def score(self, color: int64) -> float:
        if color == 1:
            count = 7.5 + self.black_dead
        else:
            count = self.white_dead
        for square in self.squares:
            squarecolor = square.color
            if squarecolor == color:
                count += 1
            elif squarecolor == 0:
                surround = 0
                for neighbour in square.neighbours:
                    if neighbour.color == color:
                        surround += 1
                if surround == len(square.neighbours):
                    count += 1
        return count

    # detyper-status: types_removed
    def check(self) -> None:
        for square in self.squares:
            if square.color == 0:
                continue
            members1 = set([square])
            changed = True
            while changed:
                changed = False
                for member in members1.copy():
                    for neighbour in member.neighbours:
                        if int64(neighbour.color) == square.color and cbool(neighbour not in members1):
                            changed = True
                            members1.add(neighbour)
            ledges1 = 0
            for member in members1:
                for neighbour in member.neighbours:
                    if neighbour.color == 0:
                        ledges1 += 1
            root = square.find(False)
            members2 = set()
            for square2 in self.squares:
                if square2.color != 0 and cbool(square2.find(False) == root):
                    members2.add(square2)
            ledges2 = root.ledges
            empties1 = set(self.emptyset.empties)
            empties2 = set()
            for square in self.squares:
                if square.color == 0:
                    empties2.add(box(square.pos))

@final
class UCTNode:

    # detyper-status: types_kept
    def __init__(self) -> None:
        self.bestchild: None | UCTNode = None
        self.pos: int64 = -1
        self.wins: int64 = 0
        self.losses: int64 = 0
        self.pos_child: CheckedList[None | UCTNode] = CheckedList[None | UCTNode]([None for x in range(9 * 9)])
        self.parent: None | UCTNode = None
        self.unexplored: CheckedList[int] = CheckedList[int]([])

    # detyper-status: types_kept
    def play(self, board: Board) -> None:
        """ uct tree search """
        color: int = box(board.color)
        node: UCTNode = self
        path: CheckedList[UCTNode] = CheckedList[UCTNode]([node])
        pos: int64 = 0
        while True:
            pos = int64(node.select(cast(Board, board)))
            if pos == -1:
                break
            board.move(pos)
            child = node.pos_child[pos]
            if not child:
                child = node.pos_child[pos] = UCTNode()
                child.unexplored = CheckedList[int](board.useful_moves())
                child.pos = pos
                child.parent = node
                path.append(child)
                break
            path.append(child)
            node = child
        self.random_playout(cast(Board, board))
        self.update_path(cast(Board, board), cast(int, color), CheckedList[UCTNode](path))

    # detyper-status: types_removed
    def select(self, _board):
        board: Board = cast(Board, _board)
        ' select move; unexplored children first, then according to uct value '
        if self.unexplored:
            i = cast(int, random.randrange(len(self.unexplored)))
            pos = box(int64(self.unexplored[cast(int, i)]))
            self.unexplored[cast(int, i)] = self.unexplored[len(self.unexplored) - 1]
            self.unexplored.pop()
            return box(int64(pos))
        elif self.bestchild is not None:
            return box(int64(self.bestchild.pos))
        else:
            return box(int64(-1))

    # detyper-status: types_removed
    def random_playout(self, _board) -> None:
        board: Board = cast(Board, _board)
        ' random play until both players pass '
        for x in range(MAXMOVES):
            if board.finished:
                break
            board.move(int64(board.random_move()))

    # detyper-status: types_removed
    def update_path(self, _board, _color, _path) -> None:
        path: CheckedList[UCTNode] = CheckedList[UCTNode](_path)
        board: Board = cast(Board, _board)
        color: int = cast(int, _color)
        ' update win/loss count along path '
        wins = board.score(2) >= board.score(1)
        for node in path:
            if color == 2:
                color = 1
            else:
                color = 2
            if wins == (color == 2):
                node.wins += 1
            else:
                node.losses += 1
            if node.parent is not None:
                mypar = node.parent
                bc = mypar.best_child()
                if node.parent is not None:
                    node.parent.bestchild = bc

    # detyper-status: types_kept
    def score(self) -> float:
        winrate = box(self.wins) / float(box(self.wins) + box(self.losses))
        parentvisits: int = 0
        if self.parent is not None:
            parentvisits += box(self.parent.wins)
        if self.parent is not None:
            parentvisits += box(self.parent.losses)
        if not parentvisits:
            return winrate
        nodevisits: int = box(self.wins + self.losses)
        return winrate + math.sqrt(math.log(parentvisits) / (5 * nodevisits))

    # detyper-status: types_kept
    def best_child(self) -> None | UCTNode:
        maxscore = -1
        maxchild = None
        for child in self.pos_child:
            if child and child.score() > maxscore:
                maxchild = child
                maxscore = child.score()
        return maxchild

    # detyper-status: types_removed
    def best_visited(self):
        maxvisits = box(int64(-1))
        maxchild = None
        for child in self.pos_child:
            if child is not None and box(child.wins + child.losses > int64(maxvisits)):
                maxvisits = box(int64(child.wins + child.losses))
                maxchild = child
        return maxchild

# detyper-status: types_removed
def computer_move(_board):
    board: Board = cast(Board, _board)
    global MOVES
    pos = int64(board.random_move())
    if pos == -1:
        return -1
    tree = UCTNode()
    tree.unexplored = CheckedList[int](board.useful_moves())
    nboard = Board()
    num_hist = len(board.history)
    ahist = Array[int64](num_hist)
    for ii in range(num_hist):
        ahist[ii] = int64(board.history[ii])
    for game in range(GAMES):
        node = tree
        nboard.reset()
        nboard.replay(ahist)
        node.play(nboard)
    bvv = cast(None | UCTNode, tree.best_visited())
    if bvv is not None:
        return box(bvv.pos)
    return -1
ITERATIONS = 2
if __name__ == '__main__':
    start_time = time.time()
    for i in range(ITERATIONS):
        random.seed(1)
        board = Board()
        pos = cast(int, computer_move(cast(Board, board)))
    end_time = time.time()
    runtime = end_time - start_time
    print(runtime)
