"""
main.py
============

Ported for the PyPy project.
Contributed by Daniel Lindsley

This implementation of the DeltaBlue benchmark was directly ported
from the `V8's source code`_, which was in turn derived
from the Smalltalk implementation by John Maloney and Mario
Wolczko. The original Javascript implementation was licensed under the GPL.

It's been updated in places to be more idiomatic to Python (for loops over
collections, a couple magic methods, ``OrderedCollection`` being a list & things
altering those collections changed to the builtin methods) but largely retains
the layout & logic from the original. (Ugh.)

.. _`V8's source code`: (https://github.com/v8/v8/blob/master/benchmarks/deltablue.js)
"""
from __future__ import annotations
import __static__
import time
from enum import IntEnum
from __static__ import CheckedList, box, cast, cbool, clen, int64, inline
from typing import final
import time

@inline
def stronger(_s1, _s2):
    return box(cbool(cast(Strength, _s1).strength < cast(Strength, _s2).strength))

@inline
def weaker(_s1, _s2):
    return box(cbool(cast(Strength, _s1).strength > cast(Strength, _s2).strength))

@inline
def weakest_of(_s1, _s2):
    return cast(Strength, _s1) if cast(Strength, _s1).strength > cast(Strength, _s2).strength else cast(Strength, _s2)

@final
class Strength:

    def __init__(self, strength: int64, name: str) -> None:
        self.strength: int64 = strength
        self.name: str = name

    def next_weaker(self):
        return STRENGTHS[self.strength]
REQUIRED = Strength(0, 'required')
STRONG_PREFERRED = Strength(1, 'strongPreferred')
PREFERRED = Strength(2, 'preferred')
STRONG_DEFAULT = Strength(3, 'strongDefault')
NORMAL = Strength(4, 'normal')
WEAK_DEFAULT = Strength(5, 'weakDefault')
WEAKEST = Strength(6, 'weakest')
STRENGTHS: CheckedList[Strength] = CheckedList[Strength]([WEAKEST, WEAK_DEFAULT, NORMAL, STRONG_DEFAULT, PREFERRED, REQUIRED])

class Constraint(object):

    def __init__(self, strength: Strength) -> None:
        self.strength: Strength = strength

    def add_constraint(self) -> None:
        planner = cast(Planner, get_planner())
        self.add_to_graph()
        planner.incremental_add(cast(Constraint, self))

    def satisfy(self, _mark):
        mark: int64 = int64(_mark)
        planner = cast(Planner, get_planner())
        self.choose_method(box(mark))
        if not self.is_satisfied():
            if self.strength == REQUIRED:
                print('Could not satisfy a required constraint!')
            return None
        self.mark_inputs(box(mark))
        out = cast(Variable, self.output())
        overridden = out.determined_by
        if overridden is not None:
            overridden.mark_unsatisfied()
        out.determined_by = self
        if not cbool(planner.add_propagate(cast(Constraint, self), box(mark))):
            print('Cycle encountered')
        out.mark = mark
        return overridden

    def destroy_constraint(self) -> None:
        planner = cast(Planner, get_planner())
        if self.is_satisfied():
            planner.incremental_remove(cast(Constraint, self))
        else:
            self.remove_from_graph()

    def is_input(self):
        return box(cbool(False))

    def mark_inputs(self, _mark) -> None:
        mark: int64 = int64(_mark)
        pass

    def inputs_known(self, _mark):
        mark: int64 = int64(_mark)
        return box(cbool(True))

    def choose_method(self, _mark) -> None:
        mark: int64 = int64(_mark)
        pass

    def output(self):
        raise NotImplementedError()

    def execute(self) -> None:
        pass

class UrnaryConstraint(Constraint):

    def __init__(self, v: Variable, strength: Strength) -> None:
        Constraint.__init__(self, strength)
        self.my_output: Variable = v
        self.satisfied: cbool = False
        self.add_constraint()

    def add_to_graph(self) -> None:
        self.my_output.add_constraint(self)
        self.satisfied = False

    def choose_method(self, _mark) -> None:
        mark: int64 = int64(_mark)
        if self.my_output.mark != mark and cbool(stronger(cast(Strength, self.strength), cast(Strength, self.my_output.walk_strength))):
            self.satisfied = True
        else:
            self.satisfied = False

    def is_satisfied(self):
        return box(cbool(self.satisfied))

    def output(self):
        return self.my_output

    def recalculate(self) -> None:
        self.my_output.walk_strength = self.strength
        self.my_output.stay = not cbool(self.is_input())
        if self.my_output.stay:
            self.execute()

    def mark_unsatisfied(self) -> None:
        self.satisfied = False

    def remove_from_graph(self) -> None:
        if self.my_output is not None:
            self.my_output.remove_constraint(cast(Constraint, self))
            self.satisfied = False

@final
class StayConstraint(UrnaryConstraint):
    pass

@final
class EditConstraint(UrnaryConstraint):

    def is_input(self):
        return box(cbool(True))

@final
class Direction(IntEnum):
    NONE = 0
    FORWARD = 1
    BACKWARD = -1

class BinaryConstraint(Constraint):

    def __init__(self, v1: Variable, v2: Variable, strength: Strength) -> None:
        Constraint.__init__(self, strength)
        self.v1: Variable = v1
        self.v2: Variable = v2
        self.direction: Direction = Direction.NONE
        self.add_constraint()

    def choose_method(self, _mark) -> None:
        mark: int64 = int64(_mark)
        if self.v1.mark == mark:
            if self.v2.mark != mark and cbool(stronger(cast(Strength, self.strength), cast(Strength, self.v2.walk_strength))):
                self.direction = Direction.FORWARD
            else:
                self.direction = Direction.BACKWARD
        if self.v2.mark == mark:
            if self.v1.mark != mark and cbool(stronger(cast(Strength, self.strength), cast(Strength, self.v1.walk_strength))):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        if cbool(weaker(cast(Strength, self.v1.walk_strength), cast(Strength, self.v2.walk_strength))):
            if cbool(stronger(cast(Strength, self.strength), cast(Strength, self.v1.walk_strength))):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        elif cbool(stronger(cast(Strength, self.strength), cast(Strength, self.v2.walk_strength))):
            self.direction = Direction.FORWARD
        else:
            self.direction = Direction.BACKWARD

    def add_to_graph(self) -> None:
        self.v1.add_constraint(self)
        self.v2.add_constraint(self)
        self.direction = Direction.NONE

    def is_satisfied(self):
        if self.direction != Direction.NONE:
            return box(cbool(True))
        return box(cbool(False))

    def mark_inputs(self, _mark) -> None:
        mark: int64 = int64(_mark)
        cast(Variable, self.input()).mark = mark

    def input(self):
        return self.v1 if self.direction == Direction.FORWARD else self.v2

    def output(self):
        return self.v2 if self.direction == Direction.FORWARD else self.v1

    def recalculate(self) -> None:
        ihn = cast(Variable, self.input())
        out = cast(Variable, self.output())
        out.walk_strength = cast(Strength, weakest_of(cast(Strength, self.strength), cast(Strength, ihn.walk_strength)))
        out.stay = ihn.stay
        if out.stay:
            self.execute()

    def mark_unsatisfied(self) -> None:
        self.direction = Direction.NONE

    def inputs_known(self, _mark):
        mark: int64 = int64(_mark)
        i = cast(Variable, self.input())
        return box(cbool(i.mark == mark or i.stay or cbool(i.determined_by is None)))

    def remove_from_graph(self):
        if self.v1 is not None:
            self.v1.remove_constraint(cast(Constraint, self))
        if self.v2 is not None:
            self.v2.remove_constraint(cast(Constraint, self))
        self.direction = Direction.NONE

@final
class ScaleConstraint(BinaryConstraint):

    def __init__(self, src: Variable, scale: Variable, offset: Variable, dest: Variable, strength: Strength) -> None:
        self.direction: Direction = Direction.NONE
        self.scale: Variable = scale
        self.offset: Variable = offset
        BinaryConstraint.__init__(self, src, dest, strength)

    def add_to_graph(self) -> None:
        BinaryConstraint.add_to_graph(self)
        self.scale.add_constraint(self)
        self.offset.add_constraint(self)

    def remove_from_graph(self):
        BinaryConstraint.remove_from_graph(self)
        if self.scale is not None:
            self.scale.remove_constraint(cast(Constraint, self))
        if self.offset is not None:
            self.offset.remove_constraint(cast(Constraint, self))

    def mark_inputs(self, _mark) -> None:
        mark: int64 = int64(_mark)
        BinaryConstraint.mark_inputs(self, box(mark))
        self.scale.mark = mark
        self.offset.mark = mark

    def execute(self) -> None:
        if self.direction == Direction.FORWARD:
            self.v2.value = self.v1.value * self.scale.value + self.offset.value
        else:
            self.v1.value = (self.v2.value - self.offset.value) / self.scale.value

    def recalculate(self) -> None:
        ihn = cast(Variable, self.input())
        out = cast(Variable, self.output())
        out.walk_strength = cast(Strength, weakest_of(cast(Strength, self.strength), cast(Strength, ihn.walk_strength)))
        out.stay = ihn.stay and self.scale.stay and self.offset.stay
        if out.stay:
            self.execute()

@final
class EqualityConstraint(BinaryConstraint):

    def execute(self) -> None:
        cast(Variable, self.output()).value = cast(Variable, self.input()).value

@final
class Variable(object):

    def __init__(self, name: str, initial_value: int64=0) -> None:
        self.name: str = name
        self.value: int64 = initial_value
        self.constraints: CheckedList[Constraint] = []
        self.determined_by: Constraint | None = None
        self.mark: int64 = 0
        self.walk_strength: Strength = WEAKEST
        self.stay: cbool = True

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def remove_constraint(self, _constraint) -> None:
        constraint: Constraint = cast(Constraint, _constraint)
        self.constraints.remove(constraint)
        if self.determined_by == constraint:
            self.determined_by = None

@final
class Planner(object):

    def __init__(self) -> None:
        self.current_mark: int64 = 0

    def incremental_add(self, _constraint) -> None:
        constraint: Constraint = cast(Constraint, _constraint)
        mark = int64(self.new_mark())
        overridden = cast(Constraint | None, constraint.satisfy(box(int64(mark))))
        while overridden is not None:
            overridden = cast(Constraint | None, overridden.satisfy(box(int64(mark))))

    def incremental_remove(self, _constraint) -> None:
        constraint: Constraint = cast(Constraint, _constraint)
        out = cast(Variable, constraint.output())
        constraint.mark_unsatisfied()
        constraint.remove_from_graph()
        unsatisfied = self.remove_propagate_from(out)
        strength = REQUIRED
        repeat = True
        while repeat:
            for u in unsatisfied:
                if u.strength == strength:
                    self.incremental_add(cast(Constraint, u))
                strength = cast(Strength, strength.next_weaker())
            repeat = strength != WEAKEST

    def new_mark(self):
        x = self.current_mark + 1
        self.current_mark = x
        return box(int64(self.current_mark))

    def make_plan(self, _sources):
        sources: CheckedList[UrnaryConstraint] = CheckedList[UrnaryConstraint](_sources)
        mark = int64(self.new_mark())
        plan = Plan()
        todo = CheckedList[Constraint]([s for s in sources])
        while clen(CheckedList[Constraint](todo)):
            c = CheckedList[Constraint](todo).pop(0)
            if cast(Variable, c.output()).mark != mark and cbool(c.inputs_known(box(int64(mark)))):
                plan.add_constraint(c)
                cast(Variable, c.output()).mark = mark
                self.add_constraints_consuming_to(cast(Variable, c.output()), CheckedList[Constraint](CheckedList[Constraint](todo)))
        return plan

    def extract_plan_from_constraints(self, _constraints):
        constraints: CheckedList[UrnaryConstraint] = CheckedList[UrnaryConstraint](_constraints)
        sources = CheckedList[UrnaryConstraint]([])
        for c in constraints:
            if cbool(c.is_input()) and cbool(c.is_satisfied()):
                CheckedList[UrnaryConstraint](sources).append(c)
        return cast(Plan, self.make_plan(CheckedList[UrnaryConstraint](CheckedList[UrnaryConstraint](sources))))

    def add_propagate(self, _c, _mark):
        mark: int64 = int64(_mark)
        c: Constraint = cast(Constraint, _c)
        todo = CheckedList[Constraint]([])
        CheckedList[Constraint](todo).append(c)
        while clen(CheckedList[Constraint](todo)):
            d = CheckedList[Constraint](todo).pop(0)
            if cast(Variable, d.output()).mark == mark:
                self.incremental_remove(cast(Constraint, c))
                return box(cbool(False))
            d.recalculate()
            self.add_constraints_consuming_to(cast(Variable, d.output()), CheckedList[Constraint](CheckedList[Constraint](todo)))
        return box(cbool(True))

    def remove_propagate_from(self, out: Variable) -> CheckedList[Constraint]:
        out.determined_by = None
        out.walk_strength = WEAKEST
        out.stay = True
        unsatisfied: CheckedList[Constraint] = []
        todo: CheckedList[Variable] = []
        todo.append(out)
        while len(todo):
            v = todo.pop(0)
            cs = v.constraints
            for c in cs:
                if not cbool(c.is_satisfied()):
                    unsatisfied.append(c)
            determining = v.determined_by
            for c in cs:
                if c != determining and cbool(c.is_satisfied()):
                    c.recalculate()
                    todo.append(cast(Variable, c.output()))
        return unsatisfied

    def add_constraints_consuming_to(self, _v, _coll) -> None:
        coll: CheckedList[Constraint] = CheckedList[Constraint](_coll)
        v: Variable = cast(Variable, _v)
        determining = v.determined_by
        cc = v.constraints
        for c in cc:
            if c != determining and cbool(c.is_satisfied()):
                coll.append(c)

@final
class Plan(object):

    def __init__(self) -> None:
        self.v: CheckedList[Constraint] = []

    def add_constraint(self, c: Constraint) -> None:
        self.v.append(c)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, index):
        return self.v[index]

    def execute(self) -> None:
        for c in self.v:
            c.execute()

def recreate_planner():
    global planner
    planner = Planner()
    return planner

def get_planner():
    global planner
    return planner

def chain_test(_n) -> None:
    n: int64 = int64(_n)
    '\n    This is the standard DeltaBlue benchmark. A long chain of equality\n    constraints is constructed with a stay constraint on one end. An\n    edit constraint is then added to the opposite end and the time is\n    measured for adding and removing this constraint, and extracting\n    and executing a constraint satisfaction plan. There are two cases.\n    In case 1, the added constraint is stronger than the stay\n    constraint and values must propagate down the entire length of the\n    chain. In case 2, the added constraint is weaker than the stay\n    constraint so it cannot be accomodated. The cost in this case is,\n    of course, very low. Typical situations lie somewhere between these\n    two extremes.\n    '
    planner = cast(Planner, recreate_planner())
    prev: Variable | None = None
    first: Variable | None = None
    last: Variable | None = None
    i = box(int64(0))
    end = box(int64(n + 1))
    while int64(i) < n + 1:
        name = 'v%s' % box(int64(i))
        v = Variable(name)
        if prev is not None:
            EqualityConstraint(prev, v, REQUIRED)
        if int64(i) == 0:
            first = v
        if int64(i) == n:
            last = v
        prev = v
        i = box(int64(int64(i) + 1))
    first = cast(Variable, first)
    last = cast(Variable, last)
    StayConstraint(last, STRONG_DEFAULT)
    edit = EditConstraint(first, PREFERRED)
    edits = CheckedList[UrnaryConstraint]([])
    CheckedList[UrnaryConstraint](edits).append(edit)
    plan = cast(Plan, planner.extract_plan_from_constraints(CheckedList[UrnaryConstraint](CheckedList[UrnaryConstraint](edits))))
    i = box(int64(0))
    while int64(i) < 100:
        first.value = int64(i)
        plan.execute()
        if last.value != int64(i):
            print('Chain test failed.')
        i = box(int64(int64(i) + 1))

def projection_test(_n) -> None:
    n: int64 = int64(_n)
    '\n    This test constructs a two sets of variables related to each\n    other by a simple linear transformation (scale and offset). The\n    time is measured to change a variable on either side of the\n    mapping and to change the scale and offset factors.\n    '
    planner = cast(Planner, recreate_planner())
    scale = Variable('scale', 10)
    offset = Variable('offset', 1000)
    src: Variable | None = None
    dests = CheckedList[Variable]([])
    i = box(int64(0))
    dst = Variable('dst%s' % 0, 0)
    while int64(i) < n:
        bi = box(int64(i))
        src = Variable('src%s' % bi, int64(i))
        dst = Variable('dst%s' % bi, int64(i))
        CheckedList[Variable](dests).append(dst)
        StayConstraint(src, NORMAL)
        ScaleConstraint(src, scale, offset, dst, REQUIRED)
        i = box(int64(int64(i) + 1))
    src = cast(Variable, src)
    change(cast(Variable, src), box(int64(17)))
    if dst.value != 1170:
        print('Projection 1 failed')
    change(cast(Variable, dst), box(int64(1050)))
    if src.value != 5:
        print('Projection 2 failed')
    change(cast(Variable, scale), box(int64(5)))
    i = box(int64(0))
    while int64(i) < n - 1:
        if CheckedList[Variable](dests)[int64(i)].value != int64(i) * 5 + 1000:
            print('Projection 3 failed')
        i = box(int64(int64(i) + 1))
    change(cast(Variable, offset), box(int64(2000)))
    i = box(int64(0))
    while int64(i) < n - 1:
        if CheckedList[Variable](dests)[int64(i)].value != int64(i) * 5 + 2000:
            print('Projection 4 failed')
        i = box(int64(int64(i) + 1))

def change(_v, _new_value) -> None:
    new_value: int64 = int64(_new_value)
    v: Variable = cast(Variable, _v)
    planner = cast(Planner, get_planner())
    edit = EditConstraint(v, PREFERRED)
    edits = CheckedList[UrnaryConstraint]([])
    CheckedList[UrnaryConstraint](edits).append(edit)
    plan = cast(Plan, planner.extract_plan_from_constraints(CheckedList[UrnaryConstraint](CheckedList[UrnaryConstraint](edits))))
    i = box(int64(0))
    while int64(i) < 10:
        v.value = new_value
        plan.execute()
        i = box(int64(int64(i) + 1))
    edit.destroy_constraint()
planner = None

def delta_blue(_i) -> None:
    i: int = cast(int, _i)
    n = int64(i)
    chain_test(box(int64(n)))
    projection_test(box(int64(n)))
if __name__ == '__main__':
    n = 10000
    startTime = time.time()
    delta_blue(cast(int, n))
    endTime = time.time()
    runtime = endTime - startTime
    print(runtime)
