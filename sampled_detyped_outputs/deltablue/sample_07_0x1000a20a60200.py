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
# detyper-status: types_kept
def stronger(s1: Strength, s2: Strength) -> cbool:
    return s1.strength < s2.strength

@inline
# detyper-status: types_kept
def weaker(s1: Strength, s2: Strength) -> cbool:
    return s1.strength > s2.strength

@inline
# detyper-status: types_kept
def weakest_of(s1: Strength, s2: Strength) -> Strength:
    return s1 if s1.strength > s2.strength else s2

@final
class Strength:

    # detyper-status: types_kept
    def __init__(self, strength: int64, name: str) -> None:
        self.strength: int64 = strength
        self.name: str = name

    # detyper-status: types_kept
    def next_weaker(self) -> Strength:
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

    # detyper-status: types_kept
    def __init__(self, strength: Strength) -> None:
        self.strength: Strength = strength

    # detyper-status: types_kept
    def add_constraint(self) -> None:
        planner = get_planner()
        self.add_to_graph()
        planner.incremental_add(cast(Constraint, self))

    # detyper-status: types_kept
    def satisfy(self, mark: int64) -> Constraint | None:
        planner = get_planner()
        self.choose_method(box(mark))
        if not self.is_satisfied():
            if self.strength == REQUIRED:
                print('Could not satisfy a required constraint!')
            return None
        self.mark_inputs(mark)
        out = self.output()
        overridden = out.determined_by
        if overridden is not None:
            overridden.mark_unsatisfied()
        out.determined_by = self
        if not planner.add_propagate(self, mark):
            print('Cycle encountered')
        out.mark = mark
        return overridden

    # detyper-status: types_kept
    def destroy_constraint(self) -> None:
        planner = get_planner()
        if self.is_satisfied():
            planner.incremental_remove(self)
        else:
            self.remove_from_graph()

    # detyper-status: types_kept
    def is_input(self) -> cbool:
        return False

    # detyper-status: types_kept
    def mark_inputs(self, mark: int64) -> None:
        pass

    # detyper-status: types_kept
    def inputs_known(self, mark: int64) -> cbool:
        return True

    # detyper-status: types_removed
    def choose_method(self, _mark) -> None:
        mark: int64 = int64(_mark)
        pass

    # detyper-status: types_kept
    def output(self) -> Variable:
        raise NotImplementedError()

    # detyper-status: types_kept
    def execute(self) -> None:
        pass

class UrnaryConstraint(Constraint):

    # detyper-status: types_kept
    def __init__(self, v: Variable, strength: Strength) -> None:
        Constraint.__init__(self, strength)
        self.my_output: Variable = v
        self.satisfied: cbool = False
        self.add_constraint()

    # detyper-status: types_kept
    def add_to_graph(self) -> None:
        self.my_output.add_constraint(self)
        self.satisfied = False

    # detyper-status: types_removed
    def choose_method(self, _mark) -> None:
        mark: int64 = int64(_mark)
        if self.my_output.mark != mark and stronger(self.strength, self.my_output.walk_strength):
            self.satisfied = True
        else:
            self.satisfied = False

    # detyper-status: types_kept
    def is_satisfied(self) -> cbool:
        return self.satisfied

    # detyper-status: types_kept
    def output(self) -> Variable:
        return self.my_output

    # detyper-status: types_kept
    def recalculate(self) -> None:
        self.my_output.walk_strength = self.strength
        self.my_output.stay = not self.is_input()
        if self.my_output.stay:
            self.execute()

    # detyper-status: types_removed
    def mark_unsatisfied(self) -> None:
        self.satisfied = False

    # detyper-status: types_kept
    def remove_from_graph(self) -> None:
        if self.my_output is not None:
            self.my_output.remove_constraint(self)
            self.satisfied = False

@final
class StayConstraint(UrnaryConstraint):
    pass

@final
class EditConstraint(UrnaryConstraint):

    # detyper-status: types_kept
    def is_input(self) -> cbool:
        return True

@final
class Direction(IntEnum):
    NONE = 0
    FORWARD = 1
    BACKWARD = -1

class BinaryConstraint(Constraint):

    # detyper-status: types_kept
    def __init__(self, v1: Variable, v2: Variable, strength: Strength) -> None:
        Constraint.__init__(self, strength)
        self.v1: Variable = v1
        self.v2: Variable = v2
        self.direction: Direction = Direction.NONE
        self.add_constraint()

    # detyper-status: types_removed
    def choose_method(self, _mark) -> None:
        mark: int64 = int64(_mark)
        if self.v1.mark == mark:
            if self.v2.mark != mark and stronger(self.strength, self.v2.walk_strength):
                self.direction = Direction.FORWARD
            else:
                self.direction = Direction.BACKWARD
        if self.v2.mark == mark:
            if self.v1.mark != mark and stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        if weaker(self.v1.walk_strength, self.v2.walk_strength):
            if stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        elif stronger(self.strength, self.v2.walk_strength):
            self.direction = Direction.FORWARD
        else:
            self.direction = Direction.BACKWARD

    # detyper-status: types_kept
    def add_to_graph(self) -> None:
        self.v1.add_constraint(self)
        self.v2.add_constraint(self)
        self.direction = Direction.NONE

    # detyper-status: types_kept
    def is_satisfied(self) -> cbool:
        if self.direction != Direction.NONE:
            return True
        return False

    # detyper-status: types_kept
    def mark_inputs(self, mark: int64) -> None:
        self.input().mark = mark

    # detyper-status: types_kept
    def input(self) -> Variable:
        return self.v1 if self.direction == Direction.FORWARD else self.v2

    # detyper-status: types_kept
    def output(self) -> Variable:
        return self.v2 if self.direction == Direction.FORWARD else self.v1

    # detyper-status: types_kept
    def recalculate(self) -> None:
        ihn = self.input()
        out = self.output()
        out.walk_strength = weakest_of(self.strength, ihn.walk_strength)
        out.stay = ihn.stay
        if out.stay:
            self.execute()

    # detyper-status: types_removed
    def mark_unsatisfied(self) -> None:
        self.direction = Direction.NONE

    # detyper-status: types_kept
    def inputs_known(self, mark: int64) -> cbool:
        i = self.input()
        return i.mark == mark or i.stay or cbool(i.determined_by is None)

    # detyper-status: types_kept
    def remove_from_graph(self):
        if self.v1 is not None:
            self.v1.remove_constraint(self)
        if self.v2 is not None:
            self.v2.remove_constraint(self)
        self.direction = Direction.NONE

@final
class ScaleConstraint(BinaryConstraint):

    # detyper-status: types_kept
    def __init__(self, src: Variable, scale: Variable, offset: Variable, dest: Variable, strength: Strength) -> None:
        self.direction: Direction = Direction.NONE
        self.scale: Variable = scale
        self.offset: Variable = offset
        BinaryConstraint.__init__(self, src, dest, strength)

    # detyper-status: types_kept
    def add_to_graph(self) -> None:
        BinaryConstraint.add_to_graph(self)
        self.scale.add_constraint(self)
        self.offset.add_constraint(self)

    # detyper-status: types_kept
    def remove_from_graph(self):
        BinaryConstraint.remove_from_graph(self)
        if self.scale is not None:
            self.scale.remove_constraint(self)
        if self.offset is not None:
            self.offset.remove_constraint(self)

    # detyper-status: types_kept
    def mark_inputs(self, mark: int64) -> None:
        BinaryConstraint.mark_inputs(self, mark)
        self.scale.mark = mark
        self.offset.mark = mark

    # detyper-status: types_kept
    def execute(self) -> None:
        if self.direction == Direction.FORWARD:
            self.v2.value = self.v1.value * self.scale.value + self.offset.value
        else:
            self.v1.value = (self.v2.value - self.offset.value) / self.scale.value

    # detyper-status: types_kept
    def recalculate(self) -> None:
        ihn = self.input()
        out = self.output()
        out.walk_strength = weakest_of(self.strength, ihn.walk_strength)
        out.stay = ihn.stay and self.scale.stay and self.offset.stay
        if out.stay:
            self.execute()

@final
class EqualityConstraint(BinaryConstraint):

    # detyper-status: types_kept
    def execute(self) -> None:
        self.output().value = self.input().value

@final
class Variable(object):

    # detyper-status: types_kept
    def __init__(self, name: str, initial_value: int64=0) -> None:
        self.name: str = name
        self.value: int64 = initial_value
        self.constraints: CheckedList[Constraint] = []
        self.determined_by: Constraint | None = None
        self.mark: int64 = 0
        self.walk_strength: Strength = WEAKEST
        self.stay: cbool = True

    # detyper-status: types_kept
    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    # detyper-status: types_kept
    def remove_constraint(self, constraint: Constraint) -> None:
        self.constraints.remove(constraint)
        if self.determined_by == constraint:
            self.determined_by = None

@final
class Planner(object):

    # detyper-status: types_kept
    def __init__(self) -> None:
        self.current_mark: int64 = 0

    # detyper-status: types_removed
    def incremental_add(self, _constraint) -> None:
        constraint: Constraint = cast(Constraint, _constraint)
        mark = self.new_mark()
        overridden = constraint.satisfy(mark)
        while overridden is not None:
            overridden = overridden.satisfy(mark)

    # detyper-status: types_kept
    def incremental_remove(self, constraint: Constraint) -> None:
        out = constraint.output()
        constraint.mark_unsatisfied()
        constraint.remove_from_graph()
        unsatisfied = self.remove_propagate_from(out)
        strength = REQUIRED
        repeat = True
        while repeat:
            for u in unsatisfied:
                if u.strength == strength:
                    self.incremental_add(cast(Constraint, u))
                strength = strength.next_weaker()
            repeat = strength != WEAKEST

    # detyper-status: types_kept
    def new_mark(self) -> int64:
        x = self.current_mark + 1
        self.current_mark = x
        return self.current_mark

    # detyper-status: types_kept
    def make_plan(self, sources: CheckedList[UrnaryConstraint]) -> Plan:
        mark = self.new_mark()
        plan = Plan()
        todo: CheckedList[Constraint] = [s for s in sources]
        while clen(todo):
            c = todo.pop(0)
            if c.output().mark != mark and c.inputs_known(mark):
                plan.add_constraint(c)
                c.output().mark = mark
                self.add_constraints_consuming_to(c.output(), todo)
        return plan

    # detyper-status: types_kept
    def extract_plan_from_constraints(self, constraints: CheckedList[UrnaryConstraint]) -> Plan:
        sources: CheckedList[UrnaryConstraint] = []
        for c in constraints:
            if c.is_input() and c.is_satisfied():
                sources.append(c)
        return self.make_plan(sources)

    # detyper-status: types_kept
    def add_propagate(self, c: Constraint, mark: int64) -> cbool:
        todo: CheckedList[Constraint] = []
        todo.append(c)
        while clen(todo):
            d = todo.pop(0)
            if d.output().mark == mark:
                self.incremental_remove(c)
                return False
            d.recalculate()
            self.add_constraints_consuming_to(d.output(), todo)
        return True

    # detyper-status: types_kept
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
                if not c.is_satisfied():
                    unsatisfied.append(c)
            determining = v.determined_by
            for c in cs:
                if c != determining and c.is_satisfied():
                    c.recalculate()
                    todo.append(c.output())
        return unsatisfied

    # detyper-status: types_kept
    def add_constraints_consuming_to(self, v: Variable, coll: CheckedList[Constraint]) -> None:
        determining = v.determined_by
        cc = v.constraints
        for c in cc:
            if c != determining and c.is_satisfied():
                coll.append(c)

@final
class Plan(object):

    # detyper-status: types_kept
    def __init__(self) -> None:
        self.v: CheckedList[Constraint] = []

    # detyper-status: types_kept
    def add_constraint(self, c: Constraint) -> None:
        self.v.append(c)

    # detyper-status: types_kept
    def __len__(self):
        return len(self.v)

    # detyper-status: types_kept
    def __getitem__(self, index):
        return self.v[index]

    # detyper-status: types_kept
    def execute(self) -> None:
        for c in self.v:
            c.execute()

# detyper-status: types_kept
def recreate_planner() -> Planner:
    global planner
    planner = Planner()
    return planner

# detyper-status: types_kept
def get_planner() -> Planner:
    global planner
    return planner

# detyper-status: types_kept
def chain_test(n: int64) -> None:
    """
    This is the standard DeltaBlue benchmark. A long chain of equality
    constraints is constructed with a stay constraint on one end. An
    edit constraint is then added to the opposite end and the time is
    measured for adding and removing this constraint, and extracting
    and executing a constraint satisfaction plan. There are two cases.
    In case 1, the added constraint is stronger than the stay
    constraint and values must propagate down the entire length of the
    chain. In case 2, the added constraint is weaker than the stay
    constraint so it cannot be accomodated. The cost in this case is,
    of course, very low. Typical situations lie somewhere between these
    two extremes.
    """
    planner = recreate_planner()
    prev: Variable | None = None
    first: Variable | None = None
    last: Variable | None = None
    i: int64 = 0
    end: int64 = n + 1
    while i < n + 1:
        name = 'v%s' % box(i)
        v = Variable(name)
        if prev is not None:
            EqualityConstraint(prev, v, REQUIRED)
        if i == 0:
            first = v
        if i == n:
            last = v
        prev = v
        i = i + 1
    first = cast(Variable, first)
    last = cast(Variable, last)
    StayConstraint(last, STRONG_DEFAULT)
    edit = EditConstraint(first, PREFERRED)
    edits: CheckedList[UrnaryConstraint] = []
    edits.append(edit)
    plan = planner.extract_plan_from_constraints(edits)
    i = 0
    while i < 100:
        first.value = i
        plan.execute()
        if last.value != i:
            print('Chain test failed.')
        i = i + 1

# detyper-status: types_kept
def projection_test(n: int64) -> None:
    """
    This test constructs a two sets of variables related to each
    other by a simple linear transformation (scale and offset). The
    time is measured to change a variable on either side of the
    mapping and to change the scale and offset factors.
    """
    planner = recreate_planner()
    scale = Variable('scale', 10)
    offset = Variable('offset', 1000)
    src: Variable | None = None
    dests: CheckedList[Variable] = []
    i: int64 = 0
    dst = Variable('dst%s' % 0, 0)
    while i < n:
        bi = box(i)
        src = Variable('src%s' % bi, i)
        dst = Variable('dst%s' % bi, i)
        dests.append(dst)
        StayConstraint(src, NORMAL)
        ScaleConstraint(src, scale, offset, dst, REQUIRED)
        i = i + 1
    src = cast(Variable, src)
    change(src, 17)
    if dst.value != 1170:
        print('Projection 1 failed')
    change(dst, 1050)
    if src.value != 5:
        print('Projection 2 failed')
    change(scale, 5)
    i = 0
    while i < n - 1:
        if dests[i].value != i * 5 + 1000:
            print('Projection 3 failed')
        i = i + 1
    change(offset, 2000)
    i = 0
    while i < n - 1:
        if dests[i].value != i * 5 + 2000:
            print('Projection 4 failed')
        i = i + 1

# detyper-status: types_kept
def change(v: Variable, new_value: int64) -> None:
    planner = get_planner()
    edit = EditConstraint(v, PREFERRED)
    edits: CheckedList[UrnaryConstraint] = []
    edits.append(edit)
    plan = planner.extract_plan_from_constraints(edits)
    i: int64 = 0
    while i < 10:
        v.value = new_value
        plan.execute()
        i = i + 1
    edit.destroy_constraint()
planner = None

# detyper-status: types_kept
def delta_blue(i: int) -> None:
    n = int64(i)
    chain_test(n)
    projection_test(n)
if __name__ == '__main__':
    n = 10000
    startTime = time.time()
    delta_blue(n)
    endTime = time.time()
    runtime = endTime - startTime
    print(runtime)
