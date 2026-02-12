"""
based on a Java version:
 Based on original version written in BCPL by Dr Martin Richards
 in 1981 at Cambridge University Computer Laboratory, England
 and a C++ version derived from a Smalltalk version written by
 L Peter Deutsch.
 Java version:  Copyright (C) 1995 Sun Microsystems, Inc.
 Translation from C++, Mario Wolczko
 Outer loop added by Alex Jacoby
"""
from __future__ import annotations
import sys
import __static__
from __static__ import cast, cbool, int64, box, inline
from typing import Final, Optional, List
from typing import Optional
import time
I_IDLE: Final[int] = 1
I_WORK: Final[int] = 2
I_HANDLERA: Final[int] = 3
I_HANDLERB: Final[int] = 4
I_DEVA: Final[int] = 5
I_DEVB: Final[int] = 6
K_DEV: Final[int] = 1000
K_WORK: Final[int] = 1001
BUFSIZE: Final[int] = 4
BUFSIZE_RANGE = range(BUFSIZE)

class Packet(object):

    def __init__(self, l: Optional[Packet], i: int64, k: int64) -> None:
        self.link: Optional[Packet] = l
        self.ident: int64 = i
        self.kind: int64 = k
        self.datum: int = 0
        self.data: List[int] = [0] * BUFSIZE

    def append_to(self, _lst):
        lst: Optional[Packet] = cast(Optional[Packet], _lst)
        self.link = None
        if lst is None:
            return self
        else:
            p = cast(Packet, lst)
            next: Optional[Packet] = cast(Packet, p).link
            while next is not None:
                p = cast(Packet, next)
                next = cast(Packet, p).link
            cast(Packet, p).link = self
            return lst

class TaskRec:
    pass

class DeviceTaskRec(TaskRec):

    def __init__(self) -> None:
        self.pending: Optional[Packet] = None

class IdleTaskRec(TaskRec):

    def __init__(self) -> None:
        self.control: int64 = 1
        self.count: int64 = 10000

class HandlerTaskRec(TaskRec):

    def __init__(self) -> None:
        self.work_in: Optional[Packet] = None
        self.device_in: Optional[Packet] = None

    def workInAdd(self, _p):
        p: Packet = cast(Packet, _p)
        self.work_in = cast(Packet, p.append_to(cast(Optional[Packet], self.work_in)))
        return self.work_in

    def deviceInAdd(self, _p):
        p: Packet = cast(Packet, _p)
        self.device_in = cast(Packet, p.append_to(cast(Optional[Packet], self.device_in)))
        return self.device_in

class WorkerTaskRec(TaskRec):

    def __init__(self) -> None:
        self.destination: int64 = int64(I_HANDLERA)
        self.count: int64 = 0

class TaskState(object):

    def __init__(self) -> None:
        self.packet_pending: cbool = True
        self.task_waiting: cbool = False
        self.task_holding: cbool = False

    def packetPending(self):
        self.packet_pending = True
        self.task_waiting = False
        self.task_holding = False
        return self

    def waiting(self):
        self.packet_pending = False
        self.task_waiting = True
        self.task_holding = False
        return self

    def running(self):
        self.packet_pending = False
        self.task_waiting = False
        self.task_holding = False
        return self

    def waitingWithPacket(self):
        self.packet_pending = True
        self.task_waiting = True
        self.task_holding = False
        return self

    @inline
    def isPacketPending(self):
        return box(cbool(self.packet_pending))

    @inline
    def isTaskWaiting(self):
        return box(cbool(self.task_waiting))

    @inline
    def isTaskHolding(self):
        return box(cbool(self.task_holding))

    @inline
    def isTaskHoldingOrWaiting(self):
        return box(cbool(self.task_holding or (not self.packet_pending and self.task_waiting)))

    @inline
    def isWaitingWithPacket(self):
        return box(cbool(self.packet_pending and self.task_waiting and (not self.task_holding)))
tracing = False
layout = 0

def trace(a):
    global layout
    layout -= 1
    if layout <= 0:
        print()
        layout = 50
    print(a, end='')
TASKTABSIZE = 10

class TaskWorkArea(object):

    def __init__(self) -> None:
        self.taskTab: List[Task] = [None] * TASKTABSIZE
        self.taskList: Optional[Task] = None
        self.holdCount: int64 = 0
        self.qpktCount: int64 = 0
taskWorkArea: TaskWorkArea = TaskWorkArea()

class Task(TaskState):

    def __init__(self, i: int64, p: int64, w: Optional[Packet], initialState: TaskState, r: TaskRec) -> None:
        wa: TaskWorkArea = taskWorkArea
        self.link: Optional[Task] = wa.taskList
        self.ident: int64 = i
        self.priority: int64 = p
        self.input: Optional[Packet] = w
        self.packet_pending = cbool(initialState.isPacketPending())
        self.task_waiting = cbool(initialState.isTaskWaiting())
        self.task_holding = cbool(initialState.isTaskHolding())
        self.handle = r
        wa.taskList = self
        wa.taskTab[i] = self

    def fn(self, _pkt, _r):
        pkt: Optional[Packet] = cast(Optional[Packet], _pkt)
        r: TaskRec = cast(TaskRec, _r)
        raise NotImplementedError

    def addPacket(self, _p, _old):
        p: Packet = cast(Packet, _p)
        old: Task = cast(Task, _old)
        if self.input is None:
            self.input = p
            self.packet_pending = True
            if self.priority > old.priority:
                return self
        else:
            cast(Packet, p.append_to(cast(Optional[Packet], self.input)))
        return old

    def runTask(self):
        if cbool(TaskState.isWaitingWithPacket(self)):
            msg: Optional[Packet] = self.input
            if msg is not None:
                self.input = msg.link
                if self.input is None:
                    cast(TaskState, self.running())
                else:
                    cast(TaskState, self.packetPending())
        else:
            msg = None
        return self.fn(cast(Optional[Packet], msg), cast(TaskRec, self.handle))

    def waitTask(self):
        self.task_waiting = True
        return self

    def hold(self):
        taskWorkArea.holdCount += 1
        self.task_holding = True
        return self.link

    def release(self, _i):
        i: int64 = int64(_i)
        t = cast(Task, Task.findtcb(self, box(i)))
        cast(Task, t).task_holding = False
        if cast(Task, t).priority > self.priority:
            return cast(Task, t)
        else:
            return self

    def qpkt(self, _pkt):
        pkt: Packet = cast(Packet, _pkt)
        t = cast(Task, Task.findtcb(self, box(int64(pkt.ident))))
        taskWorkArea.qpktCount += 1
        pkt.link = None
        pkt.ident = self.ident
        return cast(Task, cast(Task, t).addPacket(cast(Packet, pkt), cast(Task, self)))

    def findtcb(self, _id):
        id: int64 = int64(_id)
        t = taskWorkArea.taskTab[id]
        return cast(Task, t)

class DeviceTask(Task):

    def __init__(self, i: int64, p: int64, w: Optional[Packet], s: TaskState, r: DeviceTaskRec) -> None:
        Task.__init__(self, i, p, w, s, r)

    def fn(self, _pkt, _r):
        pkt: Optional[Packet] = cast(Optional[Packet], _pkt)
        r: TaskRec = cast(TaskRec, _r)
        d = cast(DeviceTaskRec, r)
        if pkt is None:
            pkt = cast(DeviceTaskRec, d).pending
            if pkt is None:
                return cast(Task, self.waitTask())
            else:
                cast(DeviceTaskRec, d).pending = None
                return cast(Task, self.qpkt(cast(Packet, pkt)))
        else:
            cast(DeviceTaskRec, d).pending = pkt
            if tracing:
                trace(pkt.datum)
            return cast(Task, cast(Optional[Task], self.hold()))

class HandlerTask(Task):

    def __init__(self, i: int64, p: int64, w: Packet, s: TaskState, r: HandlerTaskRec) -> None:
        Task.__init__(self, i, p, w, s, r)

    def fn(self, _pkt, _r):
        pkt: Optional[Packet] = cast(Optional[Packet], _pkt)
        r: TaskRec = cast(TaskRec, _r)
        h = cast(HandlerTaskRec, r)
        if pkt is not None:
            if pkt.kind == int64(K_WORK):
                cast(Optional[Packet], cast(HandlerTaskRec, h).workInAdd(cast(Packet, pkt)))
            else:
                cast(Optional[Packet], cast(HandlerTaskRec, h).deviceInAdd(cast(Packet, pkt)))
        work: Optional[Packet] = cast(HandlerTaskRec, h).work_in
        if work is None:
            return cast(Task, self.waitTask())
        count = cast(int, work.datum)
        if cast(int, count) >= BUFSIZE:
            cast(HandlerTaskRec, h).work_in = work.link
            return cast(Task, self.qpkt(cast(Packet, work)))
        dev: Optional[Packet] = cast(HandlerTaskRec, h).device_in
        if dev is None:
            return cast(Task, self.waitTask())
        cast(HandlerTaskRec, h).device_in = dev.link
        dev.datum = work.data[cast(int, count)]
        work.datum = cast(int, count) + 1
        return cast(Task, self.qpkt(cast(Packet, dev)))

class IdleTask(Task):

    def __init__(self, i: int64, p: int64, w: int, s: TaskState, r: IdleTaskRec) -> None:
        Task.__init__(self, i, 0, None, s, r)

    def fn(self, _pkt, _r):
        pkt: Optional[Packet] = cast(Optional[Packet], _pkt)
        r: TaskRec = cast(TaskRec, _r)
        i = cast(IdleTaskRec, r)
        cast(IdleTaskRec, i).count -= 1
        if cast(IdleTaskRec, i).count == 0:
            return cast(Optional[Task], self.hold())
        elif cast(IdleTaskRec, i).control & 1 == 0:
            cast(IdleTaskRec, i).control //= 2
            return cast(Task, Task.release(self, box(int64(I_DEVA))))
        else:
            cast(IdleTaskRec, i).control = cast(IdleTaskRec, i).control // 2 ^ 53256
            return cast(Task, Task.release(self, box(int64(I_DEVB))))
A: Final[int] = 65

class WorkTask(Task):

    def __init__(self, i: int64, p: int64, w: Packet, s: TaskState, r: WorkerTaskRec) -> None:
        Task.__init__(self, i, p, w, s, r)

    def fn(self, _pkt, _r):
        pkt: Optional[Packet] = cast(Optional[Packet], _pkt)
        r: TaskRec = cast(TaskRec, _r)
        w = cast(WorkerTaskRec, r)
        if pkt is None:
            return cast(Task, self.waitTask())
        if cast(WorkerTaskRec, w).destination == int64(I_HANDLERA):
            dest = box(int64(I_HANDLERB))
        else:
            dest = box(int64(I_HANDLERA))
        cast(WorkerTaskRec, w).destination = int64(dest)
        pkt.ident = int64(dest)
        pkt.datum = 0
        i = 0
        while i < BUFSIZE:
            x = box(int64(cast(WorkerTaskRec, w).count + 1))
            cast(WorkerTaskRec, w).count = int64(x)
            if cast(WorkerTaskRec, w).count > 26:
                cast(WorkerTaskRec, w).count = 1
            pkt.data[i] = A + box(cast(WorkerTaskRec, w).count) - 1
            i = i + 1
        return cast(Task, self.qpkt(cast(Packet, pkt)))

def schedule() -> None:
    t: Optional[Task] = taskWorkArea.taskList
    while t is not None:
        if tracing:
            print('tcb =', box(t.ident))
        if cbool(TaskState.isTaskHoldingOrWaiting(t)):
            t = t.link
        else:
            if tracing:
                trace(chr(ord('0') + box(t.ident)))
            t = cast(Optional[Task], t.runTask())

class Richards(object):

    def run(self, _iterations):
        iterations: int = cast(int, _iterations)
        for i in range(iterations):
            taskWorkArea.holdCount = 0
            taskWorkArea.qpktCount = 0
            IdleTask(int64(I_IDLE), 1, 10000, cast(TaskState, TaskState().running()), IdleTaskRec())
            wkq: Optional[Packet] = Packet(None, 0, int64(K_WORK))
            wkq = Packet(wkq, 0, int64(K_WORK))
            WorkTask(int64(I_WORK), 1000, wkq, cast(TaskState, TaskState().waitingWithPacket()), WorkerTaskRec())
            wkq = Packet(None, int64(I_DEVA), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVA), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVA), int64(K_DEV))
            HandlerTask(int64(I_HANDLERA), 2000, wkq, cast(TaskState, TaskState().waitingWithPacket()), HandlerTaskRec())
            wkq = Packet(None, int64(I_DEVB), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVB), int64(K_DEV))
            wkq = Packet(wkq, int64(I_DEVB), int64(K_DEV))
            HandlerTask(int64(I_HANDLERB), 3000, wkq, cast(TaskState, TaskState().waitingWithPacket()), HandlerTaskRec())
            wkq = None
            DeviceTask(int64(I_DEVA), 4000, wkq, cast(TaskState, TaskState().waiting()), DeviceTaskRec())
            DeviceTask(int64(I_DEVB), 5000, wkq, cast(TaskState, TaskState().waiting()), DeviceTaskRec())
            schedule()
            if taskWorkArea.holdCount == 9297 and taskWorkArea.qpktCount == 23246:
                pass
            else:
                print('err')
                return False
        return True
if __name__ == '__main__':
    num_iterations = 8
    if len(sys.argv) > 1:
        num_iterations = int(sys.argv[1])
    start_time = time.time()
    cast(bool, Richards().run(cast(int, num_iterations)))
    end_time = time.time()
    runtime = end_time - start_time
    print(runtime)
