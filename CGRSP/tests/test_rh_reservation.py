"""Load-gated tight-deadline reservation in the Rolling-Horizon scheduler.

RH reserves idle capacity for anticipated tight (rush) arrivals ONLY when the
offered load is below saturation; under saturation the gate disables reservation
so the saturated-load results are unchanged. Tight jobs may always use a slot;
loose jobs are withheld from the reserved slots.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node, NodeState
from cgrsp.config import ResourceType
from cgrsp.data.runpod_availability import StockStatus
from cgrsp.schedulers.rolling_horizon_scheduler import RollingHorizonScheduler


class FakeFleet:
    def __init__(self, idle, busy=None, stock=None):
        self._idle = list(idle)
        self._busy = list(busy or [])
        self._stock = stock or {}

    def get_idle_nodes(self):
        return list(self._idle)

    def get_all_nodes(self):
        return list(self._idle) + list(self._busy)

    def get_stock_status(self, rt):
        return self._stock.get(rt, StockStatus.HIGH)


def node(node_id, rt=ResourceType.RTX_A4000, state=NodeState.IDLE):
    return Node(node_id=node_id, resource_type=rt, state=state)


def job(job_id, complexity, deadline, submit_time=0.0, required_memory=None):
    return Job(job_id=job_id, complexity=complexity, deadline=deadline,
               submit_time=submit_time, estimated_duration=0.0,
               required_memory=required_memory)


def make_sched(idle, busy=None, stock=None, **kw):
    return RollingHorizonScheduler(fleet_manager=FakeFleet(idle, busy=busy, stock=stock),
                                   seed=0, **kw)


def _dispatched(out):
    return sum(v is not None for v in out.values())


SAT_RATE = 1.0      # far above saturation -> gate disables reservation
LOW_RATE = 0.001    # far below saturation -> reservation active
LOOSE = 100000.0    # loose deadline (window >= 2 h)
TIGHT = 3600.0      # tight deadline (1 h window < 2 h)


def test_no_reservation_under_saturation():
    # 2 idle slots, 2 loose jobs, saturated load: gate disables reservation,
    # so both loose jobs are dispatched now.
    s = make_sched(idle=[node(0), node(1)],
                   reservation_aggressiveness=1.0, arrival_rate=SAT_RATE)
    out = s.schedule([job(1, 1, LOOSE), job(2, 1, LOOSE)], 0.0)
    assert _dispatched(out) == 2


def test_reserves_one_slot_below_saturation():
    # Same jobs, low load: one slot is held for anticipated tight arrivals,
    # so only one loose job is dispatched now.
    s = make_sched(idle=[node(0), node(1)],
                   reservation_aggressiveness=1.0, arrival_rate=LOW_RATE)
    out = s.schedule([job(1, 1, LOOSE), job(2, 1, LOOSE)], 0.0)
    assert _dispatched(out) == 1


def test_aggressiveness_zero_dispatches_all_below_saturation():
    # Reservation disabled (aggressiveness 0): both loose jobs dispatched even
    # at low load. This is the byte-identical baseline behaviour.
    s = make_sched(idle=[node(0), node(1)],
                   reservation_aggressiveness=0.0, arrival_rate=LOW_RATE)
    out = s.schedule([job(1, 1, LOOSE), job(2, 1, LOOSE)], 0.0)
    assert _dispatched(out) == 2


def test_tight_job_uses_reserved_capacity():
    # 2 idle, reserve 1, low load: a tight job is dispatched; the loose job is
    # withheld so a slot stays available for tight demand.
    s = make_sched(idle=[node(0), node(1)],
                   reservation_aggressiveness=1.0, arrival_rate=LOW_RATE)
    out = s.schedule([job(1, 1, LOOSE), job(2, 1, TIGHT)], 0.0)
    assert out[2] is not None      # tight dispatched
    assert out[1] is None          # loose withheld for reservation
    assert _dispatched(out) == 1


def test_tight_committed_eagerly_when_anticipating():
    # 1 idle slot + a running slot freeing soon. A fresh tight job (longer exec,
    # complexity 3) and a shorter loose job (complexity 1). With anticipation OFF
    # the forward-sim defers the tight job (SPT prefers the shorter loose job);
    # with anticipation ON (below saturation) the tight job is committed eagerly.
    def build(aggr):
        s = RollingHorizonScheduler(
            fleet_manager=FakeFleet(idle=[node(0)],
                                    busy=[node(9, state=NodeState.RUNNING)]),
            seed=0, reservation_aggressiveness=aggr, arrival_rate=LOW_RATE)
        s._expected_free[9] = 500.0
        return s

    tight = job(1, 3, TIGHT)        # tight window, longer exec
    loose_short = job(2, 1, 80000.0)  # loose window, shorter exec

    off = build(0.0).schedule([tight, loose_short], 0.0)
    assert off[2] == 0      # SPT commits the shorter loose job
    assert off[1] is None   # tight deferred

    on = build(1.0).schedule([tight, loose_short], 0.0)
    assert on[1] == 0       # tight committed eagerly
    assert on[2] is None    # loose withheld


def test_spare_slots_beyond_reservation_serve_loose():
    # 3 idle, reserve 1, low load, 3 loose jobs: two are dispatched, one slot
    # is held in reserve.
    s = make_sched(idle=[node(0), node(1), node(2)],
                   reservation_aggressiveness=1.0, arrival_rate=LOW_RATE)
    out = s.schedule([job(1, 1, LOOSE), job(2, 1, LOOSE), job(3, 1, LOOSE)], 0.0)
    assert _dispatched(out) == 2
