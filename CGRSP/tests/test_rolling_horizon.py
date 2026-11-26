import os, sys
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


def test_no_idle_nodes_all_wait():
    s = make_sched(idle=[])
    assert s.schedule([job(1, 1, 100000.0)], 0.0) == {1: None}


def test_shortest_first_when_no_deadline_risk():
    # 1 idle node, 2 loose jobs; the shorter-exec job is committed now.
    s = make_sched(idle=[node(0)])
    short = job(1, 1, 100000.0)   # low complexity -> shorter exec
    long_ = job(2, 3, 100000.0)   # high complexity -> longer exec
    out = s.schedule([long_, short], 0.0)
    assert out[1] == 0
    assert out[2] is None


def test_hopeless_demoted():
    # 1 idle node; an infeasible job (deadline already unreachable) must not take
    # the slot from a feasible one.
    s = make_sched(idle=[node(0)])
    hopeless = job(1, 1, 5.0)
    feasible = job(2, 1, 100000.0)
    out = s.schedule([hopeless, feasible], 0.0)
    assert out[2] == 0
    assert out[1] is None


def test_critical_promoted_over_shorter_when_one_slot():
    # 1 idle + 1 running slot freeing far away; a near-deadline job is committed
    # ahead of a shorter loose job.
    s = make_sched(idle=[node(0)], busy=[node(9, state=NodeState.RUNNING)])
    s._expected_free[9] = 100000.0
    near = job(1, 3, 200.0)            # high complexity, tight feasible deadline
    loose_short = job(2, 1, 100000.0)
    out = s.schedule([loose_short, near], 0.0)
    assert out[1] == 0
    assert out[2] is None


def test_memory_incompatible_job_is_not_dispatched():
    # A job needing more VRAM than the node has is never dispatched; a feasible
    # job wins the slot.
    s = make_sched(idle=[node(0)])
    incompatible = job(1, 1, 100000.0, required_memory=100000.0)
    out_alone = s.schedule([incompatible], 0.0)
    assert out_alone[1] is None
    s2 = make_sched(idle=[node(0)])
    feasible = job(2, 1, 100000.0)
    out_pair = s2.schedule([incompatible, feasible], 0.0)
    assert out_pair[2] == 0
    assert out_pair[1] is None


def test_node_selection_prefers_high_stock_when_close():
    # Two capable nodes, near-equal exec; the LOW-stock one is penalised.
    n_high = node(0, rt=ResourceType.RTX_A4000)
    n_low = node(1, rt=ResourceType.RTX_A4500)
    s = make_sched(idle=[n_high, n_low],
                   stock={ResourceType.RTX_A4000: StockStatus.HIGH,
                          ResourceType.RTX_A4500: StockStatus.LOW})
    out = s.schedule([job(1, 1, 100000.0)], 0.0)
    assert out[1] == 0  # HIGH-stock node preferred


def test_no_anticipation_when_aggressiveness_zero():
    # 2 idle + a busy slot freeing far; with anticipation off both loose jobs go.
    s = make_sched(idle=[node(0), node(1)], busy=[node(9, state=NodeState.RUNNING)],
                   reservation_aggressiveness=0.0, arrival_rate=0.001)
    s._expected_free[9] = 100000.0
    out = s.schedule([job(1, 1, 100000.0), job(2, 1, 100000.0)], 0.0)
    assert _dispatched(out) == 2


def test_no_anticipation_under_saturation():
    # Below the gate the load is saturated (high arrival rate), so anticipation is
    # disabled even with a positive aggressiveness: both loose jobs dispatch.
    s = make_sched(idle=[node(0), node(1)], busy=[node(9, state=NodeState.RUNNING)],
                   reservation_aggressiveness=1.0, arrival_rate=0.1)
    s._expected_free[9] = 100000.0
    out = s.schedule([job(1, 1, 100000.0), job(2, 1, 100000.0)], 0.0)
    assert _dispatched(out) == 2


def test_anticipation_reserves_capacity_below_saturation():
    # Same setup but below saturation (low arrival rate): one slot is held for
    # anticipated tight arrivals, so fewer loose jobs dispatch now.
    s = make_sched(idle=[node(0), node(1)], busy=[node(9, state=NodeState.RUNNING)],
                   reservation_aggressiveness=1.0, arrival_rate=0.001)
    s._expected_free[9] = 100000.0
    out = s.schedule([job(1, 1, 100000.0), job(2, 1, 100000.0)], 0.0)
    assert _dispatched(out) < 2
    # but never reserves everything: at least one job still dispatched
    assert _dispatched(out) >= 1


def test_normal_tier_uses_spt_not_edf():
    # 1 idle slot + a running slot freeing soon (so both jobs are NON-urgent).
    # Among non-urgent jobs the SHORTER one wins the slot even though it has the
    # LATER deadline (SPT, not EDF).
    # complexity 3 -> ~98.2s exec; complexity 1 -> ~60s exec on RTX_A4000.
    # next_free = 500s; 500 + 98.2 = 598.2 < 50000 and 500 + 60 = 560 < 80000
    # => both jobs are non-urgent, so normal SPT tier applies.
    s = make_sched(idle=[node(0)], busy=[node(9, state=NodeState.RUNNING)])
    s._expected_free[9] = 500.0
    long_early = job(1, 3, 50000.0)   # long exec (complexity 3), earlier deadline
    short_late = job(2, 1, 80000.0)   # short exec (complexity 1), later deadline
    out = s.schedule([long_early, short_late], 0.0)
    assert out[2] == 0      # shorter job (SPT) committed
    assert out[1] is None
