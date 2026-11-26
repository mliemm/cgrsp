import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cgrsp.simulator.queue import Job
from cgrsp.simulator.fleet import Node, NodeState
from cgrsp.config import ResourceType
from cgrsp.data.runpod_availability import StockStatus
from cgrsp.schedulers.cost_aware_deadline_risk import CostAwareDeadlineRiskScheduler


class FakeFleet:
    def __init__(self, idle, stock=None):
        self._idle = list(idle)
        self._stock = stock or {}

    def get_idle_nodes(self):
        return list(self._idle)

    def get_stock_status(self, rt):
        return self._stock.get(rt, StockStatus.HIGH)


def node(node_id, rt=ResourceType.RTX_A4000, state=NodeState.IDLE):
    return Node(node_id=node_id, resource_type=rt, state=state)


def job(job_id, complexity, deadline, submit_time=0.0, required_memory=None):
    return Job(job_id=job_id, complexity=complexity, deadline=deadline,
               submit_time=submit_time, estimated_duration=0.0,
               required_memory=required_memory)


def make_sched(idle, stock=None, **kw):
    return CostAwareDeadlineRiskScheduler(
        fleet_manager=FakeFleet(idle, stock=stock), seed=0, **kw)


def test_no_idle_nodes_all_wait():
    s = make_sched(idle=[])
    assert s.schedule([job(1, 1, 100000.0)], 0.0) == {1: None}


def test_safe_jobs_ordered_shortest_first():
    # 1 idle node, two loose (low-risk) jobs: shorter exec dispatched now.
    s = make_sched(idle=[node(0)])
    short = job(1, 1, 100000.0)   # low complexity -> shorter exec
    long_ = job(2, 3, 100000.0)   # high complexity -> longer exec
    out = s.schedule([long_, short], 0.0)
    assert out[1] == 0
    assert out[2] is None


def test_atrisk_beats_safe():
    # 1 idle node: an at-risk job (small CR but feasible) goes before a safe short job.
    s = make_sched(idle=[node(0)], cr_star=3.0)
    # low-complexity exec on A4000 ~75s; deadline 150s -> CR=2.0 (<=3) at-risk
    at_risk = job(1, 1, 150.0)
    safe = job(2, 1, 100000.0)
    out = s.schedule([safe, at_risk], 0.0)
    assert out[1] == 0
    assert out[2] is None


def test_doomed_demoted():
    # 1 idle node: doomed job (CR<=1) must not take the slot from a feasible one.
    s = make_sched(idle=[node(0)])
    doomed = job(1, 1, 5.0)         # cannot finish -> CR<1
    feasible = job(2, 1, 100000.0)
    out = s.schedule([doomed, feasible], 0.0)
    assert out[2] == 0
    assert out[1] is None


def test_cost_aware_picks_cheapest_feasible():
    # Two idle nodes, both deadline-feasible for a loose job:
    # RTX_3090 (fast, $0.46/hr) vs RTX_A4000 (slower, $0.25/hr).
    fast_pricey = node(0, rt=ResourceType.RTX_3090)
    slow_cheap = node(1, rt=ResourceType.RTX_A4000)
    s = make_sched(idle=[fast_pricey, slow_cheap], cost_aware=True)
    out = s.schedule([job(1, 1, 100000.0)], 0.0)
    assert out[1] == 1  # A4000 (cheapest feasible)


def test_order_only_picks_fastest():
    fast_pricey = node(0, rt=ResourceType.RTX_3090)
    slow_cheap = node(1, rt=ResourceType.RTX_A4000)
    s = make_sched(idle=[fast_pricey, slow_cheap], cost_aware=False)
    out = s.schedule([job(1, 1, 100000.0)], 0.0)
    assert out[1] == 0  # RTX_3090 (fastest)


def test_cost_aware_falls_back_to_fastest_when_infeasible():
    # Tight deadline only the fast node can meet -> pick fast even though pricey.
    fast_pricey = node(0, rt=ResourceType.RTX_3090)
    slow_cheap = node(1, rt=ResourceType.RTX_A4500)
    s = make_sched(idle=[fast_pricey, slow_cheap], cost_aware=True)
    # complexity 1: 3090 ~59.7s < A4500 ~62.2s; deadline 61s -> only 3090 feasible.
    out = s.schedule([job(1, 1, 61.0)], 0.0)
    assert out[1] == 0  # only 3090 is deadline-feasible


def test_importable_from_package():
    from cgrsp.schedulers import CostAwareDeadlineRiskScheduler as C
    assert C is CostAwareDeadlineRiskScheduler
