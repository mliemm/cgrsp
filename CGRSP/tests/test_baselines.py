import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cgrsp.config import ResourceType
from cgrsp.data.runpod_availability import StockStatus
from cgrsp.schedulers.bandit_scheduler import (
    ARMS,
    BanditScheduler,
    EVAL_SEEDS,
    TRAIN_SEEDS,
    assert_disjoint_seeds,
    episode_reward,
    update_q_table,
)
from cgrsp.schedulers.heft_lpt_scheduler import HEFTLPTScheduler
from cgrsp.schedulers.max_min_scheduler import MaxMinScheduler
from cgrsp.schedulers.min_min_scheduler import MinMinScheduler
from cgrsp.simulator.fleet import Node, NodeState
from cgrsp.simulator.queue import Job


class FakeFleet:
    """Minimal fleet double: idle slots, all slots, qualitative stock."""

    def __init__(self, idle, all_nodes=None, stock=None):
        self._idle = list(idle)
        self._all = list(all_nodes) if all_nodes is not None else list(idle)
        self._stock = stock or {}

    def get_idle_nodes(self):
        return list(self._idle)

    def get_all_nodes(self):
        return list(self._all)

    def get_stock_status(self, resource_type):
        return self._stock.get(resource_type, StockStatus.HIGH)


def node(node_id, rt=ResourceType.RTX_A4000, state=NodeState.IDLE):
    return Node(node_id=node_id, resource_type=rt, state=state)


def job(job_id, complexity, deadline=100000.0, submit_time=0.0, required_memory=None):
    return Job(job_id=job_id, complexity=complexity, deadline=deadline,
               submit_time=submit_time, estimated_duration=0.0,
               required_memory=required_memory)


def make(scheduler_cls, idle, all_nodes=None, stock=None, **kwargs):
    fleet = FakeFleet(idle, all_nodes=all_nodes, stock=stock)
    return scheduler_cls(fleet, seed=0, **kwargs)


# Interface invariants

def test_interface_one_job_per_idle_node():
    jobs = [job(1, 1), job(2, 2), job(3, 3)]
    for scheduler_cls in (HEFTLPTScheduler, MinMinScheduler,
                          MaxMinScheduler, BanditScheduler):
        scheduler = make(scheduler_cls, [node(0), node(1)])
        assignments = scheduler.schedule(jobs, 0.0)
        placed = [node_id for node_id in assignments.values() if node_id is not None]
        assert len(placed) == 2
        assert len(set(placed)) == 2
        assert set(placed) == {0, 1}


def test_interface_every_queued_job_gets_a_key():
    jobs = [job(1, 1), job(2, 2), job(3, 3)]
    for scheduler_cls in (HEFTLPTScheduler, MinMinScheduler,
                          MaxMinScheduler, BanditScheduler):
        scheduler = make(scheduler_cls, [node(0)])
        assignments = scheduler.schedule(jobs, 0.0)
        assert set(assignments) == {1, 2, 3}
        assert sum(1 for value in assignments.values() if value is not None) == 1


def test_no_idle_nodes_keys_every_job_as_waiting():
    jobs = [job(1, 1), job(2, 2)]
    for scheduler_cls in (HEFTLPTScheduler, MinMinScheduler,
                          MaxMinScheduler, BanditScheduler):
        scheduler = make(scheduler_cls, [])
        assert scheduler.schedule(jobs, 0.0) == {1: None, 2: None}


def test_memory_filter_leaves_infeasible_job_queued():
    # The only slot is an A4000 (16 GB VRAM), so the 20 GB job cannot be placed.
    small = job(1, 1, required_memory=8.0)
    big = job(2, 1, required_memory=20.0)
    scheduler = make(HEFTLPTScheduler, [node(0, rt=ResourceType.RTX_A4000)])
    assignments = scheduler.schedule([small, big], 0.0)
    assert assignments[1] == 0
    assert assignments[2] is None


# HEFT / LPT

def test_lpt_sort_key_prefers_longest_average_job():
    # One idle slot: LPT must offer the high-complexity (longest mean) job first.
    # Reversing the sort key (or reusing SPT's ascending order) fails here.
    jobs = [job(1, 1, submit_time=0.0), job(2, 3, submit_time=0.0)]
    scheduler = make(HEFTLPTScheduler, [node(0)])
    assignments = scheduler.schedule(jobs, 0.0)
    assert assignments[2] == 0
    assert assignments[1] is None


def test_lpt_uses_fleet_mean_not_the_fastest_type():
    # Means over the four fleet types: kappa=1 -> 61.4 s, kappa=3 -> 93.2 s, so
    # the kappa=3 job ranks first even though the only idle slot is an A4500,
    # where kappa=1 (62.0 s) is faster than kappa=3 (88.3 s). An ascending
    # slot-time rule (SPT-like) would assign the kappa=1 job. A descending
    # slot-time rule happens to agree with the fleet-mean order on this
    # workload, so this test rules out the ascending mistake, not every
    # possible alternative key.
    idle = [node(0, rt=ResourceType.RTX_A4500)]
    fleet_nodes = [
        node(0, rt=ResourceType.RTX_A4500),
        node(1, rt=ResourceType.RTX_A5000),
        node(2, rt=ResourceType.RTX_3090),
        node(3, rt=ResourceType.RTX_A4000),
    ]
    scheduler = make(HEFTLPTScheduler, idle, all_nodes=fleet_nodes)
    jobs = [job(1, 1), job(2, 3)]
    assignments = scheduler.schedule(jobs, 0.0)
    assert assignments[2] == 0
    assert assignments[1] is None


def test_heft_picks_earliest_finish_node():
    # kappa=3: A4500 (88.3 s) finishes before 3090 (91.2 s) and A4000 (98.6 s).
    idle = [node(0, rt=ResourceType.RTX_A4000), node(1, rt=ResourceType.RTX_A4500)]
    scheduler = make(HEFTLPTScheduler, idle)
    assignments = scheduler.schedule([job(1, 3)], 0.0)
    assert assignments[1] == 1


# Min-Min / Max-Min

def test_min_min_argmin_prefers_shortest_earliest_completion():
    # One idle slot, short and long job: Min-Min picks the short one, Max-Min
    # the long one. Inverting the pick (or the sign of the completion key)
    # swaps these two assertions.
    jobs = [job(1, 3), job(2, 1)]
    min_min = make(MinMinScheduler, [node(0)])
    assert min_min.schedule(jobs, 0.0) == {1: None, 2: 0}

    max_min = make(MaxMinScheduler, [node(0)])
    assert max_min.schedule(jobs, 0.0) == {1: 0, 2: None}


def test_min_min_tie_break_is_job_id_then_node_id():
    # Equal complexity and equal node type: identical completion times, so the
    # lower job id takes the lower node id. A descending tie-break fails here.
    jobs = [job(1, 2), job(2, 2)]
    scheduler = make(MinMinScheduler, [node(0), node(1)])
    assert scheduler.schedule(jobs, 0.0) == {1: 0, 2: 1}


def test_max_min_tie_break_is_job_id_then_node_id():
    jobs = [job(1, 2), job(2, 2)]
    scheduler = make(MaxMinScheduler, [node(0), node(1)])
    assert scheduler.schedule(jobs, 0.0) == {1: 0, 2: 1}


def test_min_min_skips_infeasible_job():
    small = job(1, 1, required_memory=8.0)
    big = job(2, 1, required_memory=20.0)
    scheduler = make(MinMinScheduler, [node(0, rt=ResourceType.RTX_A4000)])
    assert scheduler.schedule([big, small], 0.0) == {1: 0, 2: None}


# Bandit

def test_bandit_frozen_table_selects_the_configured_arm():
    jobs = [
        job(1, 1, deadline=50000.0, submit_time=0.0),
        job(2, 1, deadline=40000.0, submit_time=10.0),
    ]
    fleet = FakeFleet([node(0)])

    empty = BanditScheduler(fleet, seed=0)
    assert empty.context(jobs, 0.0) == "q0|t0|s0"
    assert empty.schedule(jobs, 0.0) == {1: 0, 2: None}  # arm 0 = FIFO

    edf_table = {"q0|t0|s0": [0.0, 0.0, 1.0, 0.0]}
    trained = BanditScheduler(fleet, seed=0, q_table=edf_table, epsilon=0.0)
    assert trained.schedule(jobs, 0.0) == {1: None, 2: 0}  # arm 2 = EDF


def test_bandit_tie_break_in_table_is_lowest_arm():
    jobs = [job(1, 1, deadline=50000.0, submit_time=0.0),
            job(2, 1, deadline=40000.0, submit_time=10.0)]
    table = {"q0|t0|s0": [0.5, 0.5, 0.5, 0.5]}
    scheduler = BanditScheduler(FakeFleet([node(0)]), seed=0, q_table=table)
    assert scheduler.schedule(jobs, 0.0) == {1: 0, 2: None}


def test_bandit_random_control_explores_all_arms():
    scheduler = BanditScheduler(FakeFleet([node(0)]), seed=0, epsilon=1.0)
    jobs = [job(1, 1)]
    for _ in range(64):
        scheduler.schedule(jobs, 0.0)
    arms = {arm for _key, arm in scheduler.decision_log}
    assert arms == set(range(len(ARMS)))


def test_bandit_rejects_malformed_table_and_epsilon():
    import pytest
    with pytest.raises(ValueError):
        BanditScheduler(FakeFleet([node(0)]), seed=0, q_table={"q0|t0|s0": [0.0]})
    with pytest.raises(ValueError):
        BanditScheduler(FakeFleet([node(0)]), seed=0, epsilon=1.5)


def test_seed_ranges_are_disjoint():
    assert TRAIN_SEEDS == tuple(range(500, 515))
    assert EVAL_SEEDS == tuple(range(0, 30))
    assert set(TRAIN_SEEDS).isdisjoint(EVAL_SEEDS)
    assert_disjoint_seeds()
    import pytest
    with pytest.raises(ValueError):
        assert_disjoint_seeds([0, 1, 2], [2, 3])


def test_update_q_table_is_incremental_mean_with_one_credit_per_pair():
    q_table, counts = {}, {}
    update_q_table(q_table, counts, [("q0|t0|s0", 1), ("q0|t0|s0", 1)], -0.2)
    assert counts["q0|t0|s0"][1] == 1          # duplicate pair counted once
    assert q_table["q0|t0|s0"][1] == -0.2

    update_q_table(q_table, counts, [("q0|t0|s0", 1)], -0.6)
    assert counts["q0|t0|s0"][1] == 2
    assert abs(q_table["q0|t0|s0"][1] - (-0.4)) < 1e-12
    assert q_table["q0|t0|s0"][0] == 0.0      # untouched arm stays zero


def test_episode_reward_penalises_misses_and_wait():
    from cgrsp.simulator.metrics import JobMetrics

    class Metrics:
        def __init__(self, rows):
            self.rows = rows

        def get_job_metrics(self, job_id):
            return self.rows.get(job_id)

    on_time = JobMetrics(job_id=0, submit_time=0.0, deadline=1000.0)
    on_time.start_time = 100.0
    on_time.completion_time = 900.0
    late = JobMetrics(job_id=1, submit_time=0.0, deadline=1000.0)
    late.start_time = 100.0
    late.completion_time = 1200.0

    dispatch_log = [(0, 0, 0.0, 100.0), (1, 0, 0.0, 100.0)]
    reward = episode_reward(dispatch_log, Metrics({0: on_time, 1: late}).get_job_metrics)
    # miss rate 0.5, mean wait 100/60 min
    assert abs(reward - (-0.5 - 1e-5 * (100.0 / 60.0))) < 1e-12
    assert reward < 0.0
