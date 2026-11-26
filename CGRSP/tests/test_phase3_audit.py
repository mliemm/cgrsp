"""Adversarial audit tests for phase3_capacity_bound.py and
phase3_beta_sensitivity.py (audit-only; safe to delete)."""
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest

CGRSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CGRSP))

import phase3_capacity_bound as cb  # noqa: E402


@pytest.fixture
def concurrency():
    """Restore module CONCURRENCY after each test."""
    orig = cb.CONCURRENCY
    yield
    cb.CONCURRENCY = orig


# ---------- brute-force reference (non-preemptive, jobs may be dropped) ----

def _seq_feasible(group):
    if not group:
        return True
    for perm in itertools.permutations(group):
        t, ok = 0.0, True
        for (r, d, w) in perm:
            t = max(t, r) + w
            if t > d + 1e-9:
                ok = False
                break
        if ok:
            return True
    return False


def _subset_feasible(S, C):
    if not S:
        return True
    for assign in itertools.product(range(C), repeat=len(S)):
        groups = [[S[k] for k in range(len(S)) if assign[k] == m] for m in range(C)]
        if all(_seq_feasible(g) for g in groups):
            return True
    return False


def brute_min_misses(jobs, C):
    n = len(jobs)
    best = 0
    for mask in range(1 << n):
        S = [jobs[k] for k in range(n) if (mask >> k) & 1]
        if len(S) <= best:
            continue
        if _subset_feasible(S, C):
            best = len(S)
    return n - best


# ---------- capacity_bound hand instances ---------------------------------

def test_bound_two_identical_jobs_single_machine(concurrency):
    cb.CONCURRENCY = 1
    jobs = [(0.0, 10.0, 10.0), (0.0, 10.0, 10.0)]
    assert cb.capacity_bound(jobs) == 1
    assert cb.relaxed_edf_misses(jobs) == 1
    assert brute_min_misses(jobs, 1) == 1


def test_bound_zero_slack_single_job(concurrency):
    cb.CONCURRENCY = 1
    jobs = [(0.0, 10.0, 10.0)]
    assert cb.capacity_bound(jobs) == 0  # exact fit: no certified miss
    assert cb.relaxed_edf_misses(jobs) == 0  # finish == deadline is on time


def test_bound_deadline_beyond_all_others(concurrency):
    cb.CONCURRENCY = 1
    jobs = [(0.0, 10.0, 10.0), (0.0, 10.0, 10.0), (0.0, 1000.0, 10.0)]
    assert cb.capacity_bound(jobs) == 1
    assert cb.relaxed_edf_misses(jobs) == 1
    assert brute_min_misses(jobs, 1) == 1


def test_bound_multi_machine(concurrency):
    cb.CONCURRENCY = 2
    jobs = [(0.0, 10.0, 10.0)] * 3
    assert cb.capacity_bound(jobs) == 1
    assert cb.relaxed_edf_misses(jobs) == 1
    assert brute_min_misses(jobs, 2) == 1


def test_bound_needs_two_misses(concurrency):
    cb.CONCURRENCY = 1
    # 3 unit-window jobs of work 10 each due at 10: demand 30, cap 10,
    # slack 20, ceil(20/10) = 2 certified misses.
    jobs = [(0.0, 10.0, 10.0)] * 3
    assert cb.capacity_bound(jobs) == 2
    assert cb.relaxed_edf_misses(jobs) == 2
    assert brute_min_misses(jobs, 1) == 2


def test_bound_release_window_restriction(concurrency):
    cb.CONCURRENCY = 1
    # Job released late: only the window starting at its release certifies
    # a miss (demand 10 in [5, 10] > 5 capacity-seconds).
    jobs = [(0.0, 20.0, 5.0), (5.0, 10.0, 10.0)]
    assert cb.capacity_bound(jobs) == 1
    assert brute_min_misses(jobs, 1) == 1


# ---------- relaxed EDF hand instances ------------------------------------

def test_edf_three_jobs_one_machine_exactly_one_miss(concurrency):
    cb.CONCURRENCY = 1
    # EDF: job1 (d=10) runs 0-5; job2 (d=6, released 1) runs 5-10 -> miss;
    # job3 runs 10-15 <= 20. Optimum also misses exactly 1.
    jobs = [(0.0, 10.0, 5.0), (1.0, 6.0, 5.0), (2.0, 20.0, 5.0)]
    assert cb.relaxed_edf_misses(jobs) == 1
    assert brute_min_misses(jobs, 1) == 1
    assert cb.capacity_bound(jobs) == 0  # bound is valid but not tight here


def test_edf_cannot_start_before_release(concurrency):
    cb.CONCURRENCY = 1
    # If the job could start at t=0 it would finish at 10 and meet d=105.
    # Started at its release (100) it finishes 110 > 105: must miss.
    assert cb.relaxed_edf_misses([(100.0, 105.0, 10.0)]) == 1
    assert cb.relaxed_edf_misses([(100.0, 200.0, 10.0)]) == 0


def test_edf_dispatches_to_soonest_free_machine(concurrency):
    cb.CONCURRENCY = 2
    # m0 busy 0-50, m1 busy 0.5-10.5. Job released at 5 (d=16, w=1) must go
    # to m1 at 10.5 (finish 11.5, on time), not wait for m0 until 50.
    jobs = [(0.0, 100.0, 50.0), (0.5, 100.0, 10.0), (5.0, 16.0, 1.0)]
    assert cb.relaxed_edf_misses(jobs) == 0


def test_edf_terminates_with_large_gaps(concurrency):
    cb.CONCURRENCY = 1
    jobs = [(0.0, 5.0, 1.0), (1000.0, 1005.0, 1.0)]
    assert cb.relaxed_edf_misses(jobs) == 0


def test_edf_idle_machine_used_on_simultaneous_release(concurrency):
    """Two jobs released at the same instant with two idle machines must both
    be dispatched at the release instant (work-conserving parallel EDF).
    FIXED (2026-07-09): the dispatch loop exhausts the ready set at the
    current instant before advancing time, so both deadlines are met."""
    cb.CONCURRENCY = 2
    jobs = [(10.0, 11.5, 1.0), (10.0, 15.5, 5.0)]
    misses = cb.relaxed_edf_misses(jobs)
    assert brute_min_misses(jobs, 2) == 0
    assert misses == 0  # fixed behaviour matches the optimum


# ---------- randomized cross-validation vs brute force --------------------

def _random_instance(rng, n, allow_ties):
    jobs = []
    for _ in range(n):
        r = float(rng.randint(0, 20)) if allow_ties else rng.uniform(0, 20)
        w = float(rng.randint(1, 10))
        d = r + w * rng.choice([0.5, 1.0, 1.0, 2.0]) + rng.randint(0, 15)
        if d <= r:
            d = r + 1.0
        jobs.append((r, d, w))
    return jobs


@pytest.mark.parametrize("C,n,trials,allow_ties", [
    (1, 6, 120, False),
    (1, 6, 120, True),
    (2, 5, 80, False),
    (2, 5, 80, True),
])
def test_bound_and_edf_bracket_brute_force(concurrency, C, n, trials, allow_ties):
    cb.CONCURRENCY = C
    rng = random.Random(1234 + C + n + int(allow_ties))
    for trial in range(trials):
        jobs = _random_instance(rng, n, allow_ties)
        opt = brute_min_misses(jobs, C)
        lb = cb.capacity_bound(jobs)
        ub = cb.relaxed_edf_misses(jobs)
        assert lb <= opt, f"capacity_bound {lb} > optimum {opt} on {jobs}"
        assert ub >= opt, f"relaxed EDF {ub} < optimum {opt} on {jobs}"
        assert lb <= ub


# ---------- reproducibility of published capacity_bound.json --------------

def test_capacity_bound_matches_published_json(concurrency):
    out = json.loads((CGRSP / "Outputs/phase3/capacity_bound.json").read_text())
    for seed in (0, 1, 2):
        jobs = cb.gen_jobs(seed)
        assert cb.capacity_bound(jobs) == out["capacity_bound_misses"][seed]
        assert cb.relaxed_edf_misses(jobs) == out["relaxed_edf_misses"][seed]


# ---------- beta sensitivity: shadowing, leakage, delays -------------------

def test_beta_shadowing_reaches_provisioning_and_no_class_leak():
    from cgrsp.simulator.cloud_fleet import RunPodCloudFleetManager
    from cgrsp.config import ResourceType
    from cgrsp.data.runpod_availability import RunPodAvailabilityModel

    baseline_class_table = dict(RunPodAvailabilityModel.BASE_HIGH_STOCK_PROB)

    def delay_trace(scale, seed=0):
        fm = RunPodCloudFleetManager(
            max_concurrency=5,
            supported_types=ResourceType.supported_gpus(),
            random_seed=seed,
        )
        fm.availability_model.BASE_HIGH_STOCK_PROB = {
            k: v * scale for k, v in RunPodAvailabilityModel.BASE_HIGH_STOCK_PROB.items()
        }
        # Midday peak band (multiplier 0.5) is where LOW becomes reachable.
        times = [43200.0 + 137.0 * i for i in range(200)]
        node_ids = sorted(fm.nodes)
        return [
            fm.request_provisioning(node_ids[i % len(node_ids)], t)
            for i, t in enumerate(times)
        ]

    d_low = delay_trace(0.25)
    d_base = delay_trace(1.00)
    d_base2 = delay_trace(1.00)

    assert d_base == d_base2, "same scale + seed must reproduce identical delays"
    assert d_low != d_base, "scaling BASE_HIGH_STOCK_PROB must change delays"
    assert sum(d_low) > sum(d_base), (
        "shrinking high-stock probability must increase total provisioning delay"
    )
    # Class attribute must be untouched (instance shadowing only).
    assert RunPodAvailabilityModel.BASE_HIGH_STOCK_PROB == baseline_class_table

    # A fresh, unshadowed model must still see the calibrated values.
    fresh = RunPodCloudFleetManager(
        max_concurrency=5,
        supported_types=ResourceType.supported_gpus(),
        random_seed=0,
    )
    assert fresh.availability_model.BASE_HIGH_STOCK_PROB is (
        RunPodAvailabilityModel.BASE_HIGH_STOCK_PROB
    )


def test_beta_scale_one_is_numerically_identity():
    from cgrsp.data.runpod_availability import RunPodAvailabilityModel
    scaled = {k: v * 1.00 for k, v in RunPodAvailabilityModel.BASE_HIGH_STOCK_PROB.items()}
    assert scaled == RunPodAvailabilityModel.BASE_HIGH_STOCK_PROB


def test_metrics_field_mapping_matches_summary_keys():
    """The beta script reads timing_metrics.avg_waiting_time (seconds),
    job_counts.deadline_miss_rate_pct (percent), top-level avg_tardiness_s,
    and job_counts.completed. Verify these keys exist with those semantics."""
    from cgrsp.simulator.metrics import MetricsCollector
    mc = MetricsCollector()
    mc.record_job_submission(job_id=1, submit_time=0.0, deadline=100.0)
    mc.record_job_start(job_id=1, start_time=60.0, node_id=0, resource_type="RTX_3090")
    mc.record_job_completion(job_id=1, completion_time=160.0, execution_cost=0.0)
    m = mc.generate_summary_report()
    assert m["timing_metrics"]["avg_waiting_time"] == pytest.approx(60.0)  # seconds
    assert m["job_counts"]["completed"] == 1
    assert m["job_counts"]["deadline_miss_rate_pct"] == pytest.approx(100.0)
    assert m["avg_tardiness_s"] == pytest.approx(60.0)
