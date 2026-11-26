"""Additional adversarial audit checks for phase3_capacity_bound.py
(audit-only; safe to delete).

Covers what tests/test_phase3_audit.py does not:
  1. Differential fidelity: relaxed_edf_misses vs an independently written
     reference non-preemptive EDF list scheduler on random instances with
     strictly distinct release times.
  2. Floating-point ceil robustness of capacity_bound (documents a latent
     invalidity when demand - capacity lands within float noise of an exact
     multiple of max_w).
  3. The 1e-9 arrival epsilon: near-tie releases (< 1e-9 apart) trigger the
     same idle-machine dispatch delay as exact ties.
  4. Canonical reachability: the minimum inter-arrival gap over all 30
     canonical hectic-day streams is orders of magnitude above 1e-9.
  5. Full 30-seed reproduction of Outputs/phase3/capacity_bound.json.
"""
import heapq
import json
import random
import sys
from pathlib import Path

import pytest

CGRSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CGRSP))

import phase3_capacity_bound as cb  # noqa: E402


@pytest.fixture
def concurrency():
    orig = cb.CONCURRENCY
    yield
    cb.CONCURRENCY = orig


# ---------- independent reference EDF --------------------------------------

def ref_edf_misses(jobs, C):
    """Reference non-preemptive EDF list scheduler, written independently of
    relaxed_edf_misses: at each decision instant t (a machine is free and at
    least one released job is waiting) dispatch the earliest-deadline released
    job onto the earliest-free machine; otherwise advance t to the next event
    (next release or earliest machine-free time). Exact arithmetic on floats,
    same miss rule (finish > d + 1e-9)."""
    order = sorted(jobs)  # by release
    n = len(order)
    machines = [0.0] * C
    ready = []
    i = 0
    t = 0.0
    misses = 0
    while i < n or ready:
        if not ready:
            t = max(t, order[i][0])
        while i < n and order[i][0] <= t:
            r, d, w = order[i]
            heapq.heappush(ready, (d, r, w))
            i += 1
        if not ready:
            continue
        m = min(range(C), key=machines.__getitem__)
        if machines[m] > t:
            nxt = machines[m]
            if i < n:
                nxt = min(nxt, order[i][0])
            t = nxt
            continue
        d, r, w = heapq.heappop(ready)
        machines[m] = t + w
        if t + w > d + 1e-9:
            misses += 1
    return misses


def _random_distinct_instance(rng, n):
    jobs = []
    for _ in range(n):
        r = rng.uniform(0.0, 200.0)          # continuous: ties have prob. 0
        w = rng.uniform(0.5, 15.0)
        d = r + w * rng.uniform(0.6, 3.0) + rng.uniform(0.0, 30.0)
        jobs.append((r, d, w))
    return jobs


@pytest.mark.parametrize("C,trials", [(1, 800), (2, 800), (3, 800), (5, 800)])
def test_relaxed_edf_matches_reference_on_distinct_releases(concurrency, C, trials):
    cb.CONCURRENCY = C
    rng = random.Random(20260709 + C)
    for _ in range(trials):
        n = rng.randint(3, 22)
        jobs = _random_distinct_instance(rng, n)
        got = cb.relaxed_edf_misses(jobs)
        want = ref_edf_misses(jobs, C)
        assert got == want, f"C={C} mismatch got={got} want={want} jobs={jobs}"


# ---------- near-tie releases inside the 1e-9 arrival epsilon ---------------

def test_edf_near_tie_within_epsilon_dispatches_all_idle_machines(concurrency):
    """Releases 5e-10 apart are both swept into `ready` by the r <= t + 1e-9
    arrival scan. FIXED (2026-07-09): the loop now dispatches exhaustively at
    the current instant before advancing time, so both idle machines are used
    and the schedule matches reference EDF (0 misses)."""
    cb.CONCURRENCY = 2
    jobs = [(10.0, 11.5, 1.0), (10.0 + 5e-10, 15.5, 5.0)]
    assert ref_edf_misses(jobs, 2) == 0
    assert cb.relaxed_edf_misses(jobs) == 0  # fixed behaviour


def test_edf_gap_above_epsilon_is_handled_correctly(concurrency):
    cb.CONCURRENCY = 2
    jobs = [(10.0, 11.5, 1.0), (10.0 + 1e-6, 15.5, 5.0)]
    assert ref_edf_misses(jobs, 2) == 0
    assert cb.relaxed_edf_misses(jobs) == 0


# ---------- float-noise ceil: latent capacity_bound invalidity --------------

def test_capacity_bound_float_ceil_overcertifies(concurrency):
    """300 jobs of work 0.1 due at 10 on one machine: exactly 100 fit, so the
    true minimum is 200 misses (relaxed EDF achieves it). Accumulated float
    error makes demand 30.0000000000001..., so ceil(slack/max_w) certifies
    201 > 200: the 'certified' lower bound exceeds both the optimum and the
    script's own upper bracket. Latent (needs demand-capacity within ~1e-13
    relative of an exact multiple of max_w; published bounds are all 0), but
    math.ceil now subtracts a float-noise epsilon so certification is safe."""
    cb.CONCURRENCY = 1
    jobs = [(0.0, 10.0, 0.1)] * 300
    lb = cb.capacity_bound(jobs)
    ub = cb.relaxed_edf_misses(jobs)
    assert ub == 200                      # true optimum, achieved by EDF
    # FIXED (2026-07-09): epsilon-guarded ceil never over-certifies
    assert lb <= 200                      # valid lower bound
    assert lb <= ub                       # bracket holds


# ---------- canonical streams: tie flaw unreachable, JSON reproduces --------

def test_canonical_min_interarrival_gap_far_above_epsilon():
    min_gap = float("inf")
    for seed in range(cb.N_SEEDS):
        subs = sorted(j[0] for j in cb.gen_jobs(seed))
        min_gap = min(
            min_gap,
            min(b - a for a, b in zip(subs, subs[1:])),
        )
    # Measured: 4.45e-05 s over the 30 canonical streams — 4+ orders of
    # magnitude above the 1e-9 arrival-scan epsilon, so the simultaneous-
    # release idle-machine flaw cannot trigger on canonical data.
    assert min_gap > 1e-6, f"min inter-arrival gap {min_gap} too close to 1e-9"


def test_capacity_bound_json_reproduces_all_30_seeds():
    out = json.loads((CGRSP / "Outputs/phase3/capacity_bound.json").read_text())
    assert out["n_seeds"] == cb.N_SEEDS == 30
    for seed in range(cb.N_SEEDS):
        jobs = cb.gen_jobs(seed)
        assert cb.capacity_bound(jobs) == out["capacity_bound_misses"][seed]
        assert cb.relaxed_edf_misses(jobs) == out["relaxed_edf_misses"][seed]
    assert out["capacity_bound_miss_pct_mean"] == 0.0
    assert out["relaxed_edf_miss_pct_mean"] == 0.0
